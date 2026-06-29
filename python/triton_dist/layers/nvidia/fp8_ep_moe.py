################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from triton_dist.function.nvidia.common import init_triton_dist_ep_op, deinit_triton_dist_ep_op
from triton_dist.function.nvidia.ep_moe_fused import TritonDistFusedFp8EpMoeFunction



FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in FP8_DTYPES


def _quantize_fp8_blockwise(tensor: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn,
                            block_k: int = 128):
    assert _is_fp8_dtype(dtype)
    assert tensor.dim() == 2
    assert tensor.shape[-1] % block_k == 0, f"K={tensor.shape[-1]} must be divisible by block_k={block_k}"
    finfo = torch.finfo(dtype)
    m, k = tensor.shape
    tensor_fp32 = tensor.float().reshape(m, k // block_k, block_k)
    amax = tensor_fp32.abs().amax(dim=-1)
    scale = torch.where(amax > 0, amax / finfo.max, torch.ones_like(amax)).to(torch.float32)
    q = torch.clamp(tensor_fp32 / scale.unsqueeze(-1), min=-finfo.max, max=finfo.max).to(dtype)
    return q.reshape(m, k).contiguous(), scale.contiguous()


def _quantize_fp8_weight_blockwise(tensor: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn,
                                   block_n: int = 128, block_k: int = 128):
    assert _is_fp8_dtype(dtype)
    assert tensor.dim() == 3
    e, n, k = tensor.shape
    assert n % block_n == 0, f"N={n} must be divisible by block_n={block_n}"
    assert k % block_k == 0, f"K={k} must be divisible by block_k={block_k}"
    finfo = torch.finfo(dtype)
    tensor_fp32 = tensor.float().reshape(e, n // block_n, block_n, k // block_k, block_k)
    amax = tensor_fp32.abs().amax(dim=(2, 4))
    scale = torch.where(amax > 0, amax / finfo.max, torch.ones_like(amax)).to(torch.float32)
    q = torch.clamp(tensor_fp32 / scale[:, :, None, :, None], min=-finfo.max, max=finfo.max).to(dtype)
    return q.reshape(e, n, k).contiguous(), scale.contiguous()


def _quantize_fp8_rowwise(tensor: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn):
    return _quantize_fp8_blockwise(tensor, dtype)


def _quantize_fp8_last_dim(tensor: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn):
    if tensor.dim() == 3:
        return _quantize_fp8_weight_blockwise(tensor, dtype)
    return _quantize_fp8_blockwise(tensor, dtype)


def _dequantize_fp8_last_dim(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    tensor_fp32 = tensor.float()
    if tensor.dim() == 2 and scale.dim() == 2 and scale.shape[0] == tensor.shape[0]:
        block_k = tensor.shape[1] // scale.shape[1]
        return (tensor_fp32.reshape(tensor.shape[0], scale.shape[1], block_k) * scale[:, :, None]).reshape_as(tensor).to(
            torch.bfloat16)
    if tensor.dim() == 2 and scale.dim() == 2:
        block_n = tensor.shape[0] // scale.shape[0]
        block_k = tensor.shape[1] // scale.shape[1]
        return (tensor_fp32.reshape(scale.shape[0], block_n, scale.shape[1], block_k) *
                scale[:, None, :, None]).reshape_as(tensor).to(torch.bfloat16)
    if tensor.dim() == 3 and scale.dim() == 3:
        block_n = tensor.shape[1] // scale.shape[1]
        block_k = tensor.shape[2] // scale.shape[2]
        return (tensor_fp32.reshape(tensor.shape[0], scale.shape[1], block_n, scale.shape[2], block_k) *
                scale[:, :, None, :, None]).reshape_as(tensor).to(torch.bfloat16)
    raise ValueError(f"Unsupported FP8 dequant shapes: tensor={tuple(tensor.shape)}, scale={tuple(scale.shape)}")



def _all_to_all_single(output: torch.Tensor, input: torch.Tensor, group: dist.ProcessGroup, output_split_sizes=None,
                       input_split_sizes=None):
    if _is_fp8_dtype(input.dtype):
        output_view = output.view(torch.int8)
        input_view = input.view(torch.int8)
    else:
        output_view = output
        input_view = input
    dist.all_to_all_single(output_view, input_view, output_split_sizes=output_split_sizes,
                           input_split_sizes=input_split_sizes, group=group)


class FP8_EP_MoE:
    """Inference-only EP MoE with FP8 dispatch tokens and FP8 expert GEMMs.

    The implementation mirrors the reference EP MoE dispatch/combine order, but stores
    expert weights in FP8 and uses block-wise activation/weight scales for both grouped GEMMs.

    """

    def __init__(self, rank=0, world_size=8, group: Optional[dist.ProcessGroup] = None,
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn, fp8_fast_accum: bool = False):
        assert _is_fp8_dtype(fp8_dtype)
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.fp8_dtype = fp8_dtype
        self.fp8_fast_accum = fp8_fast_accum

        self.top_k = None
        self.num_experts = None
        self.hidden_size = None
        self.gate = None
        self.gate_up_proj = None
        self.gate_up_proj_scale = None
        self.down_proj = None
        self.down_proj_scale = None
        self.gate_up_proj_ref = None
        self.down_proj_ref = None

    def _init_parameters(self, mlp: nn.Module, verbose=False):
        assert mlp.num_experts % self.world_size == 0, "num_experts must be divisible by world_size."
        self.num_experts = mlp.num_experts
        self.top_k = mlp.top_k
        self.gate = mlp.gate.weight.detach().to("cuda")
        self.hidden_size = self.gate.shape[1]
        intermediate_size = mlp.experts[0].gate_proj.weight.detach().shape[0]

        num_local_experts = self.num_experts // self.world_size
        gate_up_proj = torch.empty(num_local_experts, intermediate_size * 2, self.hidden_size, dtype=torch.bfloat16,
                                   device="cuda")
        down_proj = torch.empty(num_local_experts, self.hidden_size, intermediate_size, dtype=torch.bfloat16,
                                device="cuda")

        group_size = self.group.size() if self.group is not None else self.world_size
        start_expert_id = self.rank * self.num_experts // group_size
        end_expert_id = (self.rank + 1) * self.num_experts // group_size

        for e in range(self.num_experts):
            if not (start_expert_id <= e < end_expert_id):
                continue
            local_idx = e - start_expert_id
            gate_proj = mlp.experts[e].gate_proj.weight.detach().to("cuda", dtype=torch.bfloat16, non_blocking=True)
            up_proj = mlp.experts[e].up_proj.weight.detach().to("cuda", dtype=torch.bfloat16, non_blocking=True)
            gate_up_proj[local_idx] = torch.cat((gate_proj, up_proj), dim=0)
            down_proj[local_idx] = mlp.experts[e].down_proj.weight.detach().to("cuda", dtype=torch.bfloat16,
                                                                               non_blocking=True)

        self.gate_up_proj_ref = gate_up_proj
        self.down_proj_ref = down_proj
        self.gate_up_proj, self.gate_up_proj_scale = _quantize_fp8_last_dim(gate_up_proj, self.fp8_dtype)
        self.down_proj, self.down_proj_scale = _quantize_fp8_last_dim(down_proj, self.fp8_dtype)


        assert mlp.experts[0].gate_proj.bias is None, "We do not support bias for now."

        if verbose:
            print(
                f"[RANK {self.rank}] FP8 MoE initialized: gate_up_proj={self.gate_up_proj.shape}, down_proj={self.down_proj.shape}, fp8_dtype={self.fp8_dtype}"
            )

    def _init_ctx(self, EP_GROUP, max_tokens_per_rank):
        import os as _os
        enable_fp8_rs = _os.environ.get("TRITON_DIST_FP8_RS", "0") in ("1", "true", "TRUE", "yes", "YES", "on", "ON")
        init_triton_dist_ep_op(
            EP_GROUP,
            max_tokens_per_rank,
            self.hidden_size,
            self.top_k,
            self.rank,
            self.num_experts,
            self.world_size,
            dtype=self.fp8_dtype,
            weight_dtype=torch.float32,
            num_sm=110,

            num_buffers=1,
            capacity=4.0,
            enable_fp8_rs=enable_fp8_rs,
        )
        torch.cuda.synchronize()
        if self.world_size > 1:
            torch.distributed.barrier(group=self.group)

    def finalize(self):
        deinit_triton_dist_ep_op()

    def _route(self, hidden_states: torch.Tensor):
        router_logits = torch.nn.functional.linear(hidden_states, self.gate)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights, selected_experts.to(torch.int32)

    def _dispatch(self, tokens: torch.Tensor, token_scales: Optional[torch.Tensor], routing_weights: torch.Tensor,
                  selected_experts: torch.Tensor):
        assert self.group is not None
        num_experts_per_rank = self.num_experts // self.group.size()
        batch_size, topk = selected_experts.shape

        tokens_rep = tokens.repeat_interleave(topk, dim=0)
        token_scales_rep = token_scales.repeat_interleave(topk, dim=0) if token_scales is not None else None
        routing_weights_flat = routing_weights.flatten()
        selected_experts_flat = selected_experts.flatten()

        dest_ranks = selected_experts_flat // num_experts_per_rank
        sort_idxs = torch.argsort(dest_ranks, stable=True)

        tokens_to_send = tokens_rep[sort_idxs].contiguous()
        weights_to_send = routing_weights_flat[sort_idxs].contiguous()
        expert_ids_to_send = selected_experts_flat[sort_idxs].contiguous()
        scales_to_send = token_scales_rep[sort_idxs].contiguous() if token_scales_rep is not None else None

        splits_send_tensor = torch.bincount(dest_ranks, minlength=self.group.size()).to(torch.long)
        splits_recv_tensor = torch.empty(self.group.size(), dtype=torch.long, device=tokens.device)
        dist.all_to_all_single(splits_recv_tensor, splits_send_tensor, group=self.group)
        splits_send = splits_send_tensor.cpu().tolist()
        splits_recv = splits_recv_tensor.cpu().tolist()
        total_recv = sum(splits_recv)

        tokens_recv = torch.empty((total_recv, tokens.shape[1]), dtype=tokens.dtype, device=tokens.device)
        weights_recv = torch.empty((total_recv, ), dtype=routing_weights.dtype, device=tokens.device)
        expert_ids_recv = torch.empty((total_recv, ), dtype=torch.int32, device=tokens.device)
        scales_recv = torch.empty((total_recv, ), dtype=torch.float32, device=tokens.device) if scales_to_send is not None else None

        _all_to_all_single(tokens_recv, tokens_to_send, self.group, output_split_sizes=splits_recv,
                           input_split_sizes=splits_send)
        _all_to_all_single(weights_recv, weights_to_send, self.group, output_split_sizes=splits_recv,
                           input_split_sizes=splits_send)
        _all_to_all_single(expert_ids_recv, expert_ids_to_send, self.group, output_split_sizes=splits_recv,
                           input_split_sizes=splits_send)
        if scales_to_send is not None:
            _all_to_all_single(scales_recv, scales_to_send, self.group, output_split_sizes=splits_recv,
                               input_split_sizes=splits_send)

        local_expert_ids = expert_ids_recv % num_experts_per_rank
        local_sort_idxs = torch.argsort(local_expert_ids, stable=True)
        local_expert_ids_sorted = local_expert_ids[local_sort_idxs]
        expert_counts = torch.bincount(local_expert_ids_sorted, minlength=num_experts_per_rank).to(torch.int32)

        return {
            "tokens_sorted": tokens_recv[local_sort_idxs].contiguous(),
            "scales_sorted": scales_recv[local_sort_idxs].contiguous() if scales_recv is not None else None,
            "weights_sorted": weights_recv[local_sort_idxs].contiguous(),
            "local_sort_idxs": local_sort_idxs,
            "expert_counts": expert_counts,
            "batch_size": batch_size,
            "topk": topk,
            "splits_send": splits_send,
            "splits_recv": splits_recv,
            "sort_idxs": sort_idxs,
        }

    def _combine(self, local_output_sorted: torch.Tensor, dispatch_info: dict):
        inv_local_sort_idxs = torch.argsort(dispatch_info["local_sort_idxs"])
        local_output = local_output_sorted[inv_local_sort_idxs].contiguous()
        combined_out_flat = torch.empty((dispatch_info["batch_size"] * dispatch_info["topk"], local_output.shape[1]),
                                        dtype=local_output.dtype, device=local_output.device)
        _all_to_all_single(combined_out_flat, local_output, self.group, output_split_sizes=dispatch_info["splits_send"],
                           input_split_sizes=dispatch_info["splits_recv"])
        inv_sort_idxs = torch.argsort(dispatch_info["sort_idxs"])
        final_out_ordered = combined_out_flat[inv_sort_idxs]
        return final_out_ordered.view(dispatch_info["batch_size"], dispatch_info["topk"], local_output.shape[1]).sum(dim=1)

    @torch.inference_mode()
    def dist_triton_fwd(self, hidden_states: torch.Tensor):
        assert len(hidden_states.size()) == 3
        bsz, seq, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim).to(torch.bfloat16).contiguous()

        routing_weights, selected_experts = self._route(hidden_states)
        hidden_fp8, hidden_scale = _quantize_fp8_rowwise(hidden_states, self.fp8_dtype)
        output = TritonDistFusedFp8EpMoeFunction.apply(
            self.num_experts, routing_weights, selected_experts, hidden_fp8, hidden_scale, self.gate_up_proj,
            self.gate_up_proj_scale, None, None, self.down_proj, self.down_proj_scale, self.group)

        return output.view(bsz, seq, hidden_dim)

    @torch.inference_mode()
    def torch_fwd(self, hidden_states: torch.Tensor):
        assert len(hidden_states.size()) == 3
        bsz, seq, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim).to(torch.bfloat16).contiguous()

        routing_weights, selected_experts = self._route(hidden_states)
        hidden_fp8, hidden_scale = _quantize_fp8_rowwise(hidden_states, self.fp8_dtype)
        dispatch_info = self._dispatch(hidden_fp8, hidden_scale, routing_weights, selected_experts)
        tokens_sorted = _dequantize_fp8_last_dim(dispatch_info["tokens_sorted"], dispatch_info["scales_sorted"])

        weights_sorted = dispatch_info["weights_sorted"]
        expert_counts = dispatch_info["expert_counts"]
        num_experts_per_rank = self.num_experts // self.group.size()

        fc2_out = torch.empty((tokens_sorted.shape[0], hidden_dim), dtype=torch.bfloat16, device=tokens_sorted.device)
        offsets = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int32, device=tokens_sorted.device), expert_counts]),
                               dim=0).cpu().tolist()
        for local_e in range(num_experts_per_rank):
            beg, end = offsets[local_e], offsets[local_e + 1]
            if beg == end:
                continue
            gate_up = _dequantize_fp8_last_dim(self.gate_up_proj[local_e], self.gate_up_proj_scale[local_e])
            down = _dequantize_fp8_last_dim(self.down_proj[local_e], self.down_proj_scale[local_e])
            fc1 = tokens_sorted[beg:end] @ gate_up.t()
            gate, val = fc1.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate.float()) * val.float()
            act = (act * weights_sorted[beg:end].float().unsqueeze(-1)).to(torch.bfloat16)
            act_fp8, act_scale = _quantize_fp8_rowwise(act, self.fp8_dtype)
            act_dequant = _dequantize_fp8_last_dim(act_fp8, act_scale)
            fc2_out[beg:end] = act_dequant @ down.t()

        output = self._combine(fc2_out, dispatch_info)
        return output.view(bsz, seq, hidden_dim)


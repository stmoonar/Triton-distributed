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

import torch
import triton

from triton_dist.kernels.nvidia.group_gemm import (
    GROUP_GEMM_BLOCK_SIZE_M,
    transposed_moe_grouped_gemm,
    build_block_row_idx_info_kernel,
)

import torch.distributed as dist
from triton_dist.kernels.nvidia.swiglu import swiglu_forward, swiglu_backward
from triton_dist.kernels.nvidia.swiglu_quantize_fp8 import swiglu_quantize_fp8

from .common import (custom_fwd, custom_bwd, init_triton_dist_ep_ctx, get_moe_optim_config,
                     get_triton_dist_moe_profile_enabled)


def _quantize_fp8_blockwise(tensor: torch.Tensor, dtype: torch.dtype, block_k: int = 128):
    finfo = torch.finfo(dtype)
    assert tensor.dim() == 2
    assert tensor.shape[-1] % block_k == 0, f"K={tensor.shape[-1]} must be divisible by block_k={block_k}"
    m, k = tensor.shape
    tensor_fp32 = tensor.float().reshape(m, k // block_k, block_k)
    amax = tensor_fp32.abs().amax(dim=-1)
    scale = torch.where(amax > 0, amax / finfo.max, torch.ones_like(amax)).to(torch.float32)
    q = torch.clamp(tensor_fp32 / scale.unsqueeze(-1), min=-finfo.max, max=finfo.max).to(dtype)
    return q.reshape(m, k).contiguous(), scale.contiguous()


class TritonDistFusedEpMoeFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        num_experts: int,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,  # [sum_t, topk]
        hidden_states: torch.Tensor,  # [sum_t, hidden_size]
        fc1_1: torch.Tensor,  # [num_expert, inner_dim, hidden_size]
        fc1_2: torch.Tensor,
        fc2: torch.Tensor,
        ep_group: dist.ProcessGroup,
    ):
        ep_rank = ep_group.rank()
        ep_size = ep_group.size()
        num_experts_per_rank = num_experts // ep_size
        topk = selected_experts.shape[-1]

        triton_dist_ep_ctx = init_triton_dist_ep_ctx(ep_group, topk, num_experts, ep_implementation="mega")

        local_scatter_indices = (selected_experts.flatten().argsort(stable=True).argsort().int().view(
            selected_experts.shape))

        ep_a2a_layout_desc = triton_dist_ep_ctx.ep_op.preprocess(selected_experts, None, local_scatter_indices)

        token_splits_this_rank = ep_a2a_layout_desc.recv_buf_tokens_per_expert[ep_rank]

        assert ep_group.size() <= 8  # only for intra-node
        optim_config = get_moe_optim_config(use_mega=True)
        profile_config = get_triton_dist_moe_profile_enabled()

        if fc1_2 is not None:
            fc1 = torch.cat([fc1_1, fc1_2], dim=1)
        else:
            fc1 = fc1_1

        build_block_row_idx_info_kernel[(optim_config.num_build_sms, )](
            token_splits_this_rank, triton_dist_ep_ctx.split_size_cum_per_expert, triton_dist_ep_ctx.expert_ids,
            triton_dist_ep_ctx.split_size_cum, triton_dist_ep_ctx.tile_num, triton_dist_ep_ctx.tile_num_cum,
            triton_dist_ep_ctx.expert_tile_offset, triton_dist_ep_ctx.num_tiles_total, num_experts_per_rank,
            triton.next_power_of_2(num_experts_per_rank), GROUP_GEMM_BLOCK_SIZE_M, optim_config.num_build_sms)
        (
            dispatch_output_local,
            dispatch_weight_in_buf,
            dispatch_layout_desc,
            fc1_output,
        ) = triton_dist_ep_ctx.ep_op.mega_dispatch_group_gemm(
            # dispatch token
            input=hidden_states,  # torch.Tensor,
            exp_indices=selected_experts,  # torch.Tensor,
            ep_a2a_layout_desc=ep_a2a_layout_desc,  # EPAllToAllLayoutDesc,

            # group gemm
            gemm_weight=fc1,
            gemm_expert_ids=triton_dist_ep_ctx.expert_ids,
            gemm_split_size=token_splits_this_rank,
            gemm_split_size_cum=triton_dist_ep_ctx.split_size_cum,
            gemm_tile_num=triton_dist_ep_ctx.tile_num,
            gemm_tile_num_cum=triton_dist_ep_ctx.tile_num_cum,
            gemm_num_tiles_total=triton_dist_ep_ctx.num_tiles_total,
            gemm_expert_offs=triton_dist_ep_ctx.split_size_cum_per_expert,

            # dispatch token
            weight=routing_weights,
            with_cpy_flag=True,
            comm_buffer_id=0,
            optional_sm=optim_config.num_dispatch_sms,
            num_tail_sms=optim_config.num_tail_sms_in_dispatch,

            # group gemm
            gemm_input_reduce_last_dim=True,
            gemm_weight_reduce_last_dim=True,
            gemm_output_data=None,
            gemm_BLOCK_SIZE_N=triton_dist_ep_ctx.ep_op.FWD_GEMM_BLOCK_SIZE_N,
            gemm_BLOCK_SIZE_K=triton_dist_ep_ctx.ep_op.BF16_FWD_GEMM_BLOCK_SIZE_K,
            gemm_GROUP_SIZE_M=1,
            gemm_num_stages=triton_dist_ep_ctx.ep_op.BF16_FWD_GEMM_NUM_STAGES,

            # common
            use_block_wise_barrier=optim_config.dispatch_use_block_wise_barrier,
            num_warps=optim_config.num_dispatch_warps,
            enable_profiler=profile_config["fwd_dispatch"],
        )

        dispatch_output = dispatch_output_local

        triton_dist_ep_ctx.ep_a2a_layout_desc = dispatch_layout_desc

        swiglu_output, swiglu_ctx = swiglu_forward(fc1_output, scale=dispatch_weight_in_buf.view(-1))
        triton_dist_ep_ctx.swiglu_ctx = swiglu_ctx

        combine_output = triton_dist_ep_ctx.ep_op.mega_group_gemm_combine(
            # group gemm
            gemm_input_data=swiglu_output,
            gemm_weight=fc2,
            gemm_expert_ids=triton_dist_ep_ctx.expert_ids,
            gemm_split_size=token_splits_this_rank,
            gemm_split_size_cum=triton_dist_ep_ctx.split_size_cum,
            gemm_tile_num=triton_dist_ep_ctx.tile_num,
            gemm_tile_num_cum=triton_dist_ep_ctx.tile_num_cum,
            gemm_num_tiles_total=triton_dist_ep_ctx.num_tiles_total,

            # combine token
            ep_a2a_layout_desc=dispatch_layout_desc,

            # group gemm
            gemm_input_reduce_last_dim=True,
            gemm_weight_reduce_last_dim=True,
            gemm_BLOCK_SIZE_N=triton_dist_ep_ctx.ep_op.FWD_GEMM_BLOCK_SIZE_N,
            gemm_BLOCK_SIZE_K=triton_dist_ep_ctx.ep_op.BF16_FWD_GEMM_BLOCK_SIZE_K,
            gemm_GROUP_SIZE_M=1,
            gemm_num_stages=triton_dist_ep_ctx.ep_op.BF16_FWD_GEMM_NUM_STAGES,

            # combine token
            gate_input=None,
            cp_flag=False,
            combine_output=None,
            output_gate=None,
            optional_sm=optim_config.num_combine_sms,
            num_reduce_sms=optim_config.num_reduce_sms_in_combine,
            optional_signal_tensor=None,
            num_warps=optim_config.num_combine_warps,
            combine_mode="fuse_scatter",
            enable_profiler=profile_config["fwd_combine"],
        )

        ctx.triton_dist_ep_ctx = triton_dist_ep_ctx
        ctx.save_for_backward(
            # triton_dist
            dispatch_output,
            fc1_output,
            selected_experts,
            routing_weights,
            fc1_1,
            fc1_2,
            fc2,
        )

        return combine_output

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        (
            # triton dist
            fwd_dispatch_output,
            fc1_output,
            selected_experts,
            routing_weights,
            fc1_1,
            fc1_2,
            fc2,
        ) = ctx.saved_tensors
        defalut_stream = torch.cuda.current_stream()
        stream1 = ctx.triton_dist_ep_ctx.triton_dist_ep_stream

        dy = dy.bfloat16()

        ep_rank = ctx.triton_dist_ep_ctx.ep_group.rank()

        triton_dist_ep_ctx = ctx.triton_dist_ep_ctx
        ep_a2a_layout_desc = triton_dist_ep_ctx.ep_a2a_layout_desc

        assert triton_dist_ep_ctx.ep_group.size() == 8  # only for intra-node
        optim_config = get_moe_optim_config(use_mega=True, is_forward=False)
        profile_config = get_triton_dist_moe_profile_enabled()

        token_splits_this_rank = ep_a2a_layout_desc.recv_buf_tokens_per_expert[ep_rank]

        (
            dispatch_dy_local,
            dispatch_weight_in_buf,
            dispatch_layout_desc,
            grad_swiglu_output,
        ) = triton_dist_ep_ctx.ep_op.mega_dispatch_group_gemm(
            # dispatch token
            input=dy,  # torch.Tensor,
            exp_indices=selected_experts,  # torch.Tensor,
            ep_a2a_layout_desc=ep_a2a_layout_desc,  # EPAllToAllLayoutDesc,

            # group gemm
            gemm_weight=fc2,
            gemm_expert_ids=triton_dist_ep_ctx.expert_ids,
            gemm_split_size=token_splits_this_rank,
            gemm_split_size_cum=triton_dist_ep_ctx.split_size_cum,
            gemm_tile_num=triton_dist_ep_ctx.tile_num,
            gemm_tile_num_cum=triton_dist_ep_ctx.tile_num_cum,
            gemm_num_tiles_total=triton_dist_ep_ctx.num_tiles_total,
            gemm_expert_offs=triton_dist_ep_ctx.split_size_cum_per_expert,

            # dispatch token
            weight=routing_weights,
            with_cpy_flag=True,
            comm_buffer_id=0,
            optional_sm=optim_config.num_dispatch_sms,
            num_tail_sms=optim_config.num_tail_sms_in_dispatch,

            # group gemm
            gemm_input_reduce_last_dim=True,
            gemm_weight_reduce_last_dim=False,
            gemm_output_data=None,
            gemm_BLOCK_SIZE_N=triton_dist_ep_ctx.ep_op.FWD_GEMM_BLOCK_SIZE_N,
            gemm_BLOCK_SIZE_K=64,
            gemm_GROUP_SIZE_M=1,
            gemm_num_stages=3,

            # common
            use_block_wise_barrier=optim_config.dispatch_use_block_wise_barrier,
            num_warps=optim_config.num_dispatch_warps,
            enable_profiler=profile_config["bwd_dispatch"],
            profile_file_name="mega_bwd_dispatch_group_gemm",
        )
        dispatch_dy = dispatch_dy_local

        grad_fc1_output, grad_gate = swiglu_backward(
            grad_swiglu_output,
            fc1_output,
            scale=dispatch_weight_in_buf.view(-1),
            ctx=triton_dist_ep_ctx.swiglu_ctx,
        )

        if fc2.requires_grad:
            recompute_swiglu_output, _ = swiglu_forward(fc1_output, scale=dispatch_weight_in_buf.view(-1))

            grad_fc2 = transposed_moe_grouped_gemm(
                grad_output=dispatch_dy,
                original_input=recompute_swiglu_output,
                split_size=token_splits_this_rank,
                split_size_cum_per_expert=triton_dist_ep_ctx.split_size_cum_per_expert,
                BLOCK_SIZE_M=64,
                BLOCK_SIZE_N=128,
                BLOCK_SIZE_K=256,
                GROUP_SIZE_M=4,
                num_warps=optim_config.num_group_gemm_warps,
                num_stages=3,
            )

        if fc1_2 is not None:
            fc1 = torch.cat([fc1_1, fc1_2], dim=1)
        else:
            fc1 = fc1_1

        (
            combine_grad_input,
            combine_grad_gate,
            # grad_fc1,
        ) = triton_dist_ep_ctx.ep_op.mega_group_gemm_combine(
            # group gemm
            gemm_input_data=grad_fc1_output,
            gemm_weight=fc1,
            gemm_expert_ids=triton_dist_ep_ctx.expert_ids,
            gemm_split_size=token_splits_this_rank,
            gemm_split_size_cum=triton_dist_ep_ctx.split_size_cum,
            gemm_tile_num=triton_dist_ep_ctx.tile_num,
            gemm_tile_num_cum=triton_dist_ep_ctx.tile_num_cum,
            gemm_num_tiles_total=triton_dist_ep_ctx.num_tiles_total,

            # combine token
            ep_a2a_layout_desc=dispatch_layout_desc,

            # group gemm
            gemm_input_reduce_last_dim=True,
            gemm_weight_reduce_last_dim=False,
            gemm_BLOCK_SIZE_N=triton_dist_ep_ctx.ep_op.FWD_GEMM_BLOCK_SIZE_N,
            gemm_BLOCK_SIZE_K=64,
            gemm_GROUP_SIZE_M=1,
            gemm_num_stages=3,

            # combine token
            gate_input=grad_gate,
            cp_flag=True,
            combine_output=None,
            output_gate=None,
            optional_sm=optim_config.num_combine_sms,
            num_reduce_sms=optim_config.num_reduce_sms_in_combine,
            optional_signal_tensor=None,
            num_warps=optim_config.num_combine_warps,
            combine_mode="fuse_scatter",

            # transposed group gemm params
            # grad_output=grad_fc1_output,
            # orig_input=fwd_dispatch_output,
            # grad_weight=None,
            # split_size_cum_per_expert=triton_dist_ep_ctx.split_size_cum_per_expert,
            # grad_BLOCK_SIZE_M=64,
            # grad_BLOCK_SIZE_N=256,
            # grad_BLOCK_SIZE_K=128,
            # grad_GROUP_SIZE_M=4,
            enable_profiler=profile_config["bwd_combine"],
            profile_file_name="mega_bwd_group_gemm_combine",
        )

        grad_fc1 = transposed_moe_grouped_gemm(
            grad_output=grad_fc1_output,
            original_input=fwd_dispatch_output,
            split_size=token_splits_this_rank,
            split_size_cum_per_expert=triton_dist_ep_ctx.split_size_cum_per_expert,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=256,
            GROUP_SIZE_M=4,
            num_warps=optim_config.num_group_gemm_warps,
            num_stages=3,
            persistent="dynamic",
            sm_margin=0,
        )

        if fc1_2 is not None:
            grad_fc1_1, grad_fc1_2 = torch.chunk(grad_fc1, 2, dim=1)
        else:
            grad_fc1_1 = grad_fc1
            grad_fc1_2 = None

        defalut_stream.wait_stream(stream1)

        return None, combine_grad_gate.bfloat16(), None, combine_grad_input, grad_fc1_1, grad_fc1_2, grad_fc2, None


class TritonDistFusedFp8EpMoeFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        num_experts: int,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        fc1_1: torch.Tensor,
        fc1_1_scale: torch.Tensor,
        fc1_2: torch.Tensor,
        fc1_2_scale: torch.Tensor,
        fc2: torch.Tensor,
        fc2_scale: torch.Tensor,
        ep_group: dist.ProcessGroup,
    ):
        ep_rank = ep_group.rank()
        ep_size = ep_group.size()
        num_experts_per_rank = num_experts // ep_size
        topk = selected_experts.shape[-1]
        fp8_dtype = fc1_1.dtype
        assert fp8_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
        assert fc2.dtype == fp8_dtype
        assert hidden_states.dtype == fp8_dtype
        assert hidden_states_scale.dtype == torch.float32
        assert fc1_1_scale.dtype == torch.float32
        assert fc2_scale.dtype == torch.float32

        triton_dist_ep_ctx = init_triton_dist_ep_ctx(ep_group, topk, num_experts, ep_implementation="mega")
        local_scatter_indices = (selected_experts.flatten().argsort(stable=True).argsort().int().view(
            selected_experts.shape))
        ep_a2a_layout_desc = triton_dist_ep_ctx.ep_op.preprocess(selected_experts, None, local_scatter_indices)
        token_splits_this_rank = ep_a2a_layout_desc.recv_buf_tokens_per_expert[ep_rank]

        assert ep_group.size() <= 8
        optim_config = get_moe_optim_config(use_mega=True)
        profile_config = get_triton_dist_moe_profile_enabled()
        fp8_dispatch_warps = min(optim_config.num_dispatch_warps, 4)
        fp8_combine_warps = min(optim_config.num_combine_warps, 4)

        if fc1_2 is not None:
            fc1 = torch.cat([fc1_1, fc1_2], dim=1)
            fc1_scale = torch.cat([fc1_1_scale, fc1_2_scale], dim=1)
        else:
            fc1 = fc1_1
            fc1_scale = fc1_1_scale

        hidden_states_fp8 = hidden_states.contiguous()
        hidden_states_scale = hidden_states_scale.contiguous()
        fc1_scale = fc1_scale.contiguous()
        fc2_scale = fc2_scale.contiguous()

        build_block_row_idx_info_kernel[(optim_config.num_build_sms, )](
            token_splits_this_rank, triton_dist_ep_ctx.split_size_cum_per_expert, triton_dist_ep_ctx.expert_ids,
            triton_dist_ep_ctx.split_size_cum, triton_dist_ep_ctx.tile_num, triton_dist_ep_ctx.tile_num_cum,
            triton_dist_ep_ctx.expert_tile_offset, triton_dist_ep_ctx.num_tiles_total, num_experts_per_rank,
            triton.next_power_of_2(num_experts_per_rank), GROUP_GEMM_BLOCK_SIZE_M, optim_config.num_build_sms)

        _, dispatch_weight_in_buf, dispatch_layout_desc, fc1_output = triton_dist_ep_ctx.ep_op.mega_dispatch_group_gemm(
            input=hidden_states_fp8,
            exp_indices=selected_experts,
            ep_a2a_layout_desc=ep_a2a_layout_desc,
            gemm_weight=fc1,
            gemm_input_scale=hidden_states_scale,
            gemm_weight_scale=fc1_scale,
            gemm_expert_ids=triton_dist_ep_ctx.expert_ids,
            gemm_split_size=token_splits_this_rank,
            gemm_split_size_cum=triton_dist_ep_ctx.split_size_cum,
            gemm_tile_num=triton_dist_ep_ctx.tile_num,
            gemm_tile_num_cum=triton_dist_ep_ctx.tile_num_cum,
            gemm_num_tiles_total=triton_dist_ep_ctx.num_tiles_total,
            gemm_expert_offs=triton_dist_ep_ctx.split_size_cum_per_expert,
            weight=routing_weights,
            with_cpy_flag=True,
            comm_buffer_id=0,
            optional_sm=optim_config.num_dispatch_sms,
            num_tail_sms=optim_config.num_tail_sms_in_dispatch,
            gemm_input_reduce_last_dim=True,
            gemm_weight_reduce_last_dim=True,
            gemm_output_data=None,
            gemm_BLOCK_SIZE_N=triton_dist_ep_ctx.ep_op.FP8_FWD_GEMM_BLOCK_SIZE_N,
            gemm_BLOCK_SIZE_K=triton_dist_ep_ctx.ep_op.FP8_FWD_GEMM_BLOCK_SIZE_K,
            gemm_GROUP_SIZE_M=1,
            gemm_num_stages=triton_dist_ep_ctx.ep_op.FP8_FWD_GEMM_NUM_STAGES,
            use_block_wise_barrier=optim_config.dispatch_use_block_wise_barrier,
            num_warps=fp8_dispatch_warps,
            enable_profiler=profile_config["fwd_dispatch"],
            profile_file_name="mega_fp8_fwd_dispatch_group_gemm",
        )

        triton_dist_ep_ctx.ep_a2a_layout_desc = dispatch_layout_desc
        swiglu_output_fp8, swiglu_output_scale = swiglu_quantize_fp8(
            fc1_output, routing_weight=dispatch_weight_in_buf.view(-1), fp8_dtype=fp8_dtype, block_k=128
        )

        combine_output = triton_dist_ep_ctx.ep_op.mega_group_gemm_combine(
            gemm_input_data=swiglu_output_fp8,
            gemm_weight=fc2,
            gemm_input_scale=swiglu_output_scale,
            gemm_weight_scale=fc2_scale,
            gemm_expert_ids=triton_dist_ep_ctx.expert_ids,
            gemm_split_size=token_splits_this_rank,
            gemm_split_size_cum=triton_dist_ep_ctx.split_size_cum,
            gemm_tile_num=triton_dist_ep_ctx.tile_num,
            gemm_tile_num_cum=triton_dist_ep_ctx.tile_num_cum,
            gemm_num_tiles_total=triton_dist_ep_ctx.num_tiles_total,
            ep_a2a_layout_desc=dispatch_layout_desc,
            gemm_input_reduce_last_dim=True,
            gemm_weight_reduce_last_dim=True,
            gemm_BLOCK_SIZE_N=triton_dist_ep_ctx.ep_op.FP8_FWD_GEMM_BLOCK_SIZE_N,
            gemm_BLOCK_SIZE_K=triton_dist_ep_ctx.ep_op.FP8_FWD_GEMM_BLOCK_SIZE_K,
            gemm_GROUP_SIZE_M=1,
            gemm_num_stages=triton_dist_ep_ctx.ep_op.FP8_FWD_GEMM_NUM_STAGES,
            gate_input=None,
            cp_flag=False,
            combine_output=None,
            output_gate=None,
            optional_sm=optim_config.num_combine_sms,
            num_reduce_sms=optim_config.num_reduce_sms_in_combine,
            optional_signal_tensor=None,
            num_warps=fp8_combine_warps,
            combine_mode="fuse_scatter",
            enable_profiler=profile_config["fwd_combine"],
            profile_file_name="mega_fp8_fwd_group_gemm_combine",
        )
        return combine_output

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        raise NotImplementedError("TritonDistFusedFp8EpMoeFunction currently supports inference forward only.")

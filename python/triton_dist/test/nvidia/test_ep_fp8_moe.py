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

import argparse
import os
from types import SimpleNamespace

import torch

from triton_dist.layers.nvidia.fp8_ep_moe import FP8_EP_MoE
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import finalize_distributed, initialize_distributed, dist_print, rand_tensor


DTYPE_MAP = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--ffn_dim", type=int, default=128)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--fp8_dtype", type=str, default="float8_e4m3fn", choices=DTYPE_MAP.keys())
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_dummy_mlp(num_experts: int, topk: int, hidden_dim: int, ffn_dim: int):
    gate = SimpleNamespace(weight=(torch.randn(num_experts, hidden_dim, dtype=torch.bfloat16) * 0.1))
    experts = []
    for _ in range(num_experts):
        experts.append(
            SimpleNamespace(
                gate_proj=SimpleNamespace(weight=(torch.randn(ffn_dim, hidden_dim, dtype=torch.bfloat16) * 0.1),
                                          bias=None),
                up_proj=SimpleNamespace(weight=(torch.randn(ffn_dim, hidden_dim, dtype=torch.bfloat16) * 0.1),
                                        bias=None),
                down_proj=SimpleNamespace(weight=(torch.randn(hidden_dim, ffn_dim, dtype=torch.bfloat16) * 0.1),
                                          bias=None),
            ))
    return SimpleNamespace(num_experts=num_experts, top_k=topk, gate=gate, experts=experts)


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ep_group = initialize_distributed(args.seed)
    assert args.num_experts % world_size == 0

    torch.manual_seed(args.seed)
    mlp = make_dummy_mlp(args.num_experts, args.topk, args.hidden_dim, args.ffn_dim)
    moe = FP8_EP_MoE(rank=rank, world_size=world_size, group=ep_group, fp8_dtype=DTYPE_MAP[args.fp8_dtype])
    moe._init_parameters(mlp, verbose=(rank == 0))

    torch.manual_seed(args.seed + rank)
    x = rand_tensor([args.bsz, args.seq_len, args.hidden_dim], dtype=torch.bfloat16)

    torch_ref = moe.torch_fwd(x)
    moe._init_ctx(EP_GROUP=ep_group, max_tokens_per_rank=args.bsz * args.seq_len)
    triton_out = moe.dist_triton_fwd(x)

    assert_allclose(triton_out, torch_ref, atol=2e-1, rtol=2e-1)
    dist_print("FP8 EP MoE check passed", need_sync=True, allowed_ranks=[0])

    moe.finalize()
    finalize_distributed()


if __name__ == "__main__":
    main()

"""
Benchmark: tl.dot_scaled (MXFP8, hardware-native microscaling) vs manual scale
 
Tests the Blackwell SM120a native microscaling dot product:
- tl.dot_scaled with E8M0 block scales (K-group=32)
- vs manual FP8 + scale broadcast multiply
- vs BF16 baseline

Key: MXFP8 uses K-group=32 (every 32 K-elements share one E8M0 scale factor)
Scale layout for dot_scaled:
  lhs_scale: [BLOCK_M, BLOCK_K // 32], dtype=uint8 (E8M0)
  rhs_scale: [BLOCK_N, BLOCK_K // 32], dtype=uint8 (E8M0)

Usage:
    python -m triton_dist.kernels.nvidia.benchmark_dot_scaled
"""

import traceback

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import numpy as np




# ============================================================
# Helper: Create E8M0 scale factors
# E8M0: 8-bit exponent-only format, value = 2^(exp - 127)
# Stored as uint8. A value of 127 means scale=1.0
# ============================================================
def float_to_e8m0(f: torch.Tensor) -> torch.Tensor:
    """Convert float32 scale values to E8M0 (uint8).
    E8M0: value = 2^(stored_byte - 127)
    So stored_byte = log2(value) + 127
    For scale=1.0 -> stored=127
    """
    # Clamp to valid E8M0 range [2^-127, 2^127]
    f = f.clamp(min=2**-127, max=2**127)
    # E8M0 stores the biased exponent
    exponent = torch.log2(f).round().to(torch.int32) + 127
    exponent = exponent.clamp(0, 254).to(torch.uint8)
    return exponent


def e8m0_to_float(e: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 (uint8) back to float32."""
    return torch.pow(2.0, e.to(torch.float32) - 127.0)


# ============================================================
# Kernel 1: BF16 GEMM baseline
# ============================================================
@triton.jit
def matmul_bf16_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_SIZE_K < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k * BLOCK_SIZE_K < K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================
# Kernel 2: FP8 GEMM without scale (pure throughput test)
# ============================================================
@triton.jit
def matmul_fp8_noscale_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_SIZE_K < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k * BLOCK_SIZE_K < K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================
# Kernel 3: FP8 GEMM with manual block-wise scale (current production path)
# Scale granularity: per 128 K-elements (FP8_BLOCK_K=128)
# ============================================================
@triton.jit
def matmul_fp8_manual_scale_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
    b_scale_ptrs = b_scale_ptr + offs_n * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_SIZE_K < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k * BLOCK_SIZE_K < K, other=0.0)
        dot = tl.dot(a, b)

        k_scale_idx = (k * BLOCK_SIZE_K) // SCALE_BLOCK_K
        a_scale = tl.load(a_scale_ptrs + k_scale_idx * stride_ask).to(tl.float32)
        b_scale = tl.load(b_scale_ptrs + k_scale_idx * stride_bsk).to(tl.float32)
        accumulator += dot * a_scale[:, None] * b_scale[None, :]

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================
# Kernel 4: MXFP8 with tl.dot_scaled (hardware-native microscaling)
# K-group = 32, scale in E8M0 (uint8)
# lhs_scale: [BLOCK_M, BLOCK_K // 32]
# rhs_scale: [BLOCK_N, BLOCK_K // 32]  (NOTE: shape is [N, K//32], NOT transposed!)
#
# Follow Triton's block-scaled tutorial pattern:
#   physical B is stored as [N, K], load [BLOCK_N, BLOCK_K], then pass b.T as logical [K, N].
# This keeps the RHS operand/scale association explicit for tl.dot_scaled lowering.
# ============================================================
@triton.jit
def matmul_mxfp8_dot_scaled_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # B is physically [N, K]
    stride_cm, stride_cn,
    stride_asm, stride_ask,  # a_scale strides: [M, K//32]
    stride_bsn, stride_bsk,  # b_scale strides: [N, K//32]
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,  # = 32 for MXFP8
):

    """
    MXFP8 GEMM using tl.dot_scaled for hardware-native microscaling.

    Data layout:
      A: [M, K] in float8_e4m3fn (row-major)
      B: [N, K] in float8_e4m3fn; kernel passes B tile transposed as logical rhs [K, N]
      a_scale: [M, K//32] in uint8 (E8M0)
      b_scale: [N, K//32] in uint8 (E8M0) - NOTE: [N, K//32] not [K//32, N]!

    IMPORTANT constraints:
      - K must be divisible by BLOCK_SIZE_K (no mask on dot_scaled inputs)
      - BLOCK_SIZE_K must be divisible by 32 (VEC_SIZE)
    """
    tl.static_assert(BLOCK_SIZE_K == BLOCK_SIZE_SCALE_K * VEC_SIZE)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)


    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A is [M, K]: load tile [BLOCK_M, BLOCK_K]
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B is physically [N, K]: load tile [BLOCK_N, BLOCK_K], pass b.T to dot_scaled
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    # Scale pointers
    # a_scale: [M, K//32], load [BLOCK_M, BLOCK_K//32] per iteration
    # b_scale: [N, K//32], load [BLOCK_N, BLOCK_K//32] per iteration
    num_scale_per_block: tl.constexpr = BLOCK_SIZE_SCALE_K  # e.g., 128//32 = 4
    offs_scale_k = tl.arange(0, BLOCK_SIZE_SCALE_K)


    a_scale_base = a_scale_ptr + offs_m[:, None] * stride_asm
    b_scale_base = b_scale_ptr + offs_n[:, None] * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_iters = K // BLOCK_SIZE_K  # K must be divisible by BLOCK_SIZE_K
    for k in range(0, num_k_iters):
        # Load A tile [BLOCK_M, BLOCK_K] and physical B tile [BLOCK_N, BLOCK_K]
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # Load scale tiles
        # a_scale: [BLOCK_M, num_scale_per_block] (uint8, E8M0)
        scale_k_offset = k * num_scale_per_block
        a_scale = tl.load(
            a_scale_base + (offs_scale_k[None, :] + scale_k_offset) * stride_ask
        )
        # b_scale: [BLOCK_N, num_scale_per_block] (uint8, E8M0)
        b_scale = tl.load(
            b_scale_base + (offs_scale_k[None, :] + scale_k_offset) * stride_bsk
        )

        # lhs: [BLOCK_M, BLOCK_K], lhs_scale: [BLOCK_M, BLOCK_K//32]
        # rhs: b.T -> [BLOCK_K, BLOCK_N], rhs_scale: [BLOCK_N, BLOCK_K//32]
        accumulator = tl.dot_scaled(
            a, a_scale, "e4m3",
            b.T, b_scale, "e4m3",
            accumulator
        )

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))



# ============================================================
# Kernel 5: MXFP8 with tl.dot_scaled - TMA version (using TensorDescriptor)
# This follows the official tutorial pattern more closely
# ============================================================
@triton.jit
def matmul_mxfp8_dot_scaled_tma_kernel(
    a_desc,
    a_scale_desc,
    b_desc,
    b_scale_desc,
    c_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    REP_M: tl.constexpr,
    REP_N: tl.constexpr,
    REP_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    MXFP8 GEMM using tl.dot_scaled with TensorDescriptor data loads and official
    NVIDIA packed 5D scale layout.

    Physical layouts:
      A: [M, K]
      B: [N, K], passed as b.T to dot_scaled logical rhs [K, N]
      scale: [1, ceil(dim/128), K//VEC_SIZE//4, 2, 256]
    """
    tl.static_assert(BLOCK_SIZE_M == REP_M * 128)
    tl.static_assert(BLOCK_SIZE_N == REP_N * 128)
    tl.static_assert(BLOCK_SIZE_K == REP_K * VEC_SIZE * 4)

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    offs_scale_m = pid_m * REP_M
    offs_scale_n = pid_n * REP_N
    offs_scale_k = 0

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=NUM_STAGES):
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        a_scale = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        b_scale = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        a_scale = a_scale.reshape(REP_M, REP_K, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(
            BLOCK_SIZE_M, BLOCK_SIZE_K // VEC_SIZE
        )
        b_scale = b_scale.reshape(REP_N, REP_K, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(
            BLOCK_SIZE_N, BLOCK_SIZE_K // VEC_SIZE
        )

        accumulator = tl.dot_scaled(
            a, a_scale, "e4m3",
            b.T, b_scale, "e4m3",
            accumulator
        )

        offs_k += BLOCK_SIZE_K
        offs_scale_k += REP_K

    c_desc.store([offs_am, offs_bn], accumulator.to(tl.bfloat16))



# ============================================================
# Benchmark runner
# ============================================================
def run_benchmark(M, N, K, configs, warmup=15, rep=80):
    """Run benchmark for given configs, return results dict."""
    results = {}

    for name, cfg in configs.items():
        try:
            kernel_fn = cfg["kernel"]
            block_m = cfg["BLOCK_SIZE_M"]
            block_n = cfg["BLOCK_SIZE_N"]
            block_k = cfg["BLOCK_SIZE_K"]
            num_warps = cfg["num_warps"]
            num_stages = cfg["num_stages"]
            variant = cfg["variant"]

            # Create tensors
            if variant == "bf16":
                a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
                c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
                grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

                def run_fn(kernel_fn=kernel_fn, a=a, b=b, c=c, M=M, N=N, K=K,
                           grid=grid, block_m=block_m, block_n=block_n, block_k=block_k,
                           num_warps=num_warps, num_stages=num_stages):
                    kernel_fn[grid](
                        a, b, c, M, N, K,
                        a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
                        num_warps=num_warps, num_stages=num_stages,
                    )

            elif variant == "fp8_noscale":
                a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
                grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

                def run_fn(kernel_fn=kernel_fn, a=a, b=b, c=c, M=M, N=N, K=K,
                           grid=grid, block_m=block_m, block_n=block_n, block_k=block_k,
                           num_warps=num_warps, num_stages=num_stages):
                    kernel_fn[grid](
                        a, b, c, M, N, K,
                        a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
                        num_warps=num_warps, num_stages=num_stages,
                    )

            elif variant == "fp8_manual_scale":
                a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
                scale_block_k = cfg.get("SCALE_BLOCK_K", 128)
                num_k_blocks = triton.cdiv(K, scale_block_k)
                # Per-token, per-K-block scale (float32)
                a_scale = torch.ones(M, num_k_blocks, device="cuda", dtype=torch.float32)
                b_scale = torch.ones(N, num_k_blocks, device="cuda", dtype=torch.float32)
                grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

                def run_fn(kernel_fn=kernel_fn, a=a, b=b, c=c, a_scale=a_scale, b_scale=b_scale,
                           M=M, N=N, K=K, grid=grid, block_m=block_m, block_n=block_n, block_k=block_k,
                           num_warps=num_warps, num_stages=num_stages, scale_block_k=scale_block_k):
                    kernel_fn[grid](
                        a, b, c, a_scale, b_scale,
                        M, N, K,
                        a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        a_scale.stride(0), a_scale.stride(1),
                        b_scale.stride(0), b_scale.stride(1),
                        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
                        SCALE_BLOCK_K=scale_block_k,
                        num_warps=num_warps, num_stages=num_stages,
                    )

            elif variant == "mxfp8_dot_scaled":
                # MXFP8 with tl.dot_scaled, TensorDescriptor data loads, and official
                # NVIDIA packed 5D scale layout.
                # A: [M, K]
                # B: [N, K], passed as b.T to dot_scaled logical rhs=[K,N]
                # scale: [1, dim//128, K//32//4, 2, 256] in uint8 E8M0
                vec_size = 32
                if block_m % 128 != 0 or block_n % 128 != 0 or block_k % (vec_size * 4) != 0:
                    results[name] = {
                        "error": "SKIP: packed scale path requires BLOCK_M/BLOCK_N multiples of 128 and BLOCK_K multiple of 128"
                    }
                    continue
                if K % block_k != 0:
                    results[name] = {"error": "SKIP: TensorDescriptor path currently requires K divisible by BLOCK_K"}
                    continue

                # Packed scale layout has 128-row granularity. For skinny M (e.g. M=64),
                # benchmark a padded tensor and report effective TFLOPS using the original M/N.
                m_kernel = triton.cdiv(M, block_m) * block_m
                n_kernel = triton.cdiv(N, block_n) * block_n

                a = torch.randn(m_kernel, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                b_nk = torch.randn(n_kernel, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                c = torch.empty(m_kernel, n_kernel, device="cuda", dtype=torch.bfloat16)

                rep_m = block_m // 128
                rep_n = block_n // 128
                rep_k = block_k // vec_size // 4
                scale_k_chunks = K // vec_size // 4

                # Packed scale layout from Triton's block-scaled matmul tutorial.
                # E8M0 scale = 127 means scale factor = 1.0.
                a_scale = torch.full((1, m_kernel // 128, scale_k_chunks, 2, 256), 127, device="cuda", dtype=torch.uint8)
                b_scale = torch.full((1, n_kernel // 128, scale_k_chunks, 2, 256), 127, device="cuda", dtype=torch.uint8)

                a_desc = TensorDescriptor(a, a.shape, a.stride(), [block_m, block_k])
                b_desc = TensorDescriptor(b_nk, b_nk.shape, b_nk.stride(), [block_n, block_k])
                c_desc = TensorDescriptor(c, c.shape, c.stride(), [block_m, block_n])

                scale_block_shape_a = [1, rep_m, rep_k, 2, 256]
                scale_block_shape_b = [1, rep_n, rep_k, 2, 256]
                a_scale_desc = TensorDescriptor(a_scale, a_scale.shape, a_scale.stride(), scale_block_shape_a)
                b_scale_desc = TensorDescriptor(b_scale, b_scale.shape, b_scale.stride(), scale_block_shape_b)

                grid = (triton.cdiv(m_kernel, block_m) * triton.cdiv(n_kernel, block_n), )

                def run_fn(kernel_fn=kernel_fn, a_desc=a_desc, a_scale_desc=a_scale_desc,
                           b_desc=b_desc, b_scale_desc=b_scale_desc, c_desc=c_desc,
                           M=m_kernel, N=n_kernel, K=K, grid=grid,
                           block_m=block_m, block_n=block_n, block_k=block_k,

                           rep_m=rep_m, rep_n=rep_n, rep_k=rep_k,
                           num_warps=num_warps, num_stages=num_stages, vec_size=vec_size):
                    kernel_fn[grid](
                        a_desc, a_scale_desc, b_desc, b_scale_desc, c_desc,
                        M, N, K,
                        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
                        VEC_SIZE=vec_size,
                        REP_M=rep_m, REP_N=rep_n, REP_K=rep_k,
                        NUM_STAGES=num_stages,
                        num_warps=num_warps,
                    )



            else:
                print(f"  [SKIP] Unknown variant: {variant}")
                continue

            # Warmup
            for _ in range(warmup):
                run_fn()
            torch.cuda.synchronize()

            # Benchmark
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

            for i in range(rep):
                start_events[i].record()
                run_fn()
                end_events[i].record()

            torch.cuda.synchronize()

            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            avg_ms = sum(times) / len(times)
            min_ms = min(times)

            flops = 2.0 * M * N * K
            avg_tflops = flops / (avg_ms * 1e-3) / 1e12
            peak_tflops = flops / (min_ms * 1e-3) / 1e12

            results[name] = {
                "avg_ms": avg_ms,
                "min_ms": min_ms,
                "avg_tflops": avg_tflops,
                "peak_tflops": peak_tflops,
            }

        except Exception as e:
            tb = traceback.format_exc()
            first_line = str(e).strip().splitlines()[0] if str(e).strip() else ""
            err = f"{type(e).__name__}: {first_line}" if first_line else type(e).__name__
            results[name] = {"error": err, "traceback": tb}



    return results


def main():
    compute_capability = torch.cuda.get_device_capability()

    print("=" * 100)
    print("MXFP8 dot_scaled Benchmark (Blackwell SM120a)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {compute_capability}")
    print(f"Triton: {getattr(triton, '__version__', 'unknown')}")
    if compute_capability[0] == 12:
        print("Note: SM120/SM120a tl.dot_scaled support depends on your Triton/CUDA build.")
    print("=" * 100)
    print()
    print("Key: tl.dot_scaled uses hardware-native microscaling (K-group=32, E8M0 scale)")
    print("     MXFP8 dot_scaled uses TensorDescriptor + NVIDIA packed 5D scale layout")
    print("     Manual scale uses software float32 broadcast multiply (K-group=128)")

    print()


    # Problem sizes (MoE-relevant)
    problem_sizes = [
        (64, 7168, 7168),
        (128, 7168, 7168),
        (64, 18432, 7168),
        (128, 18432, 7168),
        (256, 18432, 7168),
        (64, 7168, 18432),
        (128, 7168, 18432),
    ]

    # Tile configs to try: (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
    tile_configs = [
        # Good configs from previous benchmark for 101KB shmem
        (64, 128, 128, 4, 3),
        (64, 128, 128, 4, 2),
        (128, 128, 128, 4, 3),
        (128, 128, 128, 4, 2),
        (128, 256, 128, 4, 3),
        (128, 256, 128, 4, 2),
        (64, 256, 128, 4, 2),

        (64, 128, 64, 4, 3),
        (64, 256, 64, 4, 3),
        (64, 256, 64, 4, 2),
        (128, 128, 64, 4, 3),
        (128, 128, 64, 4, 2),
        (128, 256, 64, 4, 2),
    ]

    printed_dot_scaled_traceback = False

    for M, N, K in problem_sizes:

        print(f"\n{'═' * 100}")
        print(f"Problem: M={M}, N={N}, K={K}  |  FLOPS={2.0*M*N*K/1e9:.2f} GFLOPS")
        print(f"{'═' * 100}")
        print(f"{'Config':<60} {'Avg(ms)':<9} {'Min(ms)':<9} {'Avg TF':<9} {'Peak TF':<9}")
        print(f"{'─' * 100}")

        for block_m, block_n, block_k, nw, ns in tile_configs:
            if block_n > N:
                continue

            tag = f"[{block_m}x{block_n}x{block_k}, w{nw}, s{ns}]"

            configs = {}

            if block_m <= M:
                # 1. BF16 baseline
                configs[f"BF16              {tag}"] = {
                    "kernel": matmul_bf16_kernel,
                    "variant": "bf16",
                    "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                    "num_warps": nw, "num_stages": ns,
                }

                # 2. FP8 no scale
                configs[f"FP8 no-scale      {tag}"] = {
                    "kernel": matmul_fp8_noscale_kernel,
                    "variant": "fp8_noscale",
                    "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                    "num_warps": nw, "num_stages": ns,
                }

                # 3. FP8 manual scale (production path)
                configs[f"FP8 manual-scale  {tag}"] = {
                    "kernel": matmul_fp8_manual_scale_kernel,
                    "variant": "fp8_manual_scale",
                    "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                    "num_warps": nw, "num_stages": ns,
                    "SCALE_BLOCK_K": 128,
                }

            # 4. MXFP8 dot_scaled (hardware native, TensorDescriptor + packed 5D scale)
            if block_m % 128 == 0 and block_n % 128 == 0 and block_k % 128 == 0:
                configs[f"MXFP8 dot_scaled  {tag}"] = {
                    "kernel": matmul_mxfp8_dot_scaled_tma_kernel,
                    "variant": "mxfp8_dot_scaled",
                    "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                    "num_warps": nw, "num_stages": ns,
                }

            if not configs:
                continue

            results = run_benchmark(M, N, K, configs, warmup=10, rep=50)


            for rname, res in results.items():
                if "error" in res:
                    err = res["error"]
                    if "shared memory" in err:
                        print(f"  {rname:<58} [SHMEM OOM]")
                    elif err.startswith("SKIP:"):
                        print(f"  {rname:<58} [SKIP] {err[5:165]}")
                    else:
                        print(f"  {rname:<58} [ERR] {err[:160]}")
                        if "MXFP8 dot_scaled" in rname and not printed_dot_scaled_traceback and "traceback" in res:

                            print("    First dot_scaled traceback tail:")
                            for line in res["traceback"].rstrip().splitlines()[-12:]:
                                print(f"    {line}")
                            printed_dot_scaled_traceback = True
                else:
                    print(f"  {rname:<58} {res['avg_ms']:<9.4f} {res['min_ms']:<9.4f} "
                          f"{res['avg_tflops']:<9.2f} {res['peak_tflops']:<9.2f}")


            print()  # spacer between tile configs

    # ============================================================
    # Summary: find best per variant per problem size
    # ============================================================
    print("\n" + "=" * 100)
    print("SUMMARY: Best Peak TFLOPS per variant")
    print("=" * 100)

    for M, N, K in problem_sizes:
        best = {"bf16": 0, "fp8_noscale": 0, "fp8_manual_scale": 0, "mxfp8_dot_scaled": 0}
        best_cfg = {"bf16": "", "fp8_noscale": "", "fp8_manual_scale": "", "mxfp8_dot_scaled": ""}

        for block_m, block_n, block_k, nw, ns in tile_configs:
            if block_n > N:
                continue

            tag = f"[{block_m}x{block_n}x{block_k}, w{nw}, s{ns}]"
            configs = {}

            if block_m <= M:
                configs.update({
                    "bf16": {"kernel": matmul_bf16_kernel, "variant": "bf16",
                             "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                             "num_warps": nw, "num_stages": ns},
                    "fp8_noscale": {"kernel": matmul_fp8_noscale_kernel, "variant": "fp8_noscale",
                                    "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                                    "num_warps": nw, "num_stages": ns},
                    "fp8_manual_scale": {"kernel": matmul_fp8_manual_scale_kernel, "variant": "fp8_manual_scale",
                                         "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                                         "num_warps": nw, "num_stages": ns, "SCALE_BLOCK_K": 128},
                })

            if block_m % 128 == 0 and block_n % 128 == 0 and block_k % 128 == 0:
                configs["mxfp8_dot_scaled"] = {
                    "kernel": matmul_mxfp8_dot_scaled_tma_kernel, "variant": "mxfp8_dot_scaled",
                    "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                    "num_warps": nw, "num_stages": ns,
                }

            if not configs:
                continue

            results = run_benchmark(M, N, K, configs, warmup=10, rep=50)

            for vname, res in results.items():
                if "error" not in res and res["peak_tflops"] > best[vname]:
                    best[vname] = res["peak_tflops"]
                    best_cfg[vname] = tag

        print(f"\nM={M}, N={N}, K={K}:")
        print(f"  BF16 baseline:      {best['bf16']:>7.2f} TFLOPS  {best_cfg['bf16']}")
        print(f"  FP8 no-scale:       {best['fp8_noscale']:>7.2f} TFLOPS  {best_cfg['fp8_noscale']}")
        print(f"  FP8 manual-scale:   {best['fp8_manual_scale']:>7.2f} TFLOPS  {best_cfg['fp8_manual_scale']}")
        print(f"  MXFP8 dot_scaled:   {best['mxfp8_dot_scaled']:>7.2f} TFLOPS  {best_cfg['mxfp8_dot_scaled']}")
        if best['bf16'] > 0:
            print(f"  --- Ratios vs BF16: "
                  f"no-scale={best['fp8_noscale']/best['bf16']:.2f}x  "
                  f"manual={best['fp8_manual_scale']/best['bf16']:.2f}x  "
                  f"dot_scaled={best['mxfp8_dot_scaled']/best['bf16']:.2f}x")


if __name__ == "__main__":
    main()

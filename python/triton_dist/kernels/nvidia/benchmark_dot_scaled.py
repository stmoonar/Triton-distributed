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

import torch
import triton
import triton.language as tl
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
# rhs_scale: [BLOCK_N, BLOCK_K // 32]
#
# NOTE: rhs for dot_scaled should be (BLOCK_N, BLOCK_K) layout,
# and the API uses b.T internally or we pass b as (N,K) and use .T
# According to tutorial: tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", acc)
# where b is loaded as (BLOCK_N, BLOCK_K)
# ============================================================
@triton.jit
def matmul_mxfp8_dot_scaled_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,  # a_scale strides: [M, K//32]
    stride_bsn, stride_bsk,  # b_scale strides: [N, K//32]
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,  # = 32 for MXFP8
):
    """
    MXFP8 GEMM using tl.dot_scaled for hardware-native microscaling.
    
    Data layout:
      A: [M, K] in float8_e4m3fn (row-major)
      B: [N, K] in float8_e4m3fn (note: stored as N×K for dot_scaled RHS)
      a_scale: [M, K//32] in uint8 (E8M0)
      b_scale: [N, K//32] in uint8 (E8M0)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A is [M, K]: load tile [BLOCK_M, BLOCK_K]
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B is [N, K]: load tile [BLOCK_N, BLOCK_K] (will use .T in dot_scaled)
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    # Scale pointers
    # a_scale: [M, K//32], load [BLOCK_M, BLOCK_K//32] per iteration
    # b_scale: [N, K//32], load [BLOCK_N, BLOCK_K//32] per iteration
    num_scale_per_block = BLOCK_SIZE_K // VEC_SIZE  # e.g., 128//32 = 4
    offs_scale_k = tl.arange(0, num_scale_per_block)

    a_scale_base = a_scale_ptr + offs_m[:, None] * stride_asm
    b_scale_base = b_scale_ptr + offs_n[:, None] * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A tile [BLOCK_M, BLOCK_K]
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_SIZE_K < K, other=0.0)
        # Load B tile [BLOCK_N, BLOCK_K]
        b = tl.load(b_ptrs, mask=offs_k[None, :] + k * BLOCK_SIZE_K < K, other=0.0)

        # Load scale tiles
        # a_scale: [BLOCK_M, num_scale_per_block]
        scale_k_offset = k * num_scale_per_block
        a_scale = tl.load(
            a_scale_base + (offs_scale_k[None, :] + scale_k_offset) * stride_ask,
            mask=offs_scale_k[None, :] + scale_k_offset < tl.cdiv(K, VEC_SIZE),
            other=127  # E8M0: 127 means scale=1.0
        )
        # b_scale: [BLOCK_N, num_scale_per_block]
        b_scale = tl.load(
            b_scale_base + (offs_scale_k[None, :] + scale_k_offset) * stride_bsk,
            mask=offs_scale_k[None, :] + scale_k_offset < tl.cdiv(K, VEC_SIZE),
            other=127
        )

        # Use hardware-native dot_scaled
        # a: [BLOCK_M, BLOCK_K], a_scale: [BLOCK_M, BLOCK_K//32]
        # b: [BLOCK_N, BLOCK_K], b_scale: [BLOCK_N, BLOCK_K//32]
        # dot_scaled expects: lhs=[M,K], rhs=[K,N] (so we pass b.T)
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
    a_desc, b_desc, c_desc,
    a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """
    MXFP8 GEMM using tl.dot_scaled with TMA descriptors for data tiles.
    Scale is still loaded via tl.load (simpler than 5D packed layout).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    num_scale_per_block = BLOCK_SIZE_K // VEC_SIZE
    offs_scale_k = tl.arange(0, num_scale_per_block)

    a_scale_base = a_scale_ptr + offs_m[:, None] * stride_asm
    b_scale_base = b_scale_ptr + offs_n[:, None] * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # TMA loads
        a = a_desc.load([offs_am, k * BLOCK_SIZE_K])
        b = b_desc.load([offs_bn, k * BLOCK_SIZE_K])

        # Load scales via tl.load
        scale_k_offset = k * num_scale_per_block
        a_scale = tl.load(
            a_scale_base + (offs_scale_k[None, :] + scale_k_offset) * stride_ask,
            mask=offs_scale_k[None, :] + scale_k_offset < tl.cdiv(K, VEC_SIZE),
            other=127
        )
        b_scale = tl.load(
            b_scale_base + (offs_scale_k[None, :] + scale_k_offset) * stride_bsk,
            mask=offs_scale_k[None, :] + scale_k_offset < tl.cdiv(K, VEC_SIZE),
            other=127
        )

        accumulator = tl.dot_scaled(
            a, a_scale, "e4m3",
            b.T, b_scale, "e4m3",
            accumulator
        )

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
                # MXFP8 with tl.dot_scaled
                # A: [M, K] in float8_e4m3fn
                # B: [N, K] in float8_e4m3fn (note: N×K layout for dot_scaled RHS)
                # a_scale: [M, K//32] in uint8 (E8M0)
                # b_scale: [N, K//32] in uint8 (E8M0)
                a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                # B stored as [N, K] for dot_scaled (RHS is transposed internally)
                b_nk = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

                vec_size = 32
                num_scale_k = triton.cdiv(K, vec_size)
                # E8M0 scale = 127 means scale factor = 1.0
                a_scale = torch.full((M, num_scale_k), 127, device="cuda", dtype=torch.uint8)
                b_scale = torch.full((N, num_scale_k), 127, device="cuda", dtype=torch.uint8)

                grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

                def run_fn(kernel_fn=kernel_fn, a=a, b_nk=b_nk, c=c,
                           a_scale=a_scale, b_scale=b_scale,
                           M=M, N=N, K=K, grid=grid,
                           block_m=block_m, block_n=block_n, block_k=block_k,
                           num_warps=num_warps, num_stages=num_stages, vec_size=vec_size):
                    kernel_fn[grid](
                        a, b_nk, c, a_scale, b_scale,
                        M, N, K,
                        a.stride(0), a.stride(1),       # A strides [M, K]
                        b_nk.stride(0), b_nk.stride(1), # B strides [N, K]
                        c.stride(0), c.stride(1),
                        a_scale.stride(0), a_scale.stride(1),
                        b_scale.stride(0), b_scale.stride(1),
                        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
                        VEC_SIZE=vec_size,
                        num_warps=num_warps, num_stages=num_stages,
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
            err = str(e)
            if len(err) > 120:
                err = err[:120] + "..."
            results[name] = {"error": err}

    return results


def main():
    print("=" * 100)
    print("MXFP8 dot_scaled Benchmark (Blackwell SM120a)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print("=" * 100)
    print()
    print("Key: tl.dot_scaled uses hardware-native microscaling (K-group=32, E8M0 scale)")
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
        (128, 128, 128, 4, 2),
        (64, 256, 128, 4, 2),
        (64, 128, 64, 4, 3),
        (64, 256, 64, 4, 3),
        (64, 256, 64, 4, 2),
        (128, 128, 64, 4, 3),
        (128, 128, 64, 4, 2),
        (128, 256, 64, 4, 2),
    ]

    for M, N, K in problem_sizes:
        print(f"\n{'═' * 100}")
        print(f"Problem: M={M}, N={N}, K={K}  |  FLOPS={2.0*M*N*K/1e9:.2f} GFLOPS")
        print(f"{'═' * 100}")
        print(f"{'Config':<60} {'Avg(ms)':<9} {'Min(ms)':<9} {'Avg TF':<9} {'Peak TF':<9}")
        print(f"{'─' * 100}")

        for block_m, block_n, block_k, nw, ns in tile_configs:
            if block_m > M or block_n > N:
                continue

            tag = f"[{block_m}x{block_n}x{block_k}, w{nw}, s{ns}]"

            configs = {}

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

            # 4. MXFP8 dot_scaled (hardware native)
            configs[f"MXFP8 dot_scaled  {tag}"] = {
                "kernel": matmul_mxfp8_dot_scaled_kernel,
                "variant": "mxfp8_dot_scaled",
                "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                "num_warps": nw, "num_stages": ns,
            }

            results = run_benchmark(M, N, K, configs, warmup=10, rep=50)

            for rname, res in results.items():
                if "error" in res:
                    err = res["error"]
                    if "shared memory" in err:
                        print(f"  {rname:<58} [SHMEM OOM]")
                    else:
                        print(f"  {rname:<58} [ERR] {err[:60]}")
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
            if block_m > M or block_n > N:
                continue

            tag = f"[{block_m}x{block_n}x{block_k}, w{nw}, s{ns}]"
            configs = {
                "bf16": {"kernel": matmul_bf16_kernel, "variant": "bf16",
                         "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                         "num_warps": nw, "num_stages": ns},
                "fp8_noscale": {"kernel": matmul_fp8_noscale_kernel, "variant": "fp8_noscale",
                                "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                                "num_warps": nw, "num_stages": ns},
                "fp8_manual_scale": {"kernel": matmul_fp8_manual_scale_kernel, "variant": "fp8_manual_scale",
                                     "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                                     "num_warps": nw, "num_stages": ns, "SCALE_BLOCK_K": 128},
                "mxfp8_dot_scaled": {"kernel": matmul_mxfp8_dot_scaled_kernel, "variant": "mxfp8_dot_scaled",
                                     "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                                     "num_warps": nw, "num_stages": ns},
            }

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

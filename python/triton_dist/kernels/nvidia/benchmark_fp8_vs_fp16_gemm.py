"""
Benchmark: FP8 vs FP16 (BF16) GEMM FLOPS on SM120a (Blackwell RTX Pro 5000)

Tests multiple configurations to find peak throughput for:
1. BF16 GEMM (tl.dot with accumulator fusion)
2. FP8 GEMM without scale (pure MMA throughput)
3. FP8 GEMM with block-wise scale (current production path)

Key insight: RTX Pro 5000 has only 101KB shared memory per SM.
We explore num_stages=2/1 and BLOCK_K=64 to fit larger tile configs.

Usage:
    python -m triton_dist.kernels.nvidia.benchmark_fp8_vs_fp16_gemm
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Kernel 1: BF16 GEMM (baseline, uses fused dot+acc)
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
# Kernel 2: FP8 GEMM with fused accumulator (test pure MMA throughput)
# ============================================================
@triton.jit
def matmul_fp8_fused_acc_kernel(
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
# Kernel 3: FP8 GEMM with block-wise scale (full production path)
# ============================================================
@triton.jit
def matmul_fp8_with_scale_kernel(
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
    FP8_BLOCK_K: tl.constexpr,
):
    """FP8 kernel with block-wise scale application (full production path)"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
    b_scale_ptrs = b_scale_ptr + (offs_n // BLOCK_SIZE_N) * stride_bsn  # simplified

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_SIZE_K < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k * BLOCK_SIZE_K < K, other=0.0)
        dot = tl.dot(a, b)

        # Load and apply block-wise scales
        k_scale_idx = (k * BLOCK_SIZE_K) // FP8_BLOCK_K
        a_scale = tl.load(a_scale_ptrs + k_scale_idx * stride_ask).to(tl.float32)
        b_scale = tl.load(b_scale_ptrs + k_scale_idx * stride_bsk).to(tl.float32)
        dot = dot * a_scale[:, None] * b_scale[None, :]

        accumulator += dot
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================
# Kernel 4: FP8 GEMM with scale - optimized (precompute scale outer product)
# ============================================================
@triton.jit
def matmul_fp8_scale_optimized_kernel(
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
    FP8_BLOCK_K: tl.constexpr,
):
    """FP8 kernel with optimized scale: precompute a_scale * b_scale outer product"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
    b_scale_ptrs = b_scale_ptr + (offs_n // BLOCK_SIZE_N) * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_SIZE_K < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k * BLOCK_SIZE_K < K, other=0.0)
        dot = tl.dot(a, b)

        # Optimized: precompute scale product as outer product, single multiply
        k_scale_idx = (k * BLOCK_SIZE_K) // FP8_BLOCK_K
        a_scale = tl.load(a_scale_ptrs + k_scale_idx * stride_ask).to(tl.float32)
        b_scale = tl.load(b_scale_ptrs + k_scale_idx * stride_bsk).to(tl.float32)
        scale = a_scale[:, None] * b_scale[None, :]
        accumulator += dot * scale

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================
# Kernel 5: FP8 GEMM with scale - post-loop scaling
# When BLOCK_SIZE_K == FP8_BLOCK_K, each K-iteration has exactly one scale.
# We accumulate unscaled results per K-block separately, then apply scale after.
# Actually for block-wise, we need per-K-block scale, so this variant
# accumulates dot results and applies combined scale at end of loop.
# This only works correctly if scale is constant across K (not our case).
# Instead, let's try: accumulate into multiple partial sums and scale at end.
# ============================================================
@triton.jit
def matmul_fp8_deferred_scale_kernel(
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
    FP8_BLOCK_K: tl.constexpr,
):
    """
    FP8 kernel that defers scale application:
    Instead of dot * a_scale[:, None] * b_scale[None, :] (2 broadcasts),
    we do: dot * (a_scale[:, None] * b_scale[None, :]) with fused multiply-add pattern.
    Also tries to use tl.dot(a, b, acc) form where possible.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
    b_scale_ptrs = b_scale_ptr + (offs_n // BLOCK_SIZE_N) * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_iters):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_SIZE_K < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k * BLOCK_SIZE_K < K, other=0.0)

        # Load scales first (can overlap with MMA)
        k_scale_idx = (k * BLOCK_SIZE_K) // FP8_BLOCK_K
        a_scale = tl.load(a_scale_ptrs + k_scale_idx * stride_ask).to(tl.float32)
        b_scale = tl.load(b_scale_ptrs + k_scale_idx * stride_bsk).to(tl.float32)

        # Compute dot
        dot = tl.dot(a, b)

        # Apply scale: single fused operation
        accumulator += dot * (a_scale[:, None] * b_scale[None, :])

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================
# Benchmark runner
# ============================================================
def benchmark_gemm(M, N, K, configs, warmup=25, rep=100):
    """Run benchmark for all configurations"""
    results = {}

    for name, config in configs.items():
        dtype = config["dtype"]
        kernel_fn = config["kernel"]
        block_m = config.get("BLOCK_SIZE_M", 64)
        block_n = config.get("BLOCK_SIZE_N", 256)
        block_k = config.get("BLOCK_SIZE_K", 128)
        num_warps = config.get("num_warps", 4)
        num_stages = config.get("num_stages", 3)
        has_scale = config.get("has_scale", False)
        fp8_block_k = config.get("FP8_BLOCK_K", 128)

        # Create input tensors
        if dtype == torch.float8_e4m3fn:
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        else:
            a = torch.randn(M, K, device="cuda", dtype=dtype)
            b = torch.randn(K, N, device="cuda", dtype=dtype)

        c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

        if has_scale:
            # Create scale tensors
            num_k_blocks = triton.cdiv(K, fp8_block_k)
            num_n_blocks = triton.cdiv(N, fp8_block_k)
            a_scale = torch.ones(M, num_k_blocks, device="cuda", dtype=torch.float32)
            b_scale = torch.ones(num_n_blocks, num_k_blocks, device="cuda", dtype=torch.float32)

            def run_kernel(kernel_fn=kernel_fn, a=a, b=b, c=c, a_scale=a_scale, b_scale=b_scale,
                           M=M, N=N, K=K, grid=grid, block_m=block_m, block_n=block_n, block_k=block_k,
                           num_warps=num_warps, num_stages=num_stages, fp8_block_k=fp8_block_k):
                kernel_fn[grid](
                    a, b, c,
                    a_scale, b_scale,
                    M, N, K,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    a_scale.stride(0), a_scale.stride(1),
                    b_scale.stride(0), b_scale.stride(1),
                    BLOCK_SIZE_M=block_m,
                    BLOCK_SIZE_N=block_n,
                    BLOCK_SIZE_K=block_k,
                    FP8_BLOCK_K=fp8_block_k,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
        else:
            def run_kernel(kernel_fn=kernel_fn, a=a, b=b, c=c,
                           M=M, N=N, K=K, grid=grid, block_m=block_m, block_n=block_n, block_k=block_k,
                           num_warps=num_warps, num_stages=num_stages):
                kernel_fn[grid](
                    a, b, c,
                    M, N, K,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_SIZE_M=block_m,
                    BLOCK_SIZE_N=block_n,
                    BLOCK_SIZE_K=block_k,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

        # Warmup
        for _ in range(warmup):
            run_kernel()
        torch.cuda.synchronize()

        # Benchmark
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

        for i in range(rep):
            start_events[i].record()
            run_kernel()
            end_events[i].record()

        torch.cuda.synchronize()

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        avg_time_ms = sum(times) / len(times)
        min_time_ms = min(times)

        # Calculate FLOPS (2*M*N*K for matmul)
        flops = 2.0 * M * N * K
        avg_tflops = flops / (avg_time_ms * 1e-3) / 1e12
        peak_tflops = flops / (min_time_ms * 1e-3) / 1e12

        results[name] = {
            "avg_time_ms": avg_time_ms,
            "min_time_ms": min_time_ms,
            "avg_tflops": avg_tflops,
            "peak_tflops": peak_tflops,
        }

    return results


def main():
    print("=" * 90)
    print("FP8 vs BF16 GEMM Benchmark (RTX Pro 5000, SM120a, 101KB shmem)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print("=" * 90)

    # Problem sizes relevant to MoE (DeepSeek-V3 like)
    problem_sizes = [
        (64, 18432, 7168),    # fc1: few tokens per expert
        (128, 18432, 7168),   # fc1: more tokens
        (256, 18432, 7168),   # fc1: many tokens
        (64, 7168, 18432),    # fc2
        (128, 7168, 18432),   # fc2: more tokens
    ]

    # Tile configurations: (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
    # Designed to fit in 101KB shared memory
    # shmem ≈ num_stages * (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * elem_size
    # BF16=2B, FP8=1B
    tile_configs = [
        # --- Current production configs ---
        (64, 256, 64, 4, 3),     # BF16 production (shmem: 3*(64*64+64*256)*2 = 147KB) -> might OOM
        (64, 128, 128, 4, 3),    # FP8 production that works (shmem: 3*(64*128+128*128)*1 = 73KB) ✓
        # --- Explore lower stages to fit bigger tiles ---
        (64, 256, 64, 4, 2),     # BF16 with 2 stages (shmem: 2*(64*64+64*256)*2 = 98KB) ✓ barely
        (64, 256, 128, 4, 2),    # FP8 bigger N, 2 stages (shmem: 2*(64*128+128*256)*1 = 80KB) ✓
        (64, 256, 64, 4, 1),     # 1 stage
        (64, 256, 128, 4, 1),    # FP8 biggest tile, 1 stage
        # --- FP8 with BLOCK_K=64 (same as BF16, allows larger N tile) ---
        (64, 256, 64, 4, 3),     # FP8 with K=64 (shmem: 3*(64*64+64*256)*1 = 73KB) ✓
        (64, 256, 64, 4, 2),     # FP8 K=64, 2 stages
        (64, 128, 64, 4, 3),     # FP8 K=64, smaller N
        # --- Try more warps ---
        (64, 128, 128, 8, 3),    # More warps with working config
        (64, 128, 128, 4, 2),    # Fewer stages
        (64, 128, 64, 4, 2),     # Smaller K, fewer stages
        # --- Larger M ---
        (128, 128, 64, 4, 2),    # Larger M, BF16 friendly
        (128, 128, 64, 4, 3),    # Larger M
        (128, 256, 64, 4, 2),    # Larger M + N
    ]

    for M, N, K in problem_sizes:
        print(f"\n{'═' * 90}")
        print(f"Problem Size: M={M}, N={N}, K={K}  |  FLOPS={2.0*M*N*K/1e9:.2f} GFLOPS")
        print(f"{'═' * 90}")
        print(f"{'Config':<55} {'Avg(ms)':<9} {'Min(ms)':<9} {'Avg TF':<9} {'Peak TF':<9}")
        print(f"{'─' * 90}")

        seen_configs = set()

        for block_m, block_n, block_k, nw, ns in tile_configs:
            # Skip configs that don't tile
            if block_m > M or block_n > N:
                continue

            suffix = f"M{block_m}xN{block_n}xK{block_k}_w{nw}_s{ns}"
            if suffix in seen_configs:
                continue
            seen_configs.add(suffix)

            configs = {}

            # BF16
            configs[f"BF16          {suffix}"] = {
                "dtype": torch.bfloat16,
                "kernel": matmul_bf16_kernel,
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "num_warps": nw,
                "num_stages": ns,
                "has_scale": False,
            }

            # FP8 no scale
            configs[f"FP8_no_scale  {suffix}"] = {
                "dtype": torch.float8_e4m3fn,
                "kernel": matmul_fp8_fused_acc_kernel,
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "num_warps": nw,
                "num_stages": ns,
                "has_scale": False,
            }

            # FP8 with scale (production style)
            configs[f"FP8_w_scale   {suffix}"] = {
                "dtype": torch.float8_e4m3fn,
                "kernel": matmul_fp8_with_scale_kernel,
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "num_warps": nw,
                "num_stages": ns,
                "has_scale": True,
                "FP8_BLOCK_K": max(block_k, 128),  # scale block is always 128
            }

            # FP8 with optimized scale (deferred multiply)
            configs[f"FP8_opt_scale {suffix}"] = {
                "dtype": torch.float8_e4m3fn,
                "kernel": matmul_fp8_deferred_scale_kernel,
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "num_warps": nw,
                "num_stages": ns,
                "has_scale": True,
                "FP8_BLOCK_K": max(block_k, 128),
            }

            try:
                results = benchmark_gemm(M, N, K, configs, warmup=10, rep=50)
                for rname, res in results.items():
                    print(f"  {rname:<53} {res['avg_time_ms']:<9.4f} {res['min_time_ms']:<9.4f} "
                          f"{res['avg_tflops']:<9.2f} {res['peak_tflops']:<9.2f}")
            except Exception as e:
                err_msg = str(e)
                if "shared memory" in err_msg:
                    print(f"  [SHMEM OOM] {suffix}")
                else:
                    print(f"  [ERROR] {suffix}: {err_msg[:80]}")

        # Print separator between problem sizes
        print()

    # ============================================================
    # Final summary: find best for each problem size
    # ============================================================
    print("\n" + "=" * 90)
    print("FINAL SUMMARY: Best config per problem size")
    print("=" * 90)

    # Re-run with only good configs for summary
    good_configs_bf16 = [
        (64, 256, 64, 4, 3),
        (64, 256, 64, 4, 2),
        (64, 128, 128, 4, 3),
        (128, 128, 64, 4, 2),
        (128, 128, 64, 4, 3),
        (128, 256, 64, 4, 2),
    ]
    good_configs_fp8 = [
        (64, 128, 128, 4, 3),
        (64, 256, 128, 4, 2),
        (64, 256, 128, 4, 1),
        (64, 256, 64, 4, 3),
        (64, 256, 64, 4, 2),
        (64, 128, 128, 8, 3),
        (64, 128, 128, 4, 2),
        (128, 128, 64, 4, 2),
        (128, 128, 64, 4, 3),
        (128, 256, 64, 4, 2),
    ]

    for M, N, K in problem_sizes:
        all_configs = {}

        for block_m, block_n, block_k, nw, ns in good_configs_bf16:
            if block_m > M or block_n > N:
                continue
            suffix = f"M{block_m}xN{block_n}xK{block_k}_w{nw}_s{ns}"
            all_configs[f"BF16 {suffix}"] = {
                "dtype": torch.bfloat16,
                "kernel": matmul_bf16_kernel,
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "num_warps": nw,
                "num_stages": ns,
                "has_scale": False,
            }

        for block_m, block_n, block_k, nw, ns in good_configs_fp8:
            if block_m > M or block_n > N:
                continue
            suffix = f"M{block_m}xN{block_n}xK{block_k}_w{nw}_s{ns}"
            all_configs[f"FP8_no_scale {suffix}"] = {
                "dtype": torch.float8_e4m3fn,
                "kernel": matmul_fp8_fused_acc_kernel,
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "num_warps": nw,
                "num_stages": ns,
                "has_scale": False,
            }
            all_configs[f"FP8_w_scale {suffix}"] = {
                "dtype": torch.float8_e4m3fn,
                "kernel": matmul_fp8_with_scale_kernel,
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "num_warps": nw,
                "num_stages": ns,
                "has_scale": True,
                "FP8_BLOCK_K": max(block_k, 128),
            }

        try:
            results = benchmark_gemm(M, N, K, all_configs, warmup=15, rep=80)

            best_bf16 = max([(k, v) for k, v in results.items() if "BF16" in k],
                           key=lambda x: x[1]["peak_tflops"], default=None)
            best_fp8_noscale = max([(k, v) for k, v in results.items() if "FP8_no_scale" in k],
                                   key=lambda x: x[1]["peak_tflops"], default=None)
            best_fp8_scale = max([(k, v) for k, v in results.items() if "FP8_w_scale" in k],
                                  key=lambda x: x[1]["peak_tflops"], default=None)

            print(f"\nM={M}, N={N}, K={K}:")
            if best_bf16:
                print(f"  Best BF16:         {best_bf16[1]['peak_tflops']:>7.2f} TFLOPS  ({best_bf16[0]})")
            if best_fp8_noscale:
                print(f"  Best FP8 no-scale: {best_fp8_noscale[1]['peak_tflops']:>7.2f} TFLOPS  ({best_fp8_noscale[0]})")
            if best_fp8_scale:
                print(f"  Best FP8 w/scale:  {best_fp8_scale[1]['peak_tflops']:>7.2f} TFLOPS  ({best_fp8_scale[0]})")
            if best_bf16 and best_fp8_noscale:
                ratio = best_fp8_noscale[1]["peak_tflops"] / best_bf16[1]["peak_tflops"]
                print(f"  FP8(no-scale)/BF16 = {ratio:.2f}x")
            if best_bf16 and best_fp8_scale:
                ratio = best_fp8_scale[1]["peak_tflops"] / best_bf16[1]["peak_tflops"]
                print(f"  FP8(w/scale)/BF16  = {ratio:.2f}x  (this is the real prod perf)")
        except Exception as e:
            print(f"\nM={M}, N={N}, K={K}: ERROR - {e}")


if __name__ == "__main__":
    main()

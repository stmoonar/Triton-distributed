"""
Benchmark: FP8 vs FP16 (BF16) GEMM FLOPS on SM120a (Blackwell RTX Pro 5000)

Tests multiple configurations to find peak throughput for:
1. BF16 GEMM (tl.dot with accumulator fusion)
2. FP8 GEMM without scale (pure MMA throughput)
3. FP8 GEMM with block-wise scale (current production path)

Usage:
    python -m triton_dist.kernels.nvidia.benchmark_fp8_vs_fp16_gemm
"""

import torch
import triton
import triton.language as tl
import time


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
# Kernel 2: FP8 GEMM without scale (test pure MMA throughput)
# ============================================================
@triton.jit
def matmul_fp8_no_scale_kernel(
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
# Kernel 3: FP8 GEMM with separate dot + accumulate (current FP8 path style)
# ============================================================
@triton.jit
def matmul_fp8_separate_acc_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """FP8 kernel using separate dot + accumulate (like current production code)"""
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
        dot = tl.dot(a, b)
        accumulator += dot
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ============================================================
# Kernel 4: FP8 GEMM with block-wise scale (full production path)
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

        # Create input tensors
        if dtype == torch.float8_e4m3fn:
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        else:
            a = torch.randn(M, K, device="cuda", dtype=dtype)
            b = torch.randn(K, N, device="cuda", dtype=dtype)

        c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

        if "with_scale" in name:
            # Create scale tensors
            num_k_blocks = triton.cdiv(K, 128)
            num_n_blocks = triton.cdiv(N, 128)
            a_scale = torch.ones(M, num_k_blocks, device="cuda", dtype=torch.float32)
            b_scale = torch.ones(num_n_blocks, num_k_blocks, device="cuda", dtype=torch.float32)

            def run_kernel():
                matmul_fp8_with_scale_kernel[grid](
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
                    FP8_BLOCK_K=128,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
        else:
            def run_kernel():
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
    print("=" * 80)
    print("FP8 vs BF16 GEMM Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print("=" * 80)

    # Test multiple problem sizes relevant to MoE workload
    # DeepSeek-V3: hidden=7168, intermediate=18432 (up+gate=2*18432=36864 for fc1, 18432 for fc2)
    problem_sizes = [
        # (M, N, K) - typical MoE shapes
        (64, 18432, 7168),    # fc1: few tokens per expert, typical
        (128, 18432, 7168),   # fc1: more tokens
        (256, 18432, 7168),   # fc1: many tokens
        (64, 7168, 18432),    # fc2: typical
        (128, 7168, 18432),   # fc2: more tokens
        # Smaller shapes for debugging
        (256, 4096, 4096),    # square-ish
        (512, 4096, 4096),    # larger M
    ]

    # Configurations to test
    tile_configs = [
        # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
        (64, 256, 64, 4, 3),    # BF16 style (current production)
        (64, 256, 128, 4, 3),   # FP8 style (current production)
        (64, 128, 128, 4, 3),   # Smaller N tile
        (128, 128, 128, 4, 3),  # Larger M tile
        (128, 256, 128, 8, 3),  # More warps
        (64, 256, 128, 8, 3),   # More warps, FP8 default
    ]

    for M, N, K in problem_sizes:
        print(f"\n{'─' * 80}")
        print(f"Problem Size: M={M}, N={N}, K={K}")
        print(f"FLOPS per matmul: {2.0 * M * N * K / 1e9:.2f} GFLOPS")
        print(f"{'─' * 80}")
        print(f"{'Config':<50} {'Avg(ms)':<10} {'Min(ms)':<10} {'Avg TFLOPS':<12} {'Peak TFLOPS':<12}")
        print(f"{'─' * 80}")

        for block_m, block_n, block_k, nw, ns in tile_configs:
            # Skip configs that don't tile evenly or are too large
            if block_m > M or block_n > N:
                continue

            configs = {}
            suffix = f"[{block_m}x{block_n}x{block_k}, w={nw}, s={ns}]"

            # BF16 fused acc
            configs[f"BF16 fused_acc {suffix}"] = {
                "dtype": torch.bfloat16,
                "kernel": matmul_bf16_kernel,
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "num_warps": nw,
                "num_stages": ns,
            }

            # FP8 fused acc (tl.dot with acc - test if this works for fp8)
            configs[f"FP8 fused_acc {suffix}"] = {
                "dtype": torch.float8_e4m3fn,
                "kernel": matmul_fp8_no_scale_kernel,
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "num_warps": nw,
                "num_stages": ns,
            }

            # FP8 separate acc (current production style)
            configs[f"FP8 separate_acc {suffix}"] = {
                "dtype": torch.float8_e4m3fn,
                "kernel": matmul_fp8_separate_acc_kernel,
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "num_warps": nw,
                "num_stages": ns,
            }

            # FP8 with scale (full production path)
            configs[f"FP8 with_scale {suffix}"] = {
                "dtype": torch.float8_e4m3fn,
                "kernel": matmul_fp8_with_scale_kernel,
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "num_warps": nw,
                "num_stages": ns,
            }

            try:
                results = benchmark_gemm(M, N, K, configs, warmup=10, rep=50)
                for name, res in results.items():
                    print(f"{name:<50} {res['avg_time_ms']:<10.4f} {res['min_time_ms']:<10.4f} "
                          f"{res['avg_tflops']:<12.2f} {res['peak_tflops']:<12.2f}")
            except Exception as e:
                print(f"  ERROR with config {suffix}: {e}")

        print()

    # Summary: best config for each dtype
    print("\n" + "=" * 80)
    print("SUMMARY: Best configurations per problem size")
    print("=" * 80)

    for M, N, K in problem_sizes:
        print(f"\nM={M}, N={N}, K={K}:")

        all_configs = {}
        for block_m, block_n, block_k, nw, ns in tile_configs:
            if block_m > M or block_n > N:
                continue
            suffix = f"[{block_m}x{block_n}x{block_k}, w={nw}, s={ns}]"
            all_configs[f"BF16 {suffix}"] = {
                "dtype": torch.bfloat16,
                "kernel": matmul_bf16_kernel,
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "num_warps": nw,
                "num_stages": ns,
            }
            all_configs[f"FP8_fused {suffix}"] = {
                "dtype": torch.float8_e4m3fn,
                "kernel": matmul_fp8_no_scale_kernel,
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "num_warps": nw,
                "num_stages": ns,
            }

        try:
            results = benchmark_gemm(M, N, K, all_configs, warmup=15, rep=80)
            # Find best BF16 and best FP8
            best_bf16 = max([(k, v) for k, v in results.items() if "BF16" in k],
                           key=lambda x: x[1]["peak_tflops"], default=None)
            best_fp8 = max([(k, v) for k, v in results.items() if "FP8" in k],
                          key=lambda x: x[1]["peak_tflops"], default=None)

            if best_bf16:
                print(f"  Best BF16: {best_bf16[0]}")
                print(f"    Peak: {best_bf16[1]['peak_tflops']:.2f} TFLOPS, Avg: {best_bf16[1]['avg_tflops']:.2f} TFLOPS")
            if best_fp8:
                print(f"  Best FP8:  {best_fp8[0]}")
                print(f"    Peak: {best_fp8[1]['peak_tflops']:.2f} TFLOPS, Avg: {best_fp8[1]['avg_tflops']:.2f} TFLOPS")
            if best_bf16 and best_fp8:
                ratio = best_fp8[1]["peak_tflops"] / best_bf16[1]["peak_tflops"]
                print(f"  FP8/BF16 ratio: {ratio:.2f}x (theoretical: 2.0x)")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()

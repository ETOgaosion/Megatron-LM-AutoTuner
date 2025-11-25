# triton_gemm_benchmark.py
# Self-contained Triton GEMM kernel generator + benchmark harness
# Requirements:
#   pip install torch triton matplotlib
# Run example:
#   python triton_gemm_benchmark.py --M 2048 --N 2048 --K 2048

import argparse
import math
import statistics
import time
from typing import List, Tuple

import torch
import triton
import triton.language as tl


def make_gemm_kernel(BM: int, BN: int, BK: int):
    """
    Returns a Triton GEMM kernel with compile-time block sizes BM, BN, BK.
    The returned kernel has signature
      kernel(A_ptr, B_ptr, C_ptr, M, N, K, lda, ldb, ldc, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn)

    We'll use simple row-major addressing and assume float32.
    """

    @triton.jit
    def kernel(
        A_ptr,
        B_ptr,
        C_ptr,
        M,
        N,
        K,
        lda,
        ldb,
        ldc,
        # strides for batched support (set to 0 if unused)
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr = BM,
        BLOCK_N: tl.constexpr = BN,
        BLOCK_K: tl.constexpr = BK,
    ):
        # program ids -> block row/col
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # compute row and col offsets for this block
        row_offset = pid_m * BLOCK_M
        col_offset = pid_n * BLOCK_N

        # ranges
        rows = row_offset + tl.arange(0, BLOCK_M)
        cols = col_offset + tl.arange(0, BLOCK_N)

        # accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # loop over K in chunks of BLOCK_K
        for k_offset in range(0, K, BLOCK_K):
            # K range for this block
            k_idxs = k_offset + tl.arange(0, BLOCK_K)

            # load A: shape BLOCK_M x BLOCK_K (with guard)
            a_ptrs = A_ptr + (rows[:, None] * lda + k_idxs[None, :])
            a_mask = (rows[:, None] < M) & (k_idxs[None, :] < K)
            A_block = tl.load(a_ptrs, mask=a_mask, other=0.0)

            # load B: shape BLOCK_K x BLOCK_N
            b_ptrs = B_ptr + (k_idxs[:, None] * ldb + cols[None, :])
            b_mask = (k_idxs[:, None] < K) & (cols[None, :] < N)
            B_block = tl.load(b_ptrs, mask=b_mask, other=0.0)

            # compute
            acc += tl.dot(A_block, B_block)

        # write back to C (with guards)
        c_ptrs = C_ptr + (rows[:, None] * ldc + cols[None, :])
        c_mask = (rows[:, None] < M) & (cols[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)

    return kernel


def benchmark_gemm(
    M: int,
    N: int,
    K: int,
    BM_list: List[int],
    BN_list: List[int],
    BK_list: List[int],
    iterations: int = 20,
    warmup: int = 5,
    dtype=torch.float32,
    device="cuda",
) -> List[Tuple[Tuple[int, int, int], float]]:
    """Benchmark the GEMM kernel across block sizes. Returns list of ((BM,BN,BK), median_ms)"""
    # allocate random inputs
    A = torch.randn((M, K), device=device, dtype=dtype)
    B = torch.randn((K, N), device=device, dtype=dtype)
    C = torch.zeros((M, N), device=device, dtype=dtype)

    lda = K
    ldb = N
    ldc = N

    results = []

    for BM in BM_list:
        for BN in BN_list:
            for BK in BK_list:
                # simple check: grid sizes
                grid_m = math.ceil(M / BM)
                grid_n = math.ceil(N / BN)

                kernel = make_gemm_kernel(BM, BN, BK)

                # wrapper to call kernel
                def launch():
                    kernel[grid_m, grid_n](
                        A, B, C, M, N, K, lda, ldb, ldc, 0, 0, 0, 0, 0, 0
                    )

                # warmup
                for _ in range(warmup):
                    launch()
                torch.cuda.synchronize()

                times = []
                for _ in range(iterations):
                    t0 = time.perf_counter()
                    launch()
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1000.0)

                median_ms = statistics.median(times)
                results.append(((BM, BN, BK, grid_m, grid_n), median_ms))
                print(
                    f"BM={BM} BN={BN} BK={BK} grid=({grid_m},{grid_n})  median_ms={median_ms:.3f}"
                )

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=2048)
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--K", type=int, default=2048)
    p.add_argument("--BM", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--BN", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--BK", nargs="+", type=int, default=[32, 64])
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("This benchmark requires a CUDA GPU and Triton.")

    print(f"Benchmarking GEMM M={args.M} N={args.N} K={args.K} on device={device}")
    results = benchmark_gemm(
        args.M,
        args.N,
        args.K,
        args.BM,
        args.BN,
        args.BK,
        iterations=args.iterations,
        warmup=args.warmup,
    )

    # sort results by median time
    results.sort(key=lambda x: x[1])
    print("\nTop 10 configs:")
    for cfg, t in results[:10]:
        BM, BN, BK, gm, gn = cfg
        print(f"BM={BM} BN={BN} BK={BK} grid=({gm},{gn})  median_ms={t:.3f}")


if __name__ == "__main__":
    main()

import argparse
import os
import time

import numpy as np
import torch
import triton
import triton.language as tl
from torch.cuda import check_error, cudart


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(GROUP_M, num_pid_m - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Note: mask should be k * BLOCK_K + offs_k < K, but simplified here for brevity
        a = tl.load(
            a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K), other=0.0
        )
        b = tl.load(
            b_ptrs, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N), other=0.0
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_cn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, block_m, block_n, block_k, group_m=8):
    # Check compatibility
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.device == b.device, "Inputs must be on the same device"
    assert a.dtype == b.dtype == torch.float16, "Inputs must be float16"

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=4,
        num_stages=3,
    )
    return c


class Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, block_m, block_n, block_k, group_m):
        c = matmul(a, b, block_m, block_n, block_k, group_m)
        ctx.save_for_backward(a, b)
        ctx.block_m = block_m
        ctx.block_n = block_n
        ctx.block_k = block_k
        ctx.group_m = group_m
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        block_m, block_n, block_k, group_m = (
            ctx.block_m,
            ctx.block_n,
            ctx.block_k,
            ctx.group_m,
        )

        # grad_a = grad_output @ b.T
        grad_a = matmul(
            grad_output, b.transpose(0, 1), block_m, block_n, block_k, group_m
        )
        # grad_b = a.T @ grad_output
        grad_b = matmul(
            a.transpose(0, 1), grad_output, block_m, block_n, block_k, group_m
        )
        return grad_a, grad_b, None, None, None, None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_warmups", type=int, default=5)
    parser.add_argument("--num_runs", type=int, default=1)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/test/custom/gemm",
        help="Output directory",
    )

    parser.add_argument("--draw", action="store_true", help="Draw performance figure")
    return parser.parse_args()


def draw_performance_figure(output_dir, results):
    """
    Draw performance comparison figure for different GEMM configurations.

    Args:
        results: List of tuples (config, forward_time, backward_time)
    """
    import matplotlib.pyplot as plt

    # Group results by MKN and BMBKBN
    mkn_groups = (
        {}
    )  # key: (M, K, N), value: list of (block_m, block_n, block_k, forward_time, backward_time)
    block_groups = (
        {}
    )  # key: (block_m, block_n, block_k), value: list of (M, K, N, forward_time, backward_time)

    for config, forward_time, backward_time in results:
        M, K, N, block_m, block_n, block_k = config
        mkn_key = (M, K, N)
        block_key = (block_m, block_n, block_k)

        if mkn_key not in mkn_groups:
            mkn_groups[mkn_key] = []
        mkn_groups[mkn_key].append(
            (block_m, block_n, block_k, forward_time, backward_time)
        )

        if block_key not in block_groups:
            block_groups[block_key] = []
        block_groups[block_key].append((M, K, N, forward_time, backward_time))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Forward time - MKN vs BMBKBN
    for mkn_key, data in sorted(mkn_groups.items()):
        M, K, N = mkn_key
        data_sorted = sorted(data, key=lambda x: (x[0], x[1], x[2]))
        block_labels = [f"{bm}_{bn}_{bk}" for bm, bn, bk, _, _ in data_sorted]
        forward_times = [ft for _, _, _, ft, _ in data_sorted]
        ax1.plot(block_labels, forward_times, marker="o", label=f"MKN={M}", linewidth=2)

    ax1.set_xlabel("Block Configuration (BM_BN_BK)")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Forward Time: MKN vs Block Configuration")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # Plot 2: Backward time - MKN vs BMBKBN
    for mkn_key, data in sorted(mkn_groups.items()):
        M, K, N = mkn_key
        data_sorted = sorted(data, key=lambda x: (x[0], x[1], x[2]))
        block_labels = [f"{bm}_{bn}_{bk}" for bm, bn, bk, _, _ in data_sorted]
        backward_times = [bt for _, _, _, _, bt in data_sorted]
        ax2.plot(
            block_labels, backward_times, marker="s", label=f"MKN={M}", linewidth=2
        )

    ax2.set_xlabel("Block Configuration (BM_BN_BK)")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Backward Time: MKN vs Block Configuration")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    # Plot 3: Forward time - BMBKBN vs MKN
    for block_key, data in sorted(block_groups.items()):
        bm, bn, bk = block_key
        data_sorted = sorted(data, key=lambda x: (x[0], x[1], x[2]))
        mkn_labels = [M for M, K, N, _, _ in data_sorted]  # Assuming M=K=N
        forward_times = [ft for _, _, _, ft, _ in data_sorted]
        ax3.plot(
            mkn_labels,
            forward_times,
            marker="o",
            label=f"Block={bm}_{bn}_{bk}",
            linewidth=2,
        )

    ax3.set_xlabel("Matrix Size (M=K=N)")
    ax3.set_ylabel("Time (seconds)")
    ax3.set_title("Forward Time: Block Configuration vs Matrix Size")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Backward time - BMBKBN vs MKN
    for block_key, data in sorted(block_groups.items()):
        bm, bn, bk = block_key
        data_sorted = sorted(data, key=lambda x: (x[0], x[1], x[2]))
        mkn_labels = [M for M, K, N, _, _ in data_sorted]
        backward_times = [bt for _, _, _, _, bt in data_sorted]
        ax4.plot(
            mkn_labels,
            backward_times,
            marker="s",
            label=f"Block={bm}_{bn}_{bk}",
            linewidth=2,
        )

    ax4.set_xlabel("Matrix Size (M=K=N)")
    ax4.set_ylabel("Time (seconds)")
    ax4.set_title("Backward Time: Block Configuration vs Matrix Size")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # configs = [f"M{c[0]}_K{c[1]}_N{c[2]}_BM{c[3]}_BN{c[4]}_BK{c[5]}" for c, _, _ in results]
    # forward_times = [f for _, f, _ in results]
    # backward_times = [b for _, _, b in results]

    # x = np.arange(len(configs))
    # width = 0.35

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # # Bar chart
    # ax1.bar(x - width / 2, forward_times, width, label="Forward", alpha=0.8)
    # ax1.bar(x + width / 2, backward_times, width, label="Backward", alpha=0.8)
    # ax1.set_xlabel("Configuration")
    # ax1.set_ylabel("Time (seconds)")
    # ax1.set_title(f"GEMM Performance: A({results[0][0][0]},{results[0][0][1]}) @ B({results[0][0][1]},{results[0][0][2]})")
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(configs, rotation=45, ha="right")
    # ax1.legend()
    # ax1.grid(True, alpha=0.3)

    # # Line chart
    # ax2.plot(configs, forward_times, marker="o", label="Forward", linewidth=2)
    # ax2.plot(configs, backward_times, marker="s", label="Backward", linewidth=2)
    # ax2.set_xlabel("Configuration")
    # ax2.set_ylabel("Time (seconds)")
    # ax2.set_title(f"GEMM Performance Trend")
    # ax2.set_xticklabels(configs, rotation=45, ha="right")
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/gemm_performance.png", dpi=300, bbox_inches="tight")
    print(f"\nPerformance figure saved to '{output_dir}/gemm_performance.png'")
    plt.close()


# Testing function
def test_matmul(configs=None, num_warmups=5, num_runs=1, draw=False):
    if configs is None:
        configs = [
            (512, 512, 512, 32, 32, 32, 8),
            (1024, 1024, 1024, 32, 32, 32, 8),
            (2048, 2048, 2048, 32, 32, 32, 8),
            (3072, 3072, 3072, 32, 32, 32, 8),
            (4096, 4096, 4096, 32, 32, 32, 8),
            (6144, 6144, 6144, 32, 32, 32, 8),
            (8192, 8192, 8192, 32, 32, 32, 8),
            (512, 512, 512, 64, 64, 32, 8),
            (1024, 1024, 1024, 64, 64, 32, 8),
            (2048, 2048, 2048, 64, 64, 32, 8),
            (3072, 3072, 3072, 64, 64, 32, 8),
            (4096, 4096, 4096, 64, 64, 32, 8),
            (6144, 6144, 6144, 64, 64, 32, 8),
            (8192, 8192, 8192, 64, 64, 32, 8),
            (512, 512, 512, 128, 128, 64, 8),
            (1024, 1024, 1024, 128, 128, 64, 8),
            (2048, 2048, 2048, 128, 128, 64, 8),
            (3072, 3072, 3072, 128, 128, 64, 8),
            (4096, 4096, 4096, 128, 128, 64, 8),
            (6144, 6144, 6144, 128, 128, 64, 8),
            (8192, 8192, 8192, 128, 128, 64, 8),
        ]

    device = torch.device("cuda")

    results = []

    for M, K, N, block_m, block_n, block_k, group_m in configs:
        print(f"Testing GEMM for shapes A({M},{K}), B({K},{N})")
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)
        a.requires_grad = True
        b.requires_grad = True
        grid_size = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), 1, 1)
        print(
            f"\nConfig: BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}, GROUP_M={group_m}, Grid={grid_size}"
        )

        # Warmup
        c = Matmul.apply(a, b, block_m, block_n, block_k, group_m)

        # Time forward
        torch.cuda.synchronize()
        # warmup
        for _ in range(num_warmups):
            c = Matmul.apply(a, b, block_m, block_n, block_k, group_m)
        torch.cuda.synchronize()

        # run
        check_error(cudart().cudaProfilerStart())
        start = time.time()
        for _ in range(num_runs):
            torch.cuda.nvtx.range_push(
                f"GEMM Forward BLOCK_M={block_m} BLOCK_N={block_n} BLOCK_K={block_k}"
            )
            c = Matmul.apply(a, b, block_m, block_n, block_k, group_m)
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        forward_time = (time.time() - start) / num_runs
        print(f"Average forward time: {forward_time:.6f} seconds")

        # Time backward
        torch.cuda.synchronize()
        # warmup
        # for _ in range(num_warmups):
        #     c = Matmul.apply(a, b, block_m, block_n, block_k, group_m)
        #     c.sum().backward(retain_graph=True)
        #     a.grad = None
        #     b.grad = None

        # run
        torch.cuda.synchronize()
        # check_error(cudart().cudaProfilerStart())
        start = time.time()
        for _ in range(num_runs):
            c = Matmul.apply(a, b, block_m, block_n, block_k, group_m)
            torch.cuda.nvtx.range_push(
                f"GEMM Backward BLOCK_M={block_m} BLOCK_N={block_n} BLOCK_K={block_k}"
            )
            c.sum().backward(retain_graph=True)
            torch.cuda.nvtx.range_pop()
            a.grad = None
            b.grad = None
        torch.cuda.synchronize()
        check_error(cudart().cudaProfilerStop())
        full_time = (time.time() - start) / num_runs
        backward_time = full_time - forward_time  # Approximate backward time
        print(f"Average full (fwd+bwd) time: {full_time:.6f} seconds")
        print(f"Approximate backward time: {backward_time:.6f} seconds")

        results.append(
            ((M, K, N, block_m, block_n, block_k), forward_time, backward_time)
        )

    if draw:
        draw_performance_figure(args.output_dir, results)


# Run the test (requires CUDA-enabled environment)
if __name__ == "__main__":
    args = get_args()
    test_matmul(num_warmups=args.num_warmups, num_runs=args.num_runs, draw=args.draw)

import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
import triton
import triton.language as tl
from mpl_toolkits.mplot3d import Axes3D
from torch.cuda import check_error, cudart
from transformer_engine.pytorch.module.layernorm import LayerNorm
from transformer_engine.pytorch.module.rmsnorm import RMSNorm

# ================================================================
# Customizable LayerNorm Kernel (Forward + Backward)
# ================================================================


def draw_performance_plots(results, output_dir="outputs/test/custom/layernorm"):
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    B_vals = sorted(set(r["B"] for r in results))
    S_vals = sorted(set(r["S"] for r in results))
    D_vals = sorted(set(r["D"] for r in results))

    # Create a dictionary for quick lookup
    data_dict = {}
    for r in results:
        key = (r["B"], r["S"], r["D"])
        data_dict[key] = r

    fig = plt.figure(figsize=(20, 16))

    metrics = [
        ("rmsnorm_fwd_ms", "RMSNorm Forward"),
        ("rmsnorm_bwd_ms", "RMSNorm Backward"),
        ("layernorm_fwd_ms", "LayerNorm Forward"),
        ("layernorm_bwd_ms", "LayerNorm Backward"),
    ]

    for idx, (metric, title) in enumerate(metrics, 1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")

        # Plot varying B (fix S, D)
        for S in S_vals:
            for D in D_vals:
                b_data, times = [], []
                for B in B_vals:
                    if (B, S, D) in data_dict:
                        b_data.append(B)
                        times.append(data_dict[(B, S, D)][metric])
                if b_data:
                    ax.plot(
                        b_data,
                        [S] * len(b_data),
                        times,
                        marker="o",
                        label=f"S={S},D={D}",
                    )

        # Plot varying S (fix B, D)
        for B in B_vals:
            for D in D_vals:
                s_data, times = [], []
                for S in S_vals:
                    if (B, S, D) in data_dict:
                        s_data.append(S)
                        times.append(data_dict[(B, S, D)][metric])
                if s_data:
                    ax.plot([B] * len(s_data), s_data, times, marker="s", alpha=0.7)

        # Plot varying D (fix B, S)
        for B in B_vals:
            for S in S_vals:
                d_data, times = [], []
                for D in D_vals:
                    if (B, S, D) in data_dict:
                        d_data.append(D)
                        times.append(data_dict[(B, S, D)][metric])
                if d_data:
                    ax.plot(
                        [B] * len(d_data),
                        [S] * len(d_data),
                        times,
                        marker="^",
                        alpha=0.7,
                    )

        ax.set_xlabel("Batch Size (B)")
        ax.set_ylabel("Sequence Length (S)")
        ax.set_zlabel("Time (ms)")
        ax.set_title(title)
        ax.legend(fontsize=6, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/layernorm_performance_3d.png", dpi=150)
    print(f"Performance plot saved to {output_dir}/layernorm_performance_3d.png")


def benchmark_te_layernorm(
    batch_sizes: Tuple[int, ...] = (1, 2, 4, 8, 16),
    seq_len=(128, 256, 512, 1024, 2048, 4096, 8192),
    hidden_sizes: Tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096),
    num_warmup: int = 50,
    num_iters: int = 200,
):
    torch.manual_seed(42)
    device = torch.device("cuda")

    results = []

    configs = []
    for B in batch_sizes:
        for S in seq_len:
            for D in hidden_sizes:
                configs.append((B, S, D))

    for B, S, D in configs:
        print(f"Testing TE LayerNorm for B={B}, S={S}, D={D}")

        hidden_states = torch.randn(S, B, D, device="cuda", dtype=torch.float32)

        hidden_states.requires_grad = True

        rmsnorm = RMSNorm(
            normalized_shape=D,
            eps=1e-6,
            device=device,
        )

        layernorm = LayerNorm(
            normalized_shape=D,
            eps=1e-6,
            device=device,
        )

        # Warmup
        for _ in range(num_warmup):
            out1 = rmsnorm(hidden_states)
            out1.sum().backward()
            hidden_states.grad = None

        torch.cuda.synchronize()
        check_error(cudart().cudaProfilerStart())
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start = time.time()
            torch.cuda.nvtx.range_push("TE RMSNorm Forward")
            out1 = rmsnorm(hidden_states)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            rmsnorm_fwd_time = (time.time() - start) * 1000

            torch.cuda.synchronize()
            start = time.time()
            torch.cuda.nvtx.range_push("TE RMSNorm Backward")
            out1.sum().backward()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            rmsnorm_bwd_time = (time.time() - start) * 1000
            hidden_states.grad = None
        torch.cuda.synchronize()
        check_error(cudart().cudaProfilerStop())

        # Warmup
        for _ in range(num_warmup):
            out1 = layernorm(hidden_states)
            out1.sum().backward()
            hidden_states.grad = None

        torch.cuda.synchronize()
        check_error(cudart().cudaProfilerStart())
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start = time.time()
            torch.cuda.nvtx.range_push("TE LayerNorm Forward")
            out2 = layernorm(hidden_states)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            layernorm_fwd_time = (time.time() - start) * 1000

            torch.cuda.synchronize()
            start = time.time()
            torch.cuda.nvtx.range_push("TE LayerNorm Backward")
            out2.sum().backward()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            layernorm_bwd_time = (time.time() - start) * 1000
            hidden_states.grad = None
        torch.cuda.synchronize()
        check_error(cudart().cudaProfilerStop())

        results.append(
            {
                "B": B,
                "S": S,
                "D": D,
                "rmsnorm_fwd_ms": rmsnorm_fwd_time / num_iters,
                "rmsnorm_bwd_ms": rmsnorm_bwd_time / num_iters,
                "layernorm_fwd_ms": layernorm_fwd_time / num_iters,
                "layernorm_bwd_ms": layernorm_bwd_time / num_iters,
            }
        )

    return results


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


if __name__ == "__main__":
    args = get_args()
    results = benchmark_te_layernorm(
        num_warmup=args.num_warmups,
        num_iters=args.num_runs,
    )
    if args.draw:
        draw_performance_plots(results, output_dir=args.output_dir)

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from transformer_engine.pytorch.module.linear import Linear

# ================================================================
# Customizable Linear Module
# ================================================================


def draw_performance_plots(results, output_dir="outputs/test/custom/linear"):
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

    fig = plt.figure(figsize=(20, 8))

    metrics = [
        ("fwd_ms", "Linear Forward"),
        ("bwd_ms", "Linear Backward"),
    ]

    for idx, (metric, title) in enumerate(metrics, 1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")

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
    plt.savefig(f"{output_dir}/linear_performance_bs.png", dpi=150)
    print(f"Performance plot saved to {output_dir}/linear_performance_bs.png")
    
    # 2D plots: D vs Time for different (B, S) combinations
    fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        
        # For each unique (B, S) combination, plot D vs time
        bs_combinations = sorted(set((r["B"], r["S"]) for r in results))
        
        for B, S in bs_combinations:
            d_vals, times = [], []
            for D in D_vals:
                if (B, S, D) in data_dict:
                    d_vals.append(D)
                    times.append(data_dict[(B, S, D)][metric])
            if d_vals:
                ax.plot(d_vals, times, marker='o', label=f"B={B}, S={S}")
        
        ax.set_xlabel("Hidden Size (D)")
        ax.set_ylabel("Time (ms)")
        ax.set_title(title)
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/linear_performance_d.png", dpi=150)
    print(f"Performance plot saved to {output_dir}/linear_performance_d.png")


def test_linear(
    batch_sizes: Tuple[int] = (1, 2, 4, 8, 16, 32),
    seqlens: Tuple[int] = (256, 512, 1024, 2048, 3072, 4096, 6144, 8192),
    hidden_sizes: Tuple[int] = (128, 256, 512, 1024, 2048, 3072, 4096),
    num_warmup: int = 50,
    num_iters: int = 100,
):
    device = torch.device("cuda")

    results = []
    for batch_size in batch_sizes:
        for seqlen in seqlens:
            for hidden_size in hidden_sizes:
                x = torch.randn(
                    batch_size, seqlen, hidden_size, device=device, dtype=torch.float16
                )
                linear = Linear(hidden_size, hidden_size).to(device).half()
                optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

                # Warm-up
                for _ in range(num_warmup):
                    optimizer.zero_grad()
                    y = linear(x)
                    loss = y.sum()
                    loss.backward()
                    optimizer.step()

                # Timing
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                for _ in range(num_iters):
                    optimizer.zero_grad()
                    y = linear(x)
                end_event.record()

                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
                avg_fwd_time_per_iter = elapsed_time / num_iters

                torch.cuda.synchronize()
                start_event.record()
                for _ in range(num_iters):
                    optimizer.zero_grad()
                    y = linear(x)
                    loss = y.sum()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
                avg_bwd_time_per_iter = elapsed_time / num_iters - avg_fwd_time_per_iter

                results.append(
                    {
                        "B": batch_size,
                        "S": seqlen,
                        "D": hidden_size,
                        "fwd_ms": avg_fwd_time_per_iter,
                        "bwd_ms": avg_bwd_time_per_iter,
                    }
                )

    return results


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Linear Benchmarking")
    parser.add_argument(
        "--num_warmup", type=int, default=20, help="Number of warm-up iterations."
    )
    parser.add_argument(
        "--num_iters", type=int, default=100, help="Number of benchmark iterations."
    )
    parser.add_argument(
        "--draw", action="store_true", help="Whether to draw the plots."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/test/custom/linear",
        help="Directory to save the benchmark results and plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    results = test_linear(num_warmup=args.num_warmup, num_iters=args.num_iters)
    if args.draw:
        draw_performance_plots(results, args.output_dir)

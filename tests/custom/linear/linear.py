from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from transformer_engine.pytorch.module.linear import Linear

# ================================================================
# Customizable Linear Module
# ================================================================


def plot_results(results, output_dir):
    """
    Plot forward and backward pass times in 3D.

    Args:
        results: List of tuples (batch_size, seqlen, hidden_size, fwd_time, bwd_time)
    """
    # Group results by batch size
    batch_size_groups = {}
    for batch_size, seqlen, hidden_size, fwd_time, bwd_time in results:
        if batch_size not in batch_size_groups:
            batch_size_groups[batch_size] = []
        batch_size_groups[batch_size].append((seqlen, hidden_size, fwd_time, bwd_time))

    # Create two subplots for forward and backward times
    fig = plt.figure(figsize=(16, 6))

    # Forward pass plot
    ax1 = fig.add_subplot(121, projection="3d")
    for batch_size, data in sorted(batch_size_groups.items()):
        seqlens = [d[0] for d in data]
        hidden_sizes = [d[1] for d in data]
        fwd_times = [d[2] for d in data]
        ax1.plot(
            seqlens, hidden_sizes, fwd_times, marker="o", label=f"Batch={batch_size}"
        )

    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Hidden Size")
    ax1.set_zlabel("Forward Time (ms)")
    ax1.set_title("Forward Pass Time")
    ax1.legend()

    # Backward pass plot
    ax2 = fig.add_subplot(122, projection="3d")
    for batch_size, data in sorted(batch_size_groups.items()):
        seqlens = [d[0] for d in data]
        hidden_sizes = [d[1] for d in data]
        bwd_times = [d[3] for d in data]
        ax2.plot(
            seqlens, hidden_sizes, bwd_times, marker="o", label=f"Batch={batch_size}"
        )

    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Hidden Size")
    ax2.set_zlabel("Backward Time (ms)")
    ax2.set_title("Backward Pass Time")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/linear_benchmark_results.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def test_linear(
    batch_sizes: Tuple[int] = (1, 2, 4, 8, 16, 32),
    seqlens: Tuple[int] = (256, 512, 1024, 2048, 4096, 8192),
    hidden_sizes: Tuple[int] = (128, 256, 512, 1024, 2048, 4096),
    num_warmup: int = 50,
    num_iters: int = 100,
    output_dir: str = "outputs/test/custom/linear",
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
                    linear.zero_grad_parameters()
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
                avg_bwd_time_per_iter = elapsed_time / num_iters - avg_fwd_time_per_iter

                results.append(
                    (
                        batch_size,
                        seqlen,
                        hidden_size,
                        avg_fwd_time_per_iter,
                        avg_bwd_time_per_iter,
                    )
                )

    plot_results(results, output_dir)


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
        "--output_dir",
        type=str,
        default="outputs/test/custom/linear",
        help="Directory to save the benchmark results and plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    test_linear(
        num_warmup=args.num_warmup, num_iters=args.num_iters, output_dir=args.output_dir
    )

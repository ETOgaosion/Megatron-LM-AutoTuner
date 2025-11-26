# flash_attention_benchmark.py
import argparse
import os
import time

# Check if flash-attn is available
import numpy as np
import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from torch.cuda import check_error, cudart
from tqdm import tqdm

FLASH_AVAILABLE = True


def benchmark_flash_attention(
    batch_sizes,
    seq_lens,
    num_heads=8,
    head_dim=64,
    dropout=0.0,
    causal=True,
    dtype=torch.float16,
    device="cuda",
    warmup=5,
    iterations=20,
):
    results = []

    print(
        f"{'Batch':<8} {'SeqLen':<8} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Total (ms)':<12} {'Memory (GB)':<12}"
    )
    print("-" * 75)

    for bsz in batch_sizes:
        for seqlen in seq_lens:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            check_error(cudart().cudaProfilerStop())

            # Create inputs
            q = torch.randn(
                bsz,
                seqlen,
                num_heads,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            k = torch.randn_like(q, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn_like(q, device=device, dtype=dtype, requires_grad=True)

            # Warmup
            for _ in range(warmup):
                out = flash_attn_func(q, k, v, dropout_p=dropout, causal=causal)
                out.sum().backward()

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            # Benchmark forward
            fwd_times = []
            check_error(cudart().cudaProfilerStart())
            for _ in range(iterations):
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push("FlashAttention Forward")
                start = time.time()
                out = flash_attn_func(q, k, v, dropout_p=dropout, causal=causal)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                fwd_times.append((time.time() - start) * 1000)
            fwd_time = sum(fwd_times) / len(fwd_times)

            # Benchmark backward
            torch.cuda.synchronize()
            check_error(cudart().cudaProfilerStart())
            bwd_times = []
            for _ in range(iterations):
                out = flash_attn_func(q, k, v, dropout_p=dropout, causal=causal)
                torch.cuda.synchronize()
                check_error(cudart().cudaProfilerStart())
                torch.cuda.nvtx.range_push("FlashAttention Backward")
                start = time.time()
                out.sum().backward(retain_graph=True)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                bwd_times.append((time.time() - start) * 1000)
            bwd_time = sum(bwd_times) / len(bwd_times)
            check_error(cudart().cudaProfilerStop())

            total_time = fwd_time + bwd_time
            memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB

            print(
                f"bsz: {bsz:<8} seqlen: {seqlen:<8} fwd_time: {fwd_time:<15.3f} bwd_time: {bwd_time:<15.3f} total_time: {total_time:<12.3f} memory: {memory:<12.2f}"
            )

            results.append(
                {
                    "batch_size": bsz,
                    "seq_len": seqlen,
                    "forward_ms": fwd_time,
                    "backward_ms": bwd_time,
                    "total_ms": total_time,
                    "memory_gb": memory,
                }
            )

    return results


def plot_results(args, results):
    """Plot forward/backward time vs batch size and sequence length"""
    import matplotlib.pyplot as plt

    # Extract unique batch sizes and sequence lengths
    batch_sizes = sorted(list(set([r["batch_size"] for r in results])))
    seq_lens = sorted(list(set([r["seq_len"] for r in results])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Time vs Batch Size (for different sequence lengths)
    for seqlen in seq_lens:
        data = [r for r in results if r["seq_len"] == seqlen]
        data = sorted(data, key=lambda x: x["batch_size"])
        bsz = [r["batch_size"] for r in data]
        fwd = [r["forward_ms"] for r in data]
        bwd = [r["backward_ms"] for r in data]

        ax1.plot(bsz, fwd, marker="o", label=f"FWD (seqlen={seqlen})")
        ax1.plot(bsz, bwd, marker="s", linestyle="--", label=f"BWD (seqlen={seqlen})")

    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Forward/Backward Time vs Batch Size")
    ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax1.grid(True, alpha=0.3)

    # Plot 2: Time vs Sequence Length (for different batch sizes)
    for bsz in batch_sizes:
        data = [r for r in results if r["batch_size"] == bsz]
        data = sorted(data, key=lambda x: x["seq_len"])
        seqs = [r["seq_len"] for r in data]
        fwd = [r["forward_ms"] for r in data]
        bwd = [r["backward_ms"] for r in data]

        ax2.plot(seqs, fwd, marker="o", label=f"FWD (bsz={bsz})")
        ax2.plot(seqs, bwd, marker="s", linestyle="--", label=f"BWD (bsz={bsz})")

    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Forward/Backward Time vs Sequence Length")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{args.output_dir}/flash_attention_benchmark.png", dpi=150, bbox_inches="tight"
    )
    print(f"\nPlot saved to: {args.output_dir}/flash_attention_benchmark.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention-2")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32],
        help="List of batch sizes to test",
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192],
        help="List of sequence lengths to test",
    )
    parser.add_argument(
        "--heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument("--dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--no-causal",
        action="store_false",
        dest="causal",
        help="Disable causal mask (use bidirectional)",
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16 instead of float16"
    )

    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=20, help="Number of benchmark iterations"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/test/custom/flash_attention",
        help="Output directory",
    )
    parser.add_argument(
        "--draw", action="store_true", help="Plot the benchmark results"
    )

    args = parser.parse_args()

    if not FLASH_AVAILABLE:
        raise RuntimeError("FlashAttention not available. Install it first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("FlashAttention requires CUDA")
    with open("tmp.txt", "w") as f:
        f.write(f"device: {device}\n")

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    print(f"Running FlashAttention-2 Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Data type: {dtype}")
    print(f"Causal: {args.causal}")
    print(f"Heads: {args.heads}, Head dim: {args.dim}")
    print()

    results = benchmark_flash_attention(
        batch_sizes=args.batch_sizes,
        seq_lens=args.seq_lens,
        num_heads=args.heads,
        head_dim=args.dim,
        dropout=0.0,
        causal=args.causal,
        dtype=dtype,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print("Benchmark results:")
    for res in results:
        print(res)

    if args.draw:
        plot_results(args, results)


if __name__ == "__main__":
    main()

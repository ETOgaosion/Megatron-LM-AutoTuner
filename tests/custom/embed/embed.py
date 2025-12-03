import os
from multiprocessing import Process, Queue
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D

DEFAULT_BATCH_SIZES = (2,)
DEFAULT_SEQ_LENGTHS = (1024, 2048, 3072, 4096, 8192)
DEFAULT_VOCAB_SIZES = (4096, 8192, 16384, 32768, 65536)
DEFAULT_HIDDEN_SIZES = (512, 1024, 1280, 2048, 2560, 4096)


def plot_results(results, output_dir: str):
    """Plot benchmark results in 3D subplots for forward and backward passes.

    Args:
        results: List of dictionaries containing benchmark results.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 6))

    # Extract data
    results = sorted(
        results, key=lambda x: (x["hidden_size"], x["seq_length"], x["vocab_size"])
    )
    seq_lengths = [r["seq_length"] for r in results]
    vocab_sizes = [r["vocab_size"] for r in results]
    forward_times = [r["forward_time_ms"] for r in results]
    backward_times = [r["backward_time_ms"] for r in results]

    # Divide data by number of hidden sizes to create second dimension
    num_hidden_sizes = len(DEFAULT_HIDDEN_SIZES)

    # Reshape data into groups based on hidden sizes
    seq_lengths_grouped = [
        seq_lengths[i : i + num_hidden_sizes]
        for i in range(0, len(seq_lengths), num_hidden_sizes)
    ]
    vocab_sizes_grouped = [
        vocab_sizes[i : i + num_hidden_sizes]
        for i in range(0, len(vocab_sizes), num_hidden_sizes)
    ]
    forward_times_grouped = [
        forward_times[i : i + num_hidden_sizes]
        for i in range(0, len(forward_times), num_hidden_sizes)
    ]
    backward_times_grouped = [
        backward_times[i : i + num_hidden_sizes]
        for i in range(0, len(backward_times), num_hidden_sizes)
    ]

    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    for i in range(len(results) // num_hidden_sizes):
        # ax1.plot(seq_lengths_grouped[i], vocab_sizes_grouped[i], forward_times_grouped[i], marker='o')
        # ax2.plot(seq_lengths_grouped[i], vocab_sizes_grouped[i], backward_times_grouped[i], marker='o')
        ax1.scatter(seq_lengths, vocab_sizes, forward_times, marker="o")
        ax2.scatter(seq_lengths, vocab_sizes, backward_times, marker="o")

    # Forward pass subplot
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Vocabulary Size")
    ax1.set_zlabel("Forward Time (ms)")
    ax1.set_title("Forward Pass Time")

    # Backward pass subplot
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Vocabulary Size")
    ax2.set_zlabel("Backward Time (ms)")
    ax2.set_title("Backward Pass Time")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/embedding_benchmark_results.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def benchmark_embedding(
    pid,
    queue: Queue,
    batch_sizes: Tuple[int] = (2,),
    seq_lengths: Tuple[int] = (1024, 2048, 3072, 4096, 8192),
    vocab_size: Tuple[int] = (4096, 8192, 16384, 32768, 65536),
    hidden_size: Tuple[int] = (512, 1024, 1280, 2048, 2560, 4096),
    num_warmup: int = 10,
    num_iters: int = 5,
):
    """Benchmark embedding layer memory usage and time.

    Args:
        batch_sizes (Tuple[int]): Batch sizes to benchmark.
        seq_lengths (Tuple[int]): Sequence lengths to benchmark.
        vocab_size (Tuple[int]): Vocabulary sizes to benchmark.
        hidden_size (Tuple[int]): Hidden sizes to benchmark.
    """
    results = []

    configs = []
    for bsz in batch_sizes:
        for seq_len in seq_lengths:
            for vocab in vocab_size:
                for hidden in hidden_size:
                    configs.append((bsz, seq_len, vocab, hidden))

    half_num_configs_per_gpu = (len(configs) + 2 * torch.cuda.device_count() - 1) // (
        2 * torch.cuda.device_count()
    )
    configs_per_gpu = configs[
        pid * half_num_configs_per_gpu : (pid + 1) * half_num_configs_per_gpu
    ]
    configs_per_gpu.extend(
        configs[-(pid + 1) * half_num_configs_per_gpu : -pid * half_num_configs_per_gpu]
    )

    for bsz, seq_len, vocab, hidden in configs_per_gpu:
        embedding = torch.nn.Embedding(vocab, hidden).cuda(pid)
        input_ids = torch.randint(
            low=0, high=vocab, size=(bsz, seq_len), device=f"cuda:{pid}"
        )

        # Warm-up iterations
        for _ in range(num_warmup):
            _ = embedding(input_ids)

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.nvtx.range_push(
            f"Embedding BS{bsz}_SL{seq_len}_V{vocab}_H{hidden} Forward"
        )
        start_event.record()
        for _ in range(num_iters):
            output = embedding(input_ids)
        end_event.record()

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        forward_elapsed_time_ms = start_event.elapsed_time(end_event) / num_iters

        for _ in range(num_warmup):
            output = embedding(input_ids)
            grad_output = torch.randn_like(output)
            output.backward(grad_output)

        torch.cuda.nvtx.range_push(
            f"Embedding BS{bsz}_SL{seq_len}_V{vocab}_H{hidden} Backward"
        )
        start_event.record()
        for _ in range(num_iters):
            output = embedding(input_ids)
            grad_output = torch.randn_like(output)
            output.backward(grad_output, retain_graph=True)
        end_event.record()

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        backward_elapsed_time_ms = (
            start_event.elapsed_time(end_event) / num_iters - forward_elapsed_time_ms
        )

        print(
            f"Batch Size: {bsz}, Seq Length: {seq_len}, Vocab Size: {vocab}, Hidden Size: {hidden} => Forward Time: {forward_elapsed_time_ms:.3f} ms, Backward Time: {backward_elapsed_time_ms:.3f} ms"
        )

        results.append(
            {
                "batch_size": bsz,
                "seq_length": seq_len,
                "vocab_size": vocab,
                "hidden_size": hidden,
                "forward_time_ms": forward_elapsed_time_ms,
                "backward_time_ms": backward_elapsed_time_ms,
            }
        )
    queue.put((pid, results))
    return results


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Embedding Benchmarking Tool")
    parser.add_argument(
        "--draw", action="store_true", help="Whether to plot the results."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/test/custom/embed",
        help="Directory to save benchmark results and plots.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    queue = Queue()
    processes = []
    print(
        f"Starting embedding benchmark processes on {torch.cuda.device_count()} GPUs..."
    )
    for pid in range(torch.cuda.device_count()):
        p = Process(target=benchmark_embedding, args=(pid, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = []
    while not queue.empty():
        pid, res = queue.get()
        results.extend(res)

    if args.draw:
        plot_results(results, args.output_dir)

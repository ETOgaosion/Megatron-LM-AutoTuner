# gpu_cpu_bandwidth_bi.py
import argparse
import time

import numpy as np
import torch


def measure_once(
    num_bytes: int,
    direction: str = "d2h",
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    pinned: bool = False,
    non_blocking: bool = True,
) -> float:
    """Measure single transfer time (seconds). direction ∈ {"d2h", "h2d"}"""
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    num_elems = num_bytes // bytes_per_elem

    if direction == "d2h":
        # GPU → CPU
        x = torch.empty(num_elems, dtype=dtype, device=device)
        x.uniform_(0, 1)  # touch memory
        if pinned:
            y = torch.empty(num_elems, dtype=dtype, device="cpu", pin_memory=True)
            torch.cuda.synchronize()
            t0 = time.time()
            y.copy_(x, non_blocking=non_blocking)
            torch.cuda.synchronize()
            t1 = time.time()
        else:
            torch.cuda.synchronize()
            t0 = time.time()
            y = x.cpu()
            torch.cuda.synchronize()
            t1 = time.time()

    elif direction == "h2d":
        # CPU → GPU
        if pinned:
            x = torch.empty(num_elems, dtype=dtype, device="cpu", pin_memory=True)
        else:
            x = torch.empty(num_elems, dtype=dtype, device="cpu")
        x.uniform_(0, 1)
        y = torch.empty(num_elems, dtype=dtype, device=device)

        torch.cuda.synchronize()
        t0 = time.time()
        y.copy_(x, non_blocking=non_blocking)
        torch.cuda.synchronize()
        t1 = time.time()
    else:
        raise ValueError("direction must be 'd2h' or 'h2d'")

    return t1 - t0


def test_sizes(
    sizes_gb: list[float],
    repeats: int = 3,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
) -> dict[str, list[dict[str, float]]]:
    results = {
        "d2h_pageable": [],
        "d2h_pinned": [],
        "h2d_pageable": [],
        "h2d_pinned": [],
    }

    for direction in ["d2h", "h2d"]:
        for pinned in [False, True]:
            key = f"{direction}_{'pinned' if pinned else 'pageable'}"
            for size_gb in sizes_gb:
                num_bytes = int(size_gb * (1024**3))
                times = []
                for r in range(repeats):
                    t = measure_once(
                        num_bytes,
                        direction=direction,
                        dtype=dtype,
                        device=device,
                        pinned=pinned,
                    )
                    times.append(t)
                best = min(times)
                mean = sum(times) / len(times)
                bw_best = size_gb / best
                bw_mean = size_gb / mean
                results[key].append(
                    {
                        "size_gb": size_gb,
                        "best_s": best,
                        "mean_s": mean,
                        "bw_best_gbps": bw_best,
                        "bw_mean_gbps": bw_mean,
                    }
                )
                print(
                    f"[{key}] {size_gb} GB: best {best:.4f}s ({bw_best:.2f} GB/s), "
                    f"mean {mean:.4f}s ({bw_mean:.2f} GB/s)"
                )
    return results


def fit_bandwidth(records: list[dict[str, float]]) -> tuple[float, float]:
    sizes = np.array([r["size_gb"] for r in records])
    times = np.array([r["best_s"] for r in records])
    A = np.vstack([np.ones_like(sizes), sizes]).T
    coef, *_ = np.linalg.lstsq(A, times, rcond=None)
    overhead, inv_bw = coef
    bw = 1.0 / inv_bw if inv_bw > 0 else float("inf")
    return overhead, bw


def print_summary(results: dict[str, list[dict[str, float]]]) -> None:
    for key, recs in results.items():
        overhead, bw = fit_bandwidth(recs)
        print(
            f"\n[{key}] Linear fit: Bandwidth ≈ {bw:.2f} GB/s, Overhead ≈ {overhead*1000:.2f} ms"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure GPU<->CPU transfer bandwidth")
    parser.add_argument(
        "--sizes", nargs="+", type=float, default=[10, 15, 20, 25], help="sizes in GB"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("Torch:", torch.__version__)
    results = test_sizes(args.sizes, repeats=args.repeats, device=args.device)
    print_summary(results)

    print("\nTheoretical PCIe bandwidths for comparison:")
    print("PCIe 3.0 x16 ≈ 15.75 GB/s")
    print("PCIe 4.0 x16 ≈ 31.5 GB/s")
    print("PCIe 5.0 x16 ≈ 63.0 GB/s")

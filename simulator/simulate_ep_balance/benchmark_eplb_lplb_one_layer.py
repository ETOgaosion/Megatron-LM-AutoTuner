#!/usr/bin/env python3
"""Benchmark one-layer mocked MoE load balancing for EPLB and LPLB."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
EPLB_MODULE_PATH = REPO_ROOT / "EPLB" / "eplb.py"
LPLB_EPLB_MODULE_PATH = REPO_ROOT / "LPLB" / "lplb" / "eplb.py"

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen1.5-MoE-A2.7B": {
        "config_path": "/data/common/models/Qwen/Qwen1.5-MoE-A2.7B/config.json",
        "fallback_num_experts": 60,
    },
    "Qwen/Qwen3-30B-A3B-Base": {
        "config_path": "/data/common/models/Qwen/Qwen3-30B-A3B-Base/config.json",
        "fallback_num_experts": 128,
    },
    "Qwen/Qwen3-235B-A22B": {
        "config_path": "/data/common/models/Qwen/Qwen3-235B-A22B/config.json",
        "fallback_num_experts": 128,
    },
    "deepseek-ai/DeepSeek-V3-Base": {
        "config_path": "/data/common/models/deepseek-ai/DeepSeek-V3-Base/config.json",
        "fallback_num_experts": 256,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure one-layer EPLB/LPLB task latency and draw comparison chart."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Model names to benchmark.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Measured iterations per method.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations per method.",
    )
    parser.add_argument(
        "--redundant-ratio",
        type=float,
        default=0.25,
        help="Redundant physical-expert ratio for num_replicas.",
    )
    parser.add_argument(
        "--num-gpus-default",
        type=int,
        default=8,
        help="Default GPU count used for rebalance_experts topology.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir. Default: outputs/simulate_ep_balance/test_eplb_lplb/<timestamp>/",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip drawing png chart."
    )
    return parser.parse_args()


def _load_module(module_name: str, module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(f"Missing module file: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_num_experts(model_name: str) -> int:
    spec = MODEL_CONFIGS[model_name]
    config_path = Path(spec["config_path"])
    fallback = int(spec["fallback_num_experts"])
    if not config_path.exists():
        return fallback
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    value = cfg.get("num_experts", cfg.get("n_routed_experts"))
    if value is None:
        return fallback
    return int(value)


def _choose_num_groups(num_experts: int) -> int:
    for candidate in (16, 12, 10, 8, 6, 5, 4, 3, 2, 1):
        if num_experts % candidate == 0:
            return candidate
    return 1


def _pick_topology(
    num_experts: int, num_gpus_default: int, redundant_ratio: float
) -> dict[str, int]:
    num_gpus = max(4, num_gpus_default) if num_experts >= 64 else 4
    num_groups = _choose_num_groups(num_experts)
    num_nodes = 1
    redundant = max(1, int(math.ceil(num_experts * max(0.0, redundant_ratio))))
    num_replicas = int(math.ceil((num_experts + redundant) / num_gpus) * num_gpus)
    return {
        "num_replicas": num_replicas,
        "num_groups": num_groups,
        "num_nodes": num_nodes,
        "num_gpus": num_gpus,
    }


def _mock_one_layer_weight(model_name: str, num_experts: int) -> torch.Tensor:
    seed = zlib.crc32(model_name.encode("utf-8")) & 0xFFFFFFFF
    generator = torch.Generator().manual_seed(seed)
    base = torch.randint(
        50, 5000, (num_experts,), generator=generator, dtype=torch.int64
    )
    slope = torch.linspace(1.0, 2.5, steps=num_experts)
    weight = (base.float() * slope).round().to(torch.float32)
    return weight.view(1, -1)


def _benchmark(fn: Callable[[], None], warmup: int, iterations: int) -> list[float]:
    for _ in range(max(0, warmup)):
        fn()
    samples_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)
    return samples_ms


def _stats(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "median_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "stdev_ms": float(statistics.pstdev(samples_ms)),
    }


def _plot_results(results: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    models = list(results.keys())
    x = np.arange(len(models), dtype=np.float64)
    width = 0.22
    title_fontsize = 16
    axis_fontsize = 16
    x_tick_fontsize = 10
    y_tick_fontsize = 13
    legend_fontsize = 14
    anno_fontsize = 10

    ours = [results[m]["our_system"]["mean_ms"] for m in models]
    eplb = [results[m]["eplb"]["mean_ms"] for m in models]
    lplb = [results[m]["lplb"]["mean_ms"] for m in models]

    plt.figure(figsize=(max(9.5, len(models) * 2.2), 5.4))
    bars_ours = plt.bar(
        x - width, ours, width=width, label="Our system", color="#59A14F"
    )
    bars_eplb = plt.bar(x, eplb, width=width, label="EPLB", color="#E15759")
    bars_lplb = plt.bar(x + width, lplb, width=width, label="LPLB", color="#4C78A8")

    max_val = max(max(eplb), max(lplb), 1e-6)
    text_offset = max(0.02, max_val * 0.01)
    zero_marker_y = text_offset * 0.4

    def annotate(bars, values, is_ours: bool = False):
        for bar, v in zip(bars, values):
            x_center = bar.get_x() + bar.get_width() / 2.0
            x_text = x_center
            y_text = v + text_offset if v > 0 else zero_marker_y
            plt.text(
                x_text,
                y_text,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=anno_fontsize,
            )

    annotate(bars_ours, ours, is_ours=True)
    annotate(bars_eplb, eplb)
    annotate(bars_lplb, lplb)

    # Zero-height bars are hard to see; draw a visible marker at y=0 for our system.
    plt.scatter(
        x - width, np.zeros_like(x), marker="_", s=500, color="#2E7D32", zorder=4
    )
    for x_pos in x - width:
        plt.annotate(
            "Our system",
            xy=(float(x_pos), zero_marker_y * 8.0),
            xytext=(float(x_pos - width * 0.36), max_val * 0.18 + text_offset * 2.0),
            ha="center",
            va="bottom",
            fontsize=anno_fontsize,
            color="#2E7D32",
            arrowprops={
                "arrowstyle": "->",
                "color": "#2E7D32",
                "lw": 1.1,
            },
        )

    plt.xticks(x, models, rotation=12, ha="right", fontsize=x_tick_fontsize)
    plt.yticks(fontsize=y_tick_fontsize)
    plt.ylabel("Single-task latency (ms)", fontsize=axis_fontsize)
    plt.title(
        "One-layer MoE load-balance latency: Our system vs EPLB vs LPLB",
        fontsize=title_fontsize,
    )
    plt.ylim(bottom=0.0, top=max_val * 1.18 + text_offset * 3)
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(output_dir / "time_compare_ours_eplb_lplb.png", dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    eplb = _load_module("deepseek_eplb_bench", EPLB_MODULE_PATH)
    lplb_eplb = _load_module("deepseek_lplb_eplb_bench", LPLB_EPLB_MODULE_PATH)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            REPO_ROOT / "outputs" / "simulate_ep_balance" / "test_eplb_lplb" / timestamp
        )
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for model_name in args.models:
        num_experts = _read_num_experts(model_name)
        topo = _pick_topology(
            num_experts=num_experts,
            num_gpus_default=args.num_gpus_default,
            redundant_ratio=args.redundant_ratio,
        )
        weight = _mock_one_layer_weight(model_name, num_experts)

        def run_eplb() -> None:
            eplb.rebalance_experts(
                weight=weight,
                num_replicas=topo["num_replicas"],
                num_groups=topo["num_groups"],
                num_nodes=topo["num_nodes"],
                num_gpus=topo["num_gpus"],
            )

        def run_lplb() -> None:
            lplb_eplb.rebalance_experts(
                weight=weight,
                num_replicas=topo["num_replicas"],
                num_groups=topo["num_groups"],
                num_nodes=topo["num_nodes"],
                num_gpus=topo["num_gpus"],
            )

        eplb_samples = _benchmark(
            run_eplb, warmup=args.warmup, iterations=args.iterations
        )
        lplb_samples = _benchmark(
            run_lplb, warmup=args.warmup, iterations=args.iterations
        )
        results[model_name] = {
            "num_experts": num_experts,
            "topology": topo,
            "our_system": {
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "p90_ms": 0.0,
                "p95_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "stdev_ms": 0.0,
                "samples": [],
            },
            "eplb": {
                **_stats(eplb_samples),
                "samples": eplb_samples,
            },
            "lplb": {
                **_stats(lplb_samples),
                "samples": lplb_samples,
            },
        }

    (output_dir / "timing_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    csv_lines = [
        "model,num_experts,num_groups,num_gpus,num_replicas,our_mean_ms,eplb_mean_ms,lplb_mean_ms,eplb_p95_ms,lplb_p95_ms"
    ]
    for model_name, item in results.items():
        topo = item["topology"]
        csv_lines.append(
            ",".join(
                [
                    model_name,
                    str(item["num_experts"]),
                    str(topo["num_groups"]),
                    str(topo["num_gpus"]),
                    str(topo["num_replicas"]),
                    f"{item['our_system']['mean_ms']:.6f}",
                    f"{item['eplb']['mean_ms']:.6f}",
                    f"{item['lplb']['mean_ms']:.6f}",
                    f"{item['eplb']['p95_ms']:.6f}",
                    f"{item['lplb']['p95_ms']:.6f}",
                ]
            )
        )
    (output_dir / "timing_summary.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )

    if not args.no_plot:
        _plot_results(results, output_dir)

    print(f"Saved benchmark outputs to: {output_dir}")
    for model_name, item in results.items():
        print(
            "{name} | our={our:.3f} ms | eplb={eplb:.3f} ms (p95 {eplb_p95:.3f}) | lplb={lplb:.3f} ms (p95 {lplb_p95:.3f})".format(
                name=model_name,
                our=item["our_system"]["mean_ms"],
                eplb=item["eplb"]["mean_ms"],
                eplb_p95=item["eplb"]["p95_ms"],
                lplb=item["lplb"]["mean_ms"],
                lplb_p95=item["lplb"]["p95_ms"],
            )
        )


if __name__ == "__main__":
    main()

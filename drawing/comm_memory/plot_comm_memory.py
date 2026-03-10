#!/usr/bin/env python3
"""Draw communication memory utilization from JSON data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required to draw figures. Install it with: pip install matplotlib"
    ) from exc


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("outputs") / "drawing" / "comm_memory" / timestamp / "comm_memory.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot communication memory with dual y-axes (total bar + runtime bar)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(__file__).with_name("data.json"),
        help="Input JSON path. Default: drawing/comm_memory/data.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output_path(),
        help="Output image path. Default: outputs/drawing/comm_memory/<timestamp>/comm_memory.png",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    parser.add_argument("--font-size", type=float, default=12.0, help="Base font size.")
    return parser.parse_args()


def load_data(path: Path) -> tuple[list[str], list[float], list[float]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    runtime = payload["runtime_memory"]
    offload = payload["offload_memory"]

    systems = list(runtime.keys())
    runtime_values = [float(runtime[name]) for name in systems]
    offload_values = [float(offload[name]) for name in systems]
    total_values = [runtime_v + offload_v for runtime_v, offload_v in zip(runtime_values, offload_values)]

    return systems, runtime_values, total_values


def _build_axis_limits(values: list[float], lower_pad_ratio: float, upper_pad_ratio: float) -> tuple[float, float]:
    min_v = min(values)
    max_v = max(values)
    span = max(1.0, max_v - min_v)
    y_min = min_v - lower_pad_ratio * span
    if y_min == 0:
        y_min = 0.01
    y_max = max_v + upper_pad_ratio * span
    return y_min, y_max


def plot(
    systems: list[str],
    runtime_values: list[float],
    total_values: list[float],
    output_path: Path,
    dpi: int,
    font_size: float,
) -> None:
    plt.rcParams.update({"font.size": font_size})
    x = list(range(len(systems)))
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax_runtime = ax.twinx()
    total_label_size = max(11.0, font_size)
    runtime_label_size = max(10.0, font_size - 1.0)

    bar_width = 0.26
    total_x = [idx - bar_width / 2 for idx in x]
    runtime_x = [idx + bar_width / 2 for idx in x]
    bars_total = ax.bar(
        total_x,
        total_values,
        color="#F58518",
        label="total",
        width=bar_width,
        alpha=0.85,
    )
    bars_runtime = ax_runtime.bar(
        runtime_x,
        runtime_values,
        color="#4C78A8",
        label="runtime",
        width=bar_width,
        alpha=0.9,
    )

    for bar, total_v in zip(bars_total, total_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            total_v + 0.35,
            f"{total_v:.2f} GB",
            ha="center",
            va="bottom",
            fontsize=total_label_size,
            color="#1f1f1f",
        )
    for bar, runtime_v in zip(bars_runtime, runtime_values):
        ax_runtime.text(
            bar.get_x() + bar.get_width() / 2,
            runtime_v + 0.20,
            f"{runtime_v:.2f}",
            ha="center",
            va="bottom",
            fontsize=runtime_label_size,
            color="#2E5E8C",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=font_size)
    ax.set_ylabel("Total Memory (GB)", fontsize=font_size + 1.0, color="#A04E09")
    ax_runtime.set_ylabel("Runtime Memory (GB)", fontsize=font_size + 1.0, color="#2E5E8C")
    ax.set_title(
        "Memory Saving with Memory-Efficient Communication Library",
        fontsize=font_size + 2.0,
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors="#A04E09")
    ax_runtime.tick_params(axis="y", colors="#2E5E8C")
    handles, labels = ax.get_legend_handles_labels()
    runtime_handles, runtime_labels = ax_runtime.get_legend_handles_labels()
    ax.legend(handles + runtime_handles, labels + runtime_labels, loc="upper right", fontsize=font_size)

    shared_min, shared_max = _build_axis_limits(
        total_values + runtime_values, lower_pad_ratio=0.8, upper_pad_ratio=0.5
    )
    ax.set_ylim(shared_min, shared_max)
    ax_runtime.set_ylim(shared_min, shared_max)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    systems, runtime_values, total_values = load_data(args.input)
    plot(
        systems=systems,
        runtime_values=runtime_values,
        total_values=total_values,
        output_path=args.output,
        dpi=args.dpi,
        font_size=args.font_size,
    )
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()

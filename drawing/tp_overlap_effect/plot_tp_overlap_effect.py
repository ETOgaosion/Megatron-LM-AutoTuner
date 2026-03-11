#!/usr/bin/env python3
"""Draw TP overlap overhead from local data.json."""

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
    return (
        Path("outputs")
        / "drawing"
        / "tp_overlap_effect"
        / timestamp
        / "tp_overlap_effect.png"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TP overlap overhead using original vs optimized latency."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(__file__).with_name("data.json"),
        help="Input JSON path. Default: drawing/tp_overlap_effect/data.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output_path(),
        help="Output image path. Default: outputs/drawing/tp_overlap_effect/<timestamp>/tp_overlap_effect.png",
    )
    parser.add_argument(
        "--title", type=str, default="TP Overlap Overhead", help="Figure title."
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=18.0,
        help="Base font size (default is large).",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    return parser.parse_args()


def load_data(
    path: Path,
) -> tuple[list[str], list[float], list[float], list[float], str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    models: list[str] = []
    original: list[float] = []
    optimized: list[float] = []
    blank: list[float] = []
    unit = "value"

    for item in payload:
        model = item.get("model")
        data = item.get("data", {})
        original_value = data.get("original")
        optimized_value = data.get("our")
        blank_value = data.get("blank")
        current_unit = item.get("unit")

        if (
            not isinstance(model, str)
            or not isinstance(original_value, (int, float))
            or not isinstance(optimized_value, (int, float))
            or not isinstance(blank_value, (int, float))
        ):
            raise ValueError(
                f"Invalid record in {path}: each item needs model + numeric data.original/data.our/data.blank"
            )

        models.append(model)
        original.append(float(original_value))
        optimized.append(float(optimized_value))
        blank.append(float(blank_value))
        if isinstance(current_unit, str) and current_unit:
            unit = current_unit

    if not models:
        raise ValueError(f"No valid data found in: {path}")

    return models, original, optimized, blank, unit


def plot(
    models: list[str],
    original: list[float],
    optimized: list[float],
    blank: list[float],
    unit: str,
    output_path: Path,
    title: str,
    font_size: float,
    dpi: int,
) -> None:
    plt.rcParams.update({"font.size": font_size})
    x = list(range(len(models)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.bar(
        [idx - width for idx in x],
        original,
        width=width,
        color="#9AA6B2",
        label="original",
        alpha=0.95,
    )
    ax.bar(
        x,
        optimized,
        width=width,
        color="#2A9D8F",
        label="our",
        alpha=0.95,
    )
    ax.bar(
        [idx + width for idx in x],
        blank,
        width=width,
        color="#E9C46A",
        label="blank",
        alpha=0.95,
    )

    all_values = original + optimized + blank
    min_value = min(all_values)
    max_value = max(all_values)
    span = max(1.0, max_value - min_value)
    y_min = min_value - 0.12 * span
    y_max = max_value + 0.30 * span
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=font_size)
    ax.set_ylabel(f"Latency ({unit})", fontsize=font_size + 1.0)
    ax.set_title(title, fontsize=font_size + 3.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=font_size)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    models, original, optimized, blank, unit = load_data(args.input)
    plot(
        models=models,
        original=original,
        optimized=optimized,
        blank=blank,
        unit=unit,
        output_path=args.output,
        title=args.title,
        font_size=args.font_size,
        dpi=args.dpi,
    )
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()

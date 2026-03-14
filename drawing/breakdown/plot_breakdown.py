#!/usr/bin/env python3
"""Draw per-machine latency breakdown for baseline vs ours."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required to draw figures. Install it with: pip install matplotlib"
    ) from exc


SEGMENTS = ("attn", "moe", "other")
SEGMENT_COLORS = {
    "attn": "#4C78A8",
    "moe": "#F58518",
    "other": "#9AA6B2",
}
SYSTEMS = ("baseline", "ours")
SYSTEM_LABELS = {"baseline": "Baseline", "ours": "Ours"}


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("outputs") / "drawing" / "breakdown" / timestamp / "breakdown.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot breakdown latency for baseline vs ours on each machine."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(__file__).with_name("data.json"),
        help="Input JSON path. Default: drawing/breakdown/data.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output_path(),
        help="Output image path. Default: outputs/drawing/breakdown/<timestamp>/breakdown.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Latency Breakdown: Baseline vs Ours",
        help="Figure title.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=14.0,
        help="Base font size.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    return parser.parse_args()


def _extract_duration(record: dict[str, object], key: str) -> tuple[float, str | None]:
    value = record.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be an object, got: {type(value)!r}")

    duration = value.get("duration")
    unit = value.get("unit")
    if not isinstance(duration, (int, float)):
        raise ValueError(f"Expected '{key}.duration' to be numeric, got: {duration!r}")
    if unit is not None and not isinstance(unit, str):
        raise ValueError(f"Expected '{key}.unit' to be a string, got: {unit!r}")

    return float(duration), unit


def load_data(path: Path) -> tuple[list[str], dict[str, list[float]], str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty list in: {path}")

    machines: list[str] = []
    series = {
        f"{system}_{segment}": []
        for system in SYSTEMS
        for segment in SEGMENTS
    }
    series.update({f"{system}_total": [] for system in SYSTEMS})
    expected_unit: str | None = None

    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Each record must be an object, got: {type(item)!r}")

        machine = item.get("machine")
        if not isinstance(machine, str) or not machine:
            raise ValueError(f"Each record must contain a non-empty 'machine': {item!r}")
        machines.append(machine)

        for system in SYSTEMS:
            values = item.get(system)
            if not isinstance(values, dict):
                raise ValueError(f"Record for {machine} is missing '{system}' data")

            attn, attn_unit = _extract_duration(values, "attn")
            moe, moe_unit = _extract_duration(values, "moe")
            total, total_unit = _extract_duration(values, "total")
            unit_candidates = [unit for unit in (attn_unit, moe_unit, total_unit) if unit]

            if unit_candidates:
                current_unit = unit_candidates[0]
                if any(unit != current_unit for unit in unit_candidates):
                    raise ValueError(
                        f"Inconsistent units for machine={machine}, system={system}: {unit_candidates}"
                    )
                if expected_unit is None:
                    expected_unit = current_unit
                elif expected_unit != current_unit:
                    raise ValueError(
                        f"Mixed units across records: expected {expected_unit}, got {current_unit}"
                    )

            other = total - attn - moe
            if other < -1e-6:
                raise ValueError(
                    f"total is smaller than attn + moe for machine={machine}, system={system}"
                )
            other = max(0.0, other)

            series[f"{system}_attn"].append(attn)
            series[f"{system}_moe"].append(moe)
            series[f"{system}_other"].append(other)
            series[f"{system}_total"].append(total)

    return machines, series, expected_unit or "value"


def plot(
    machines: list[str],
    series: dict[str, list[float]],
    unit: str,
    output_path: Path,
    title: str,
    font_size: float,
    dpi: int,
) -> None:
    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    x = list(range(len(machines)))
    width = 0.28
    offsets = {"baseline": -width / 1.6, "ours": width / 1.6}
    alphas = {"baseline": 0.55, "ours": 0.92}
    hatches = {"baseline": "//", "ours": None}

    all_totals = series["baseline_total"] + series["ours_total"]
    y_max = max(all_totals) * 1.24
    ax.set_ylim(0.0, y_max)

    for system in SYSTEMS:
        bottoms = [0.0] * len(machines)
        positions = [idx + offsets[system] for idx in x]

        for segment in SEGMENTS:
            heights = series[f"{system}_{segment}"]
            bars = ax.bar(
                positions,
                heights,
                width=width,
                bottom=bottoms,
                color=SEGMENT_COLORS[segment],
                alpha=alphas[system],
                hatch=hatches[system],
                edgecolor="#3A3A3A" if system == "baseline" else "none",
                linewidth=0.6 if system == "baseline" else 0.0,
            )

            for bar, height, bottom in zip(bars, heights, bottoms):
                if height <= 0:
                    continue
                if height >= y_max * 0.06:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom + height / 2,
                        f"{height:.1f}",
                        ha="center",
                        va="center",
                        fontsize=max(font_size - 2.0, 9.0),
                        color="white" if segment != "other" else "#1F1F1F",
                    )
            bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

        for xpos, total in zip(positions, series[f"{system}_total"]):
            ax.text(
                xpos,
                total + y_max * 0.018,
                f"{total:.1f}",
                ha="center",
                va="bottom",
                fontsize=font_size,
                fontweight="bold",
                color="#1F1F1F",
            )

    for idx, machine in enumerate(machines):
        baseline_total = series["baseline_total"][idx]
        ours_total = series["ours_total"][idx]
        reduction = (baseline_total - ours_total) / baseline_total * 100.0
        ax.text(
            idx,
            max(baseline_total, ours_total) + y_max * 0.08,
            f"-{reduction:.1f}%",
            ha="center",
            va="bottom",
            fontsize=font_size,
            color="#12715B",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(machines, fontsize=font_size + 1.0)
    ax.set_ylabel(f"Latency ({unit})", fontsize=font_size + 3.0)
    ax.set_title(title, fontsize=font_size + 5.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    ax.set_axisbelow(True)

    segment_handles = [
        Patch(facecolor=SEGMENT_COLORS[segment], label=segment.capitalize())
        for segment in SEGMENTS
    ]
    system_handles = [
        Patch(
            facecolor="#D9D9D9",
            edgecolor="#3A3A3A",
            hatch="//",
            label=SYSTEM_LABELS["baseline"],
        ),
        Patch(facecolor="#D9D9D9", edgecolor="#3A3A3A", label=SYSTEM_LABELS["ours"]),
    ]
    legend_segments = ax.legend(
        handles=segment_handles,
        loc="upper left",
        bbox_to_anchor=(0.58, 0.99),
        ncol=1,
        title="Components",
        fontsize=max(font_size - 1.0, 10.0),
        title_fontsize=max(font_size - 0.5, 10.0),
        frameon=False,
    )
    legend_systems = ax.legend(
        handles=system_handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.99),
        ncol=1,
        title="Systems",
        fontsize=max(font_size - 1.0, 10.0),
        title_fontsize=max(font_size - 0.5, 10.0),
        frameon=False,
    )
    ax.add_artist(legend_segments)

    for idx, system in enumerate(SYSTEMS):
        label_y = y_max * 0.015
        for x_pos in [value + offsets[system] for value in x]:
            ax.text(
                x_pos,
                label_y,
                SYSTEM_LABELS[system],
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=max(font_size - 3.0, 9.0),
                color="#4A4A4A",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    machines, series, unit = load_data(args.input)
    plot(
        machines=machines,
        series=series,
        unit=unit,
        output_path=args.output,
        title=args.title,
        font_size=args.font_size,
        dpi=args.dpi,
    )
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()

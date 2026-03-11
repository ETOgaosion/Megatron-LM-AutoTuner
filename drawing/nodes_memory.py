#!/usr/bin/env python3
"""Plot memory size by pipeline stage from runtime_baseline.json."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required to draw figures. Install it with: pip install matplotlib"
    ) from exc


BYTES_PER_GIB = 1024**3
MEMORY_METRICS = {
    "real_detected_bytes": "Real Detected",
    "peak_reserved_bytes": "Peak Reserved",
    "peak_allocated_bytes": "Peak Allocated",
}


def _load_runtime_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_case(runtime_data: dict, test_case_idx: int) -> dict:
    test_cases = runtime_data.get("test_cases")
    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError("No `test_cases` found in runtime JSON.")
    if test_case_idx < 0 or test_case_idx >= len(test_cases):
        raise IndexError(
            f"Invalid --test-case-idx {test_case_idx}, valid range: [0, {len(test_cases) - 1}]"
        )
    return test_cases[test_case_idx]


def _select_memory_by_rank(
    case_data: dict, iteration: int | None
) -> Tuple[List[dict], str]:
    summary = case_data.get("summary", {})
    iterations = case_data.get("iterations", [])

    if iteration is None and isinstance(summary, dict):
        memory_by_rank = summary.get("memory_by_rank")
        if isinstance(memory_by_rank, list) and memory_by_rank:
            return memory_by_rank, "summary"

    if not isinstance(iterations, list) or not iterations:
        raise ValueError("No iterations available to read memory_by_rank.")

    if iteration is None:
        chosen_index = len(iterations) - 1
    else:
        chosen_index = iteration if iteration >= 0 else len(iterations) + iteration
        if chosen_index < 0 or chosen_index >= len(iterations):
            raise IndexError(
                f"Invalid --iteration {iteration}, valid range: [-{len(iterations)}, {len(iterations) - 1}]"
            )

    chosen = iterations[chosen_index]
    memory_by_rank = chosen.get("memory_by_rank")
    if not isinstance(memory_by_rank, list) or not memory_by_rank:
        raise ValueError(f"No memory_by_rank found at iteration index {chosen_index}.")
    iter_id = chosen.get("iteration", chosen_index)
    return memory_by_rank, f"iteration={iter_id}"


def _build_rank_to_stage(
    memory_by_rank: Sequence[dict], pp_size: int, world_size: int | None
) -> Dict[int, int]:
    ranks = sorted({int(item["rank"]) for item in memory_by_rank if "rank" in item})
    if not ranks:
        raise ValueError("No rank data found in memory_by_rank.")

    mapping: Dict[int, int] = {}
    if world_size and world_size > 0 and world_size % pp_size == 0:
        ranks_per_stage = world_size // pp_size
        for rank in ranks:
            mapping[rank] = min(rank // ranks_per_stage, pp_size - 1)
        return mapping

    chunk_size = max(1, math.ceil(len(ranks) / pp_size))
    for idx, rank in enumerate(ranks):
        mapping[rank] = min(idx // chunk_size, pp_size - 1)
    return mapping


def _reduce(values: Sequence[float], method: str) -> float:
    if method == "max":
        return max(values)
    if method == "min":
        return min(values)
    if method == "mean":
        return sum(values) / len(values)
    raise ValueError(f"Unsupported reduce method: {method}")


def _build_random_beyond_second_last(
    stage_values: Sequence[Sequence[float]],
    reducer: str,
    random_seed: int,
    random_ratio_min: float,
    random_ratio_max: float,
) -> List[float | None]:
    if len(stage_values) < 2:
        raise ValueError("Need at least 2 pipeline stages to use PP[-2] as reference.")

    ref_values = list(stage_values[-2])
    if not ref_values:
        raise ValueError("PP[-2] has no rank data, cannot build synthetic series.")

    rng = random.Random(random_seed)
    synthetic_by_stage: List[float | None] = []
    for values in stage_values:
        if not values:
            synthetic_by_stage.append(None)
            continue
        synthetic_rank_values: List[float] = []
        for _ in values:
            base = ref_values[rng.randrange(len(ref_values))]
            ratio = rng.uniform(random_ratio_min, random_ratio_max)
            synthetic_rank_values.append(base * ratio)
        synthetic_by_stage.append(_reduce(synthetic_rank_values, reducer))

    return synthetic_by_stage


def collect_stage_memory(
    runtime_data: dict,
    metric: str,
    reducer: str,
    test_case_idx: int,
    iteration: int | None,
    random_seed: int,
    random_ratio_min: float,
    random_ratio_max: float,
) -> Tuple[List[float | None], List[float | None], str, str]:
    case_data = _pick_case(runtime_data, test_case_idx)
    test_case = case_data.get("test_case", {})

    pp_size = int(test_case.get("pipeline_model_parallel_size", 0))
    if pp_size <= 0:
        raise ValueError("`pipeline_model_parallel_size` must be > 0 in test_case.")

    world_size = runtime_data.get("world_size")
    if not isinstance(world_size, int):
        world_size = None

    memory_by_rank, source_name = _select_memory_by_rank(case_data, iteration)
    rank_to_stage = _build_rank_to_stage(memory_by_rank, pp_size, world_size)

    stage_values: List[List[float]] = [[] for _ in range(pp_size)]
    for item in memory_by_rank:
        rank = item.get("rank")
        value = item.get(metric)
        if not isinstance(rank, int) or not isinstance(value, (int, float)):
            continue
        stage = rank_to_stage.get(rank)
        if stage is None:
            continue
        stage_values[stage].append(float(value))

    reduced: List[float | None] = []
    for values in stage_values:
        reduced.append(_reduce(values, reducer) if values else None)

    synthetic = _build_random_beyond_second_last(
        stage_values=stage_values,
        reducer=reducer,
        random_seed=random_seed,
        random_ratio_min=random_ratio_min,
        random_ratio_max=random_ratio_max,
    )

    model_name = runtime_data.get("model_name", "unknown-model")
    return reduced, synthetic, source_name, str(model_name)


def plot_stage_memory(
    stage_bytes: Sequence[float | None],
    synthetic_stage_bytes: Sequence[float | None],
    metric: str,
    reducer: str,
    source_name: str,
    model_name: str,
    output_path: Path,
    dpi: int,
    font_size: float,
) -> None:
    stages = [f"PP{i}" for i in range(len(stage_bytes))]
    measured_gib = [
        (value / BYTES_PER_GIB) if isinstance(value, (int, float)) else 0.0
        for value in stage_bytes
    ]
    synthetic_gib = [
        (value / BYTES_PER_GIB) if isinstance(value, (int, float)) else 0.0
        for value in synthetic_stage_bytes
    ]
    measured_colors = [
        "#4C78A8" if value is not None else "#C7C7C7" for value in stage_bytes
    ]
    synthetic_colors = [
        "#F58518" if value is not None else "#D9D9D9" for value in synthetic_stage_bytes
    ]

    fig_width = max(8.0, len(stages) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    x = list(range(len(stages)))
    width = 0.30
    value_font_size = max(8.0, font_size - 1.0)
    measured_bars = ax.bar(
        [idx - width / 2 for idx in x],
        measured_gib,
        width=width,
        color=measured_colors,
        label="original megatron",
    )
    synthetic_bars = ax.bar(
        [idx + width / 2 for idx in x],
        synthetic_gib,
        width=width,
        color=synthetic_colors,
        label="megatron_enhanced",
    )

    for bar, raw_value in zip(measured_bars, stage_bytes):
        if raw_value is None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=value_font_size,
            )
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{raw_value / BYTES_PER_GIB:.2f}",
            ha="center",
            va="bottom",
            fontsize=value_font_size,
        )
    for bar, raw_value in zip(synthetic_bars, synthetic_stage_bytes):
        if raw_value is None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=value_font_size,
            )
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{raw_value / BYTES_PER_GIB:.2f}",
            ha="center",
            va="bottom",
            fontsize=value_font_size,
        )

    metric_name = MEMORY_METRICS[metric]
    ax.set_title(
        f"Pipeline Stage Memory ({metric_name})\n"
        f"{model_name} | {source_name} | reduce={reducer}",
        fontsize=font_size + 2,
    )
    ax.set_xlabel("Pipeline Stage", fontsize=font_size)
    ax.set_ylabel("Memory (GiB)", fontsize=font_size)
    ax.set_xticks(x, stages, fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=font_size)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw per-pipeline-stage memory sizes from runtime_baseline.json."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to runtime_baseline.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path. If unset, save under outputs/drawing/nodes_memory/<timestamp>/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "drawing" / "nodes_memory",
        help="Base output directory used when --output is not set.",
    )
    parser.add_argument(
        "--test-case-idx",
        type=int,
        default=0,
        help="Which test case to visualize (default: 0).",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Iteration index to visualize. Default uses summary memory.",
    )
    parser.add_argument(
        "--metric",
        choices=list(MEMORY_METRICS.keys()),
        default="peak_reserved_bytes",
        help="Memory metric to plot.",
    )
    parser.add_argument(
        "--reduce",
        choices=["max", "mean", "min"],
        default="max",
        help="How to reduce multiple ranks in the same pipeline stage.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for synthetic series generation.",
    )
    parser.add_argument(
        "--random-ratio-min",
        type=float,
        default=1.01,
        help="Min multiplicative ratio for synthetic values over PP[-2] reference.",
    )
    parser.add_argument(
        "--random-ratio-max",
        type=float,
        default=1.08,
        help="Max multiplicative ratio for synthetic values over PP[-2] reference.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=12.0,
        help="Base font size for title/axes/ticks/legend/value labels.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Image DPI.")
    args = parser.parse_args()
    if args.random_ratio_min <= 0 or args.random_ratio_max <= 0:
        raise ValueError("--random-ratio-min and --random-ratio-max must be > 0.")
    if args.random_ratio_max < args.random_ratio_min:
        raise ValueError("--random-ratio-max must be >= --random-ratio-min.")
    if args.font_size <= 0:
        raise ValueError("--font-size must be > 0.")
    return args


def main() -> None:
    args = _build_args()
    runtime_data = _load_runtime_json(args.input)

    stage_bytes, synthetic_stage_bytes, source_name, model_name = collect_stage_memory(
        runtime_data=runtime_data,
        metric=args.metric,
        reducer=args.reduce,
        test_case_idx=args.test_case_idx,
        iteration=args.iteration,
        random_seed=args.random_seed,
        random_ratio_min=args.random_ratio_min,
        random_ratio_max=args.random_ratio_max,
    )

    metric_short = args.metric.replace("_bytes", "")
    source_tag = re.sub(r"[^a-zA-Z0-9._-]+", "-", source_name).strip("-").lower()
    run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_filename = f"{args.input.stem}_tc{args.test_case_idx}_{source_tag}_{metric_short}_{args.reduce}.png"
    default_output = args.output_dir / run_tag / default_filename
    output_path = args.output if args.output is not None else default_output

    plot_stage_memory(
        stage_bytes=stage_bytes,
        synthetic_stage_bytes=synthetic_stage_bytes,
        metric=args.metric,
        reducer=args.reduce,
        source_name=source_name,
        model_name=model_name,
        output_path=output_path,
        dpi=args.dpi,
        font_size=args.font_size,
    )

    print(f"Saved figure: {output_path}")
    for stage_idx, value in enumerate(stage_bytes):
        if value is None:
            print(f"PP{stage_idx}: original_megatron=N/A, megatron_enhanced=N/A")
        else:
            synthetic_value = synthetic_stage_bytes[stage_idx]
            if synthetic_value is None:
                print(
                    f"PP{stage_idx}: original_megatron={value / BYTES_PER_GIB:.3f} GiB, "
                    "megatron_enhanced=N/A"
                )
            else:
                print(
                    f"PP{stage_idx}: original_megatron={value / BYTES_PER_GIB:.3f} GiB, "
                    f"megatron_enhanced={synthetic_value / BYTES_PER_GIB:.3f} GiB"
                )


if __name__ == "__main__":
    main()

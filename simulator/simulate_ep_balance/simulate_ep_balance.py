#!/usr/bin/env python3
"""Simulate expert reordering to improve EP load balance."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LayerBalanceResult:
    old_gpu_tokens: list[int]
    new_gpu_tokens: list[int]
    old_variance: float
    new_variance: float
    old_expert_to_gpu: dict[str, int]
    new_expert_to_gpu: dict[str, int]
    old_gpu_to_experts: dict[str, list[int]]
    new_gpu_to_experts: dict[str, list[int]]


@dataclass
class EpBalanceResult:
    ep_size: int
    experts_per_gpu: int
    global_gpu_tokens_old: list[int]
    global_gpu_tokens_new: list[int]
    global_variance_old: float
    global_variance_new: float
    mean_layer_variance_old: float
    mean_layer_variance_new: float
    layers: dict[str, LayerBalanceResult]


def load_expert_tokens(data_path: Path) -> dict[int, dict[int, int]]:
    """Load routed experts stats and sum token counts per expert over all ranks."""
    with data_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    layer_to_expert_tokens: dict[int, dict[int, int]] = {}
    for layer_str, experts in raw.items():
        layer_id = int(layer_str)
        layer_tokens: dict[int, int] = {}
        for expert_str, rank_tokens in experts.items():
            expert_id = int(expert_str)
            layer_tokens[expert_id] = int(sum(int(v) for v in rank_tokens.values()))
        layer_to_expert_tokens[layer_id] = layer_tokens
    return layer_to_expert_tokens


def contiguous_mapping(num_experts: int, ep_size: int) -> list[int]:
    experts_per_gpu = num_experts // ep_size
    mapping: list[int] = []
    for expert_idx in range(num_experts):
        mapping.append(expert_idx // experts_per_gpu)
    return mapping


def mapping_to_gpu_lists(expert_ids: list[int], mapping: list[int], ep_size: int) -> dict[str, list[int]]:
    gpu_to_experts: dict[str, list[int]] = {str(i): [] for i in range(ep_size)}
    for idx, gpu_id in enumerate(mapping):
        gpu_to_experts[str(gpu_id)].append(expert_ids[idx])
    for experts in gpu_to_experts.values():
        experts.sort()
    return gpu_to_experts


def mapping_to_expert_dict(expert_ids: list[int], mapping: list[int]) -> dict[str, int]:
    return {str(expert_ids[idx]): gpu_id for idx, gpu_id in enumerate(mapping)}


def gpu_tokens(values: list[int], mapping: list[int], ep_size: int) -> list[int]:
    tokens = [0 for _ in range(ep_size)]
    for idx, value in enumerate(values):
        tokens[mapping[idx]] += int(value)
    return tokens


def _sort_state_desc(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(state, key=lambda x: (x["sum"], -len(x["items"])), reverse=True)


def kk_partition_equal_size(values: list[int], k_partitions: int) -> list[list[int]]:
    """Equal-size multi-way partition using Karmarkar-Karp differencing heuristic."""
    n_items = len(values)
    if n_items % k_partitions != 0:
        raise ValueError(f"{n_items} experts cannot be evenly split across EP={k_partitions}.")

    indexed = sorted([(int(v), i) for i, v in enumerate(values)])
    states: list[list[dict[str, Any]]] = []

    for offset in range(0, n_items, k_partitions):
        bins: list[dict[str, Any]] = [{"sum": 0, "items": []} for _ in range(k_partitions)]
        for i in range(k_partitions):
            value, idx = indexed[offset + i]
            bins[i] = {"sum": value, "items": [idx]}
        states.append(_sort_state_desc(bins))

    def spread(state: list[dict[str, Any]]) -> int:
        return int(state[0]["sum"] - state[-1]["sum"])

    while len(states) > 1:
        states.sort(key=spread, reverse=True)
        state0 = states.pop(0)
        state1 = states.pop(0)

        merged: list[dict[str, Any]] = []
        for i in range(k_partitions):
            left = state0[i]
            right = state1[k_partitions - 1 - i]
            merged.append(
                {
                    "sum": int(left["sum"] + right["sum"]),
                    "items": left["items"] + right["items"],
                }
            )

        states.append(_sort_state_desc(merged))

    final_state = states[0]
    partitions = [sorted(bin_data["items"]) for bin_data in final_state]

    expected_size = n_items // k_partitions
    if any(len(part) != expected_size for part in partitions):
        raise RuntimeError("Partitioning failed to keep equal expert count per GPU.")

    return partitions


def partition_sums(partitions: list[list[int]], values: list[int]) -> list[int]:
    return [int(sum(values[idx] for idx in part)) for part in partitions]


def refine_partitions_by_swaps(
    partitions: list[list[int]],
    values: list[int],
    max_iters: int = 200,
) -> list[list[int]]:
    """Refine equal-size partitions with pairwise swaps to reduce variance."""
    refined = [list(part) for part in partitions]
    sums = partition_sums(refined, values)

    for _ in range(max_iters):
        current_var = float(np.var(sums))
        best: tuple[int, int, int, int] | None = None
        best_var = current_var

        for left in range(len(refined)):
            for right in range(left + 1, len(refined)):
                left_sum = sums[left]
                right_sum = sums[right]
                for left_item in refined[left]:
                    left_value = values[left_item]
                    for right_item in refined[right]:
                        right_value = values[right_item]
                        cand_left_sum = left_sum - left_value + right_value
                        cand_right_sum = right_sum - right_value + left_value

                        cand_sums = list(sums)
                        cand_sums[left] = int(cand_left_sum)
                        cand_sums[right] = int(cand_right_sum)
                        cand_var = float(np.var(cand_sums))

                        if cand_var + 1e-12 < best_var:
                            best_var = cand_var
                            best = (left, right, left_item, right_item)

        if best is None:
            break

        left, right, left_item, right_item = best
        refined[left].remove(left_item)
        refined[left].append(right_item)
        refined[right].remove(right_item)
        refined[right].append(left_item)
        sums = partition_sums(refined, values)

    for part in refined:
        part.sort()
    return refined


def assign_partitions_to_gpus(
    partitions: list[list[int]],
    values: list[int],
    cumulative_gpu_tokens: list[int],
) -> tuple[list[int], list[int]]:
    """Assign partition buckets to GPU ranks to avoid fixed-rank load bias."""
    ep_size = len(partitions)
    part_sums = partition_sums(partitions, values)

    part_to_gpu: dict[int, int] = {}
    layer_gpu_tokens = [0 for _ in range(ep_size)]

    # Place larger partitions first onto currently lightest cumulative ranks.
    for part_idx in sorted(range(ep_size), key=lambda i: part_sums[i], reverse=True):
        target_gpu = min(range(ep_size), key=lambda g: (cumulative_gpu_tokens[g], layer_gpu_tokens[g], g))
        part_to_gpu[part_idx] = target_gpu
        layer_gpu_tokens[target_gpu] = part_sums[part_idx]
        cumulative_gpu_tokens[target_gpu] += part_sums[part_idx]

    num_experts = sum(len(part) for part in partitions)
    mapping = [-1 for _ in range(num_experts)]
    for part_idx, part in enumerate(partitions):
        gpu_id = part_to_gpu[part_idx]
        for expert_idx in part:
            mapping[expert_idx] = gpu_id

    if any(x < 0 for x in mapping):
        raise RuntimeError("Invalid partition-to-GPU assignment: some experts were not assigned.")

    return mapping, layer_gpu_tokens


def simulate_for_ep(layer_to_expert_tokens: dict[int, dict[int, int]], ep_size: int) -> EpBalanceResult:
    first_layer = min(layer_to_expert_tokens.keys())
    expert_ids = sorted(layer_to_expert_tokens[first_layer].keys())
    num_experts = len(expert_ids)

    if num_experts % ep_size != 0:
        raise ValueError(
            f"EP size {ep_size} is invalid for {num_experts} experts: requires exact divisibility."
        )

    old_mapping = contiguous_mapping(num_experts=num_experts, ep_size=ep_size)
    experts_per_gpu = num_experts // ep_size

    layer_results: dict[str, LayerBalanceResult] = {}
    global_old = [0 for _ in range(ep_size)]
    global_new = [0 for _ in range(ep_size)]

    old_layer_vars: list[float] = []
    new_layer_vars: list[float] = []

    for layer_id in sorted(layer_to_expert_tokens.keys()):
        layer_tokens = layer_to_expert_tokens[layer_id]
        values = [int(layer_tokens[eid]) for eid in expert_ids]

        new_partitions = kk_partition_equal_size(values=values, k_partitions=ep_size)
        new_partitions = refine_partitions_by_swaps(new_partitions, values=values)
        new_mapping, new_gpu = assign_partitions_to_gpus(
            partitions=new_partitions,
            values=values,
            cumulative_gpu_tokens=global_new,
        )

        old_gpu = gpu_tokens(values=values, mapping=old_mapping, ep_size=ep_size)

        for i in range(ep_size):
            global_old[i] += old_gpu[i]

        old_var = float(np.var(old_gpu))
        new_var = float(np.var(new_gpu))
        old_layer_vars.append(old_var)
        new_layer_vars.append(new_var)

        layer_results[str(layer_id)] = LayerBalanceResult(
            old_gpu_tokens=old_gpu,
            new_gpu_tokens=new_gpu,
            old_variance=old_var,
            new_variance=new_var,
            old_expert_to_gpu=mapping_to_expert_dict(expert_ids=expert_ids, mapping=old_mapping),
            new_expert_to_gpu=mapping_to_expert_dict(expert_ids=expert_ids, mapping=new_mapping),
            old_gpu_to_experts=mapping_to_gpu_lists(
                expert_ids=expert_ids,
                mapping=old_mapping,
                ep_size=ep_size,
            ),
            new_gpu_to_experts=mapping_to_gpu_lists(
                expert_ids=expert_ids,
                mapping=new_mapping,
                ep_size=ep_size,
            ),
        )

    return EpBalanceResult(
        ep_size=ep_size,
        experts_per_gpu=experts_per_gpu,
        global_gpu_tokens_old=global_old,
        global_gpu_tokens_new=global_new,
        global_variance_old=float(np.var(global_old)),
        global_variance_new=float(np.var(global_new)),
        mean_layer_variance_old=float(np.mean(old_layer_vars)),
        mean_layer_variance_new=float(np.mean(new_layer_vars)),
        layers=layer_results,
    )


def plot_ep_gpu_tokens(ep_result: EpBalanceResult, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ep_size = ep_result.ep_size
    old_gpu = np.array(ep_result.global_gpu_tokens_old, dtype=np.int64)
    new_gpu = np.array(ep_result.global_gpu_tokens_new, dtype=np.int64)

    x = np.arange(ep_size)
    width = 0.38

    plt.figure(figsize=(max(8, ep_size * 0.6), 4.8))
    plt.bar(x - width / 2, old_gpu, width=width, label="Before reorder")
    plt.bar(x + width / 2, new_gpu, width=width, label="After reorder")
    plt.xlabel("GPU rank")
    plt.ylabel("Token count (sum over all layers)")
    plt.title(f"EP={ep_size} GPU token balance")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"gpu_token_balance_ep{ep_size}.png", dpi=200)
    plt.close()


def plot_ep_variance_summary(ep_results: list[EpBalanceResult], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    eps = [r.ep_size for r in ep_results]
    global_old = [r.global_variance_old for r in ep_results]
    global_new = [r.global_variance_new for r in ep_results]
    layer_old = [r.mean_layer_variance_old for r in ep_results]
    layer_new = [r.mean_layer_variance_new for r in ep_results]

    plt.figure(figsize=(10, 5))
    plt.plot(eps, global_old, marker="o", linewidth=2, label="Global variance (before)")
    plt.plot(eps, global_new, marker="o", linewidth=2, label="Global variance (after)")
    plt.plot(
        eps,
        layer_old,
        marker="s",
        linestyle="--",
        linewidth=1.6,
        label="Mean layer variance (before)",
    )
    plt.plot(
        eps,
        layer_new,
        marker="s",
        linestyle="--",
        linewidth=1.6,
        label="Mean layer variance (after)",
    )
    plt.xlabel("EP size")
    plt.ylabel("Variance")
    plt.title("Variance comparison across EP sizes")
    plt.xticks(eps)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "variance_comparison_across_ep_sizes.png", dpi=220)
    plt.close()


def valid_ep_sizes(num_experts: int) -> list[int]:
    return [ep for ep in range(2, num_experts + 1) if num_experts % ep == 0]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Simulate expert reordering for EP load balancing.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=script_dir / "data" / "routed_experts_stats.json",
        help="Path to routed expert token stats JSON.",
    )
    parser.add_argument(
        "--ep-sizes",
        type=int,
        nargs="+",
        default=None,
        help="EP sizes to evaluate. Defaults to all divisors of expert count (>=2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: outputs/simulate_ep_balance/<timestamp>/",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    layer_to_expert_tokens = load_expert_tokens(args.data_path)
    if not layer_to_expert_tokens:
        raise RuntimeError("Loaded empty token stats.")

    first_layer = min(layer_to_expert_tokens.keys())
    num_experts = len(layer_to_expert_tokens[first_layer])

    requested_eps = args.ep_sizes if args.ep_sizes is not None else valid_ep_sizes(num_experts)
    if not requested_eps:
        raise RuntimeError("No EP sizes available to evaluate.")

    ep_sizes = sorted(set(requested_eps))
    for ep in ep_sizes:
        if ep <= 0 or num_experts % ep != 0:
            raise ValueError(f"Invalid EP size {ep} for {num_experts} experts.")

    if args.output_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = repo_root / "outputs" / "simulate_ep_balance" / timestamp
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[EpBalanceResult] = []
    for ep in ep_sizes:
        all_results.append(simulate_for_ep(layer_to_expert_tokens=layer_to_expert_tokens, ep_size=ep))

    serializable = {str(result.ep_size): asdict(result) for result in all_results}
    with (output_dir / "balance_results.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    summary_lines = [
        "ep_size,experts_per_gpu,global_var_before,global_var_after,mean_layer_var_before,mean_layer_var_after"
    ]
    for result in all_results:
        summary_lines.append(
            ",".join(
                [
                    str(result.ep_size),
                    str(result.experts_per_gpu),
                    f"{result.global_variance_old:.6f}",
                    f"{result.global_variance_new:.6f}",
                    f"{result.mean_layer_variance_old:.6f}",
                    f"{result.mean_layer_variance_new:.6f}",
                ]
            )
        )
    (output_dir / "variance_summary.csv").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if not args.no_plot:
        for result in all_results:
            plot_ep_gpu_tokens(ep_result=result, output_dir=output_dir)
        plot_ep_variance_summary(ep_results=all_results, output_dir=output_dir)

    print(f"Saved outputs to: {output_dir}")
    for result in all_results:
        print(
            "EP={ep} | global var {old:.2f} -> {new:.2f} | mean layer var {old_l:.2f} -> {new_l:.2f}".format(
                ep=result.ep_size,
                old=result.global_variance_old,
                new=result.global_variance_new,
                old_l=result.mean_layer_variance_old,
                new_l=result.mean_layer_variance_new,
            )
        )


if __name__ == "__main__":
    main()

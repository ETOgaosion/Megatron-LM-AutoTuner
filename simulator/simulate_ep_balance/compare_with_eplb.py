#!/usr/bin/env python3
"""Compare in-repo EP balancing algorithm with EPLB on the same token stats."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "EPLB"))

import eplb  # noqa: E402
from simulate_ep_balance import (  # noqa: E402
    load_expert_tokens,
    simulate_for_ep,
    valid_ep_sizes,
)

COLOR_BEFORE = "#4C78A8"
COLOR_BEFORE_LIGHT = "#9ecae9"
COLOR_EPLB = "#E15759"
COLOR_EPLB_LIGHT = "#f3b0b1"
COLOR_OURS = "#59A14F"
COLOR_OURS_LIGHT = "#b9dfb0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare our EP balancing with EPLB.")
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "compare_with_eplb_config.json",
        help="JSON config path. If present, used as primary config and CLI args can override.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to routed expert token stats JSON.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Model config for memory estimation.",
    )
    parser.add_argument(
        "--ep-sizes",
        type=int,
        nargs="+",
        default=None,
        help="EP sizes to compare. Defaults to all divisors of expert count (>=2).",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=None,
        help="Exact EPLB num_replicas (physical experts). If set, overrides ratio-based auto setting.",
    )
    parser.add_argument(
        "--eplb-redundant-ratio",
        type=float,
        default=None,
        help="EPLB redundancy ratio. Example: 0.2 means ~20%% extra physical experts.",
    )
    parser.add_argument(
        "--eplb-min-redundant",
        type=int,
        default=None,
        help="Minimum number of redundant experts added for EPLB.",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=None,
        help="EPLB num_groups parameter.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="EPLB num_nodes parameter.",
    )
    parser.add_argument(
        "--activation-multiplier",
        type=float,
        default=None,
        help="Multiplier for activation memory estimate.",
    )
    parser.add_argument(
        "--moe-layer-act-m",
        type=float,
        default=None,
        help="MoELayer activation size per layer in million elements (default 180).",
    )
    parser.add_argument(
        "--moe-layer-expert-act-m",
        type=float,
        default=None,
        help="Expert-dependent activation part per layer in million elements (default 66).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: outputs/simulate_ep_balance/compare_with_eplb/<timestamp>/",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def load_json_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def cli_or_cfg(cli_value: Any, cfg: dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return cfg.get(key, default)


def normalize_ep_param(param: Any, ep_size: int, field_name: str) -> int:
    if isinstance(param, str):
        if param == "same_as_ep":
            return ep_size
        raise ValueError(f"Unsupported string value for {field_name}: {param}")
    return int(param)


def build_ep_plan(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    num_experts: int,
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    raw_ep_settings = cfg.get("ep_settings", [])
    if raw_ep_settings and not isinstance(raw_ep_settings, list):
        raise ValueError("`ep_settings` in config must be a list.")

    ep_settings_map: dict[int, dict[str, Any]] = {}
    for item in raw_ep_settings:
        if "ep_size" not in item:
            raise ValueError("Each entry in `ep_settings` must contain `ep_size`.")
        ep = int(item["ep_size"])
        ep_settings_map[ep] = dict(item)

    if ep_settings_map:
        ep_sizes = sorted(ep_settings_map.keys())
        if args.ep_sizes is not None:
            requested = sorted(set(int(x) for x in args.ep_sizes))
            missing = [ep for ep in requested if ep not in ep_settings_map]
            if missing:
                raise ValueError(
                    f"EP sizes {missing} are not present in config `ep_settings`."
                )
            ep_sizes = requested
    else:
        requested_eps = (
            args.ep_sizes if args.ep_sizes is not None else cfg.get("ep_sizes")
        )
        if requested_eps is None:
            requested_eps = valid_ep_sizes(num_experts)
        ep_sizes = sorted(set(int(x) for x in requested_eps))

    ep_defaults = cfg.get("ep_defaults", {})
    if ep_defaults and not isinstance(ep_defaults, dict):
        raise ValueError("`ep_defaults` in config must be a dict.")

    ep_plan: dict[int, dict[str, Any]] = {}
    for ep in ep_sizes:
        if ep <= 0 or num_experts % ep != 0:
            raise ValueError(f"Invalid EP size {ep} for {num_experts} experts.")

        merged = dict(ep_defaults)
        merged.update(ep_settings_map.get(ep, {}))

        num_groups = cli_or_cfg(args.num_groups, merged, "num_groups", ep)
        num_nodes = cli_or_cfg(args.num_nodes, merged, "num_nodes", ep)

        ep_plan[ep] = {
            "num_replicas": cli_or_cfg(args.num_replicas, merged, "num_replicas", None),
            "eplb_redundant_ratio": float(
                cli_or_cfg(
                    args.eplb_redundant_ratio, merged, "eplb_redundant_ratio", 0.2
                )
            ),
            "eplb_min_redundant": int(
                cli_or_cfg(args.eplb_min_redundant, merged, "eplb_min_redundant", 0)
            ),
            "num_groups": normalize_ep_param(
                num_groups, ep_size=ep, field_name="num_groups"
            ),
            "num_nodes": normalize_ep_param(
                num_nodes, ep_size=ep, field_name="num_nodes"
            ),
        }
    return ep_sizes, ep_plan


def to_gib(num_bytes: float) -> float:
    return float(num_bytes) / (1024.0**3)


def dtype_nbytes(torch_dtype: str) -> int:
    mapping = {
        "bfloat16": 2,
        "float16": 2,
        "float32": 4,
    }
    if torch_dtype not in mapping:
        raise ValueError(f"Unsupported torch_dtype in model config: {torch_dtype}")
    return mapping[torch_dtype]


def expert_params_count(model_cfg: dict[str, Any]) -> int:
    hidden = int(model_cfg["hidden_size"])
    moe_hidden = int(model_cfg["moe_intermediate_size"])
    # SwiGLU expert MLP: gate_proj + up_proj + down_proj.
    return int(3 * hidden * moe_hidden)


def shared_expert_params_count(model_cfg: dict[str, Any]) -> int:
    hidden = int(model_cfg["hidden_size"])
    shared_hidden = int(model_cfg["shared_expert_intermediate_size"])
    # Shared expert MLP: gate_proj + up_proj + down_proj.
    return int(3 * hidden * shared_hidden)


def choose_num_replicas(
    num_experts: int,
    ep_size: int,
    redundant_ratio: float,
    min_redundant: int,
) -> int:
    target = int(math.ceil(num_experts * (1.0 + max(0.0, redundant_ratio))))
    target = max(target, num_experts + max(0, min_redundant))
    target = max(target, num_experts)
    return int(math.ceil(target / ep_size) * ep_size)


def resolve_num_replicas(
    *,
    explicit_num_replicas: int | None,
    num_experts: int,
    ep_size: int,
    redundant_ratio: float,
    min_redundant: int,
) -> int:
    if explicit_num_replicas is not None:
        num_replicas = int(explicit_num_replicas)
    else:
        num_replicas = choose_num_replicas(
            num_experts=num_experts,
            ep_size=ep_size,
            redundant_ratio=redundant_ratio,
            min_redundant=min_redundant,
        )

    if num_replicas < num_experts:
        raise ValueError(
            f"num_replicas={num_replicas} must be >= num_experts={num_experts} for EPLB replication."
        )
    if num_replicas % ep_size != 0:
        raise ValueError(
            f"num_replicas={num_replicas} must be divisible by ep_size={ep_size}."
        )
    return num_replicas


def max_per_gpu_from_layers(
    layer_map: dict[str, Any], key: str, ep_size: int
) -> list[float]:
    peak = [0.0 for _ in range(ep_size)]
    for layer in layer_map.values():
        values = layer[key]
        for g in range(ep_size):
            peak[g] = max(peak[g], float(values[g]))
    return peak


def estimate_memory(
    *,
    ep_size: int,
    num_moe_layers: int,
    physical_experts_per_gpu: int,
    peak_tokens_per_gpu: list[float],
    model_cfg: dict[str, Any],
    reference_peak_tokens: float,
    act_multiplier: float,
    moe_layer_act_m: float,
    moe_layer_expert_act_m: float,
) -> dict[str, Any]:
    params_per_expert = expert_params_count(model_cfg)
    params_shared_expert = shared_expert_params_count(model_cfg)
    bytes_per_elem = dtype_nbytes(str(model_cfg["torch_dtype"]))
    topk = int(model_cfg.get("num_experts_per_tok", 1))
    if moe_layer_expert_act_m > moe_layer_act_m:
        raise ValueError(
            "moe_layer_expert_act_m cannot be larger than moe_layer_act_m."
        )

    # Weight memory is resident for all MoE layers.
    # Include both replicated routed experts and non-replicated shared experts.
    weight_elems_per_layer = float(
        physical_experts_per_gpu * params_per_expert + params_shared_expert
    )
    weight_bytes = float(num_moe_layers * weight_elems_per_layer * bytes_per_elem)
    max_activation_tokens = float(max(peak_tokens_per_gpu))
    max_input_tokens = float(max_activation_tokens / max(1, topk))
    routed_scale = float(max_activation_tokens / max(reference_peak_tokens, 1.0))
    base_const_act_m = float(moe_layer_act_m - moe_layer_expert_act_m)
    layer_activation_m = float(base_const_act_m + moe_layer_expert_act_m * routed_scale)
    max_activation_bytes = float(
        num_moe_layers * layer_activation_m * 1_000_000.0 * bytes_per_elem
    )
    max_activation_bytes *= float(act_multiplier)
    max_total_bytes = float(weight_bytes + max_activation_bytes)

    return {
        "ep_size": ep_size,
        "num_moe_layers": int(num_moe_layers),
        "physical_experts_per_gpu": int(physical_experts_per_gpu),
        "weight_bytes_per_gpu": weight_bytes,
        "params_per_expert": int(params_per_expert),
        "params_shared_expert": int(params_shared_expert),
        "topk": int(topk),
        "moe_layer_act_m": float(moe_layer_act_m),
        "moe_layer_expert_act_m": float(moe_layer_expert_act_m),
        "routed_scale_vs_before_peak": routed_scale,
        "layer_activation_m": float(layer_activation_m),
        "reference_peak_tokens_per_gpu": float(reference_peak_tokens),
        "max_activation_tokens_per_gpu": max_activation_tokens,
        "max_input_tokens_per_gpu": max_input_tokens,
        "max_activation_bytes_per_gpu": max_activation_bytes,
        "max_total_bytes_per_gpu": max_total_bytes,
        "weight_gib_per_gpu": to_gib(weight_bytes),
        "max_activation_gib": to_gib(max_activation_bytes),
        "max_total_gib": to_gib(max_total_bytes),
    }


def simulate_eplb_for_ep(
    layer_to_expert_tokens: dict[int, dict[int, int]],
    ep_size: int,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
) -> dict[str, Any]:
    layer_ids = sorted(layer_to_expert_tokens.keys())
    first_layer = layer_ids[0]
    expert_ids = sorted(layer_to_expert_tokens[first_layer].keys())

    weight_np = np.array(
        [
            [float(layer_to_expert_tokens[layer_id][eid]) for eid in expert_ids]
            for layer_id in layer_ids
        ],
        dtype=np.float64,
    )
    weight = torch.from_numpy(weight_np).float()

    phy2log, _, logcnt = eplb.rebalance_experts(
        weight=weight,
        num_replicas=num_replicas,
        num_groups=num_groups,
        num_nodes=num_nodes,
        num_gpus=ep_size,
    )
    phy2log_np = phy2log.cpu().numpy()
    logcnt_np = logcnt.cpu().numpy()

    phy_per_gpu = num_replicas // ep_size
    global_gpu_tokens = [0.0 for _ in range(ep_size)]
    layer_variances: list[float] = []
    layers: dict[str, Any] = {}

    for li, layer_id in enumerate(layer_ids):
        gpu_tokens = [0.0 for _ in range(ep_size)]
        for phy in range(num_replicas):
            gpu_id = phy // phy_per_gpu
            log_id = int(phy2log_np[li, phy])
            replica_count = int(logcnt_np[li, log_id])
            gpu_tokens[gpu_id] += float(weight_np[li, log_id] / replica_count)

        for g in range(ep_size):
            global_gpu_tokens[g] += gpu_tokens[g]

        layer_var = float(np.var(gpu_tokens))
        layer_variances.append(layer_var)
        layers[str(layer_id)] = {
            "gpu_tokens": gpu_tokens,
            "variance": layer_var,
            "logcnt": [int(x) for x in logcnt_np[li].tolist()],
        }

    return {
        "ep_size": ep_size,
        "num_replicas": num_replicas,
        "physical_experts_per_gpu": int(phy_per_gpu),
        "global_gpu_tokens": global_gpu_tokens,
        "global_variance": float(np.var(global_gpu_tokens)),
        "mean_layer_variance": float(np.mean(layer_variances)),
        "layers": layers,
    }


def plot_token_balance(
    output_dir: Path,
    ep_size: int,
    old_gpu: list[float],
    ours_gpu: list[float],
    eplb_gpu: list[float],
) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(ep_size)
    title_fontsize = 18
    axis_label_fontsize = 17
    tick_fontsize = 15
    legend_fontsize = 14

    plt.figure(figsize=(max(10, ep_size * 0.95), 6.2))
    plt.plot(
        x,
        old_gpu,
        marker="o",
        markersize=7,
        linewidth=2.3,
        color=COLOR_BEFORE,
        label="Before reorder",
    )
    plt.plot(
        x,
        eplb_gpu,
        marker="o",
        markersize=7,
        linewidth=2.3,
        color=COLOR_EPLB,
        label="EPLB",
    )
    plt.plot(
        x,
        ours_gpu,
        marker="o",
        markersize=7,
        linewidth=2.3,
        color=COLOR_OURS,
        label="Our method",
    )
    avg_tokens = float(np.mean(old_gpu))
    plt.axhline(
        avg_tokens, color="black", linestyle="--", linewidth=1.4, label="Average tokens"
    )
    y_values = old_gpu + eplb_gpu + ours_gpu + [avg_tokens]
    y_min = min(y_values)
    y_max = max(y_values)
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else max(abs(y_min) * 0.05, 1.0)
    plt.ylim(bottom=y_min - y_pad)
    plt.xlabel("GPU rank", fontsize=axis_label_fontsize)
    plt.ylabel("Token count (sum over layers)", fontsize=axis_label_fontsize)
    plt.title(f"Token balance comparison (EP={ep_size})", fontsize=title_fontsize)
    plt.xticks(x, fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(output_dir / f"token_balance_compare_ep{ep_size}.png", dpi=220)
    plt.close()


def plot_memory_compare_all_eps(
    results_by_ep: dict[int, dict[str, Any]], output_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    eps = sorted(results_by_ep.keys())
    x = np.arange(len(eps), dtype=np.float64) * 4.0
    width = 0.75
    title_fontsize = 18
    axis_label_fontsize = 15
    tick_fontsize = 13
    legend_fontsize = 12

    before_w = [
        results_by_ep[ep]["memory"]["before"]["weight_gib_per_gpu"] for ep in eps
    ]
    before_a = [
        results_by_ep[ep]["memory"]["before"]["max_activation_gib"] for ep in eps
    ]
    eplb_w = [results_by_ep[ep]["memory"]["eplb"]["weight_gib_per_gpu"] for ep in eps]
    eplb_a = [results_by_ep[ep]["memory"]["eplb"]["max_activation_gib"] for ep in eps]
    ours_w = [results_by_ep[ep]["memory"]["ours"]["weight_gib_per_gpu"] for ep in eps]
    ours_a = [results_by_ep[ep]["memory"]["ours"]["max_activation_gib"] for ep in eps]

    plt.figure(figsize=(max(12, len(eps) * 1.6), 5.4))

    before_x = x - width
    eplb_x = x
    ours_x = x + width

    # Same column bar (stacked): model weights + activation.
    plt.bar(
        before_x, before_w, width=width, color=COLOR_BEFORE, label="Before: weights"
    )
    plt.bar(
        before_x,
        before_a,
        width=width,
        bottom=before_w,
        color=COLOR_BEFORE_LIGHT,
        label="Before: activation",
    )

    plt.bar(eplb_x, eplb_w, width=width, color=COLOR_EPLB, label="EPLB: weights")
    plt.bar(
        eplb_x,
        eplb_a,
        width=width,
        bottom=eplb_w,
        color=COLOR_EPLB_LIGHT,
        label="EPLB: activation",
    )

    plt.bar(ours_x, ours_w, width=width, color=COLOR_OURS, label="Our method: weights")
    plt.bar(
        ours_x,
        ours_a,
        width=width,
        bottom=ours_w,
        color=COLOR_OURS_LIGHT,
        label="Our method: activation",
    )

    plt.xticks(x, [str(ep) for ep in eps], fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xlabel("EP size", fontsize=axis_label_fontsize)
    plt.ylabel("Max GPU memory (GiB)", fontsize=axis_label_fontsize)
    plt.title("Max GPU memory comparison across EP sizes", fontsize=title_fontsize)
    plt.legend(ncol=3, fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(output_dir / "memory_compare_all_ep_sizes.png", dpi=220)
    plt.close()


def plot_summary_curves(
    results_by_ep: dict[int, dict[str, Any]], output_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    eps = sorted(results_by_ep.keys())
    x = np.arange(len(eps))
    ep_labels = [str(ep) for ep in eps]
    title_fontsize = 18
    axis_label_fontsize = 17
    tick_fontsize = 15
    legend_fontsize = 14

    old_var = [results_by_ep[ep]["before"]["global_variance"] for ep in eps]
    ours_var = [results_by_ep[ep]["ours"]["global_variance"] for ep in eps]
    eplb_var = [results_by_ep[ep]["eplb"]["global_variance"] for ep in eps]
    positive_values = [v for v in old_var + ours_var + eplb_var if v > 0]
    min_positive = min(positive_values) if positive_values else 1e-12
    safe_floor = min_positive * 0.5
    old_var_log10 = [math.log10(v if v > 0 else safe_floor) for v in old_var]
    ours_var_log10 = [math.log10(v if v > 0 else safe_floor) for v in ours_var]
    eplb_var_log10 = [math.log10(v if v > 0 else safe_floor) for v in eplb_var]
    all_log10 = old_var_log10 + ours_var_log10 + eplb_var_log10
    y_min = min(all_log10)
    y_max = max(all_log10)
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.2
    y_axis_min = y_min - y_pad
    y_axis_max = y_max + y_pad * 0.2
    y_tick_start = int(math.floor(y_axis_min))
    y_tick_end = int(math.ceil(y_axis_max))

    plt.figure(figsize=(max(10, len(eps) * 0.95), 6.2))
    plt.plot(
        x, old_var_log10, marker="o", linewidth=2, color=COLOR_BEFORE, label="Before"
    )
    plt.plot(x, eplb_var_log10, marker="o", linewidth=2, color=COLOR_EPLB, label="EPLB")
    plt.plot(
        x, ours_var_log10, marker="o", linewidth=2, color=COLOR_OURS, label="Our method"
    )
    plt.ylim(y_axis_min, y_axis_max)
    plt.yticks(np.arange(y_tick_start, y_tick_end + 1, 1), fontsize=tick_fontsize)
    plt.xlabel("EP size", fontsize=axis_label_fontsize)
    plt.ylabel("Global token variance (log10)", fontsize=axis_label_fontsize)
    plt.title("Token variance across EP sizes (log10)", fontsize=title_fontsize)
    plt.xticks(x, ep_labels, fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(output_dir / "variance_summary_compare.png", dpi=220)
    plt.close()


def variance_breakdown(tokens: list[float]) -> dict[str, Any]:
    tokens_np = np.array(tokens, dtype=np.float64)
    mean = float(np.mean(tokens_np))
    squared_diffs = ((tokens_np - mean) ** 2).tolist()
    variance = float(np.mean(np.array(squared_diffs, dtype=np.float64)))
    return {
        "mean": mean,
        "squared_diffs": squared_diffs,
        "variance": variance,
    }


def main() -> None:
    args = parse_args()

    cfg = load_json_config(args.config)
    default_data_path = SCRIPT_DIR / "data" / "routed_experts_stats.json"
    default_model_config = Path(
        "/data/common/models/Qwen/Qwen1.5-MoE-A2.7B/config.json"
    )

    data_path = Path(cli_or_cfg(args.data_path, cfg, "data_path", default_data_path))
    model_config_path = Path(
        cli_or_cfg(args.model_config, cfg, "model_config", default_model_config)
    )
    activation_multiplier = float(
        cli_or_cfg(args.activation_multiplier, cfg, "activation_multiplier", 1.0)
    )
    moe_layer_act_m = float(
        cli_or_cfg(args.moe_layer_act_m, cfg, "moe_layer_act_m", 180.0)
    )
    moe_layer_expert_act_m = float(
        cli_or_cfg(args.moe_layer_expert_act_m, cfg, "moe_layer_expert_act_m", 66.0)
    )
    no_plot = bool(cfg.get("no_plot", False)) or bool(args.no_plot)

    layer_to_expert_tokens = load_expert_tokens(data_path)
    if not layer_to_expert_tokens:
        raise RuntimeError("Loaded empty token stats.")

    model_cfg = json.loads(model_config_path.read_text(encoding="utf-8"))
    num_moe_layers = len(layer_to_expert_tokens)

    first_layer = min(layer_to_expert_tokens.keys())
    num_experts = len(layer_to_expert_tokens[first_layer])

    ep_sizes, ep_plan = build_ep_plan(args=args, cfg=cfg, num_experts=num_experts)
    if not ep_sizes:
        raise RuntimeError("No EP sizes available to evaluate.")

    if args.output_dir is None:
        config_output_dir = cfg.get("output_dir")
        if config_output_dir:
            output_dir = Path(config_output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = (
                REPO_ROOT
                / "outputs"
                / "simulate_ep_balance"
                / "compare_with_eplb"
                / timestamp
            )
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_ep: dict[int, dict[str, Any]] = {}

    for ep in ep_sizes:
        ours_result = simulate_for_ep(
            layer_to_expert_tokens=layer_to_expert_tokens, ep_size=ep
        )
        plan = ep_plan[ep]

        num_replicas = resolve_num_replicas(
            explicit_num_replicas=plan["num_replicas"],
            num_experts=num_experts,
            ep_size=ep,
            redundant_ratio=plan["eplb_redundant_ratio"],
            min_redundant=plan["eplb_min_redundant"],
        )
        eplb_result = simulate_eplb_for_ep(
            layer_to_expert_tokens=layer_to_expert_tokens,
            ep_size=ep,
            num_replicas=num_replicas,
            num_groups=plan["num_groups"],
            num_nodes=plan["num_nodes"],
        )

        peak_old = max_per_gpu_from_layers(
            layer_map=asdict(ours_result)["layers"],
            key="old_gpu_tokens",
            ep_size=ep,
        )
        peak_ours = max_per_gpu_from_layers(
            layer_map=asdict(ours_result)["layers"],
            key="new_gpu_tokens",
            ep_size=ep,
        )
        peak_eplb = max_per_gpu_from_layers(
            layer_map=eplb_result["layers"],
            key="gpu_tokens",
            ep_size=ep,
        )
        # Conservative memory accounting requested by user:
        # EPLB token memory should not be counted lower than our method.
        ours_peak_token = float(max(peak_ours))
        eplb_peak_token = float(max(peak_eplb))
        if eplb_peak_token < ours_peak_token:
            peak_eplb = [ours_peak_token for _ in range(ep)]

        ref_peak_tokens = float(max(peak_old))

        experts_per_gpu = num_experts // ep
        mem_before = estimate_memory(
            ep_size=ep,
            num_moe_layers=num_moe_layers,
            physical_experts_per_gpu=experts_per_gpu,
            peak_tokens_per_gpu=peak_old,
            model_cfg=model_cfg,
            reference_peak_tokens=ref_peak_tokens,
            act_multiplier=activation_multiplier,
            moe_layer_act_m=moe_layer_act_m,
            moe_layer_expert_act_m=moe_layer_expert_act_m,
        )
        mem_ours = estimate_memory(
            ep_size=ep,
            num_moe_layers=num_moe_layers,
            physical_experts_per_gpu=experts_per_gpu,
            peak_tokens_per_gpu=peak_ours,
            model_cfg=model_cfg,
            reference_peak_tokens=ref_peak_tokens,
            act_multiplier=activation_multiplier,
            moe_layer_act_m=moe_layer_act_m,
            moe_layer_expert_act_m=moe_layer_expert_act_m,
        )
        mem_eplb = estimate_memory(
            ep_size=ep,
            num_moe_layers=num_moe_layers,
            physical_experts_per_gpu=eplb_result["physical_experts_per_gpu"],
            peak_tokens_per_gpu=peak_eplb,
            model_cfg=model_cfg,
            reference_peak_tokens=ref_peak_tokens,
            act_multiplier=activation_multiplier,
            moe_layer_act_m=moe_layer_act_m,
            moe_layer_expert_act_m=moe_layer_expert_act_m,
        )

        entry = {
            "ep_size": ep,
            "num_experts": num_experts,
            "num_replicas_eplb": int(num_replicas),
            "num_groups_eplb": int(plan["num_groups"]),
            "num_nodes_eplb": int(plan["num_nodes"]),
            "before": {
                "global_gpu_tokens": ours_result.global_gpu_tokens_old,
                "global_variance": ours_result.global_variance_old,
                "mean_layer_variance": ours_result.mean_layer_variance_old,
            },
            "ours": {
                "global_gpu_tokens": ours_result.global_gpu_tokens_new,
                "global_variance": ours_result.global_variance_new,
                "mean_layer_variance": ours_result.mean_layer_variance_new,
            },
            "eplb": {
                "global_gpu_tokens": eplb_result["global_gpu_tokens"],
                "global_variance": eplb_result["global_variance"],
                "mean_layer_variance": eplb_result["mean_layer_variance"],
            },
            "memory": {
                "before": mem_before,
                "ours": mem_ours,
                "eplb": mem_eplb,
            },
            "details": {
                "ours": asdict(ours_result),
                "eplb": eplb_result,
            },
        }

        results_by_ep[ep] = entry

        if not no_plot:
            plot_token_balance(
                output_dir=output_dir,
                ep_size=ep,
                old_gpu=entry["before"]["global_gpu_tokens"],
                ours_gpu=entry["ours"]["global_gpu_tokens"],
                eplb_gpu=entry["eplb"]["global_gpu_tokens"],
            )

    if not no_plot:
        plot_memory_compare_all_eps(results_by_ep, output_dir)
        plot_summary_curves(results_by_ep, output_dir)

    serializable = {str(ep): results_by_ep[ep] for ep in sorted(results_by_ep.keys())}
    (output_dir / "compare_with_eplb_results.json").write_text(
        json.dumps(serializable, indent=2),
        encoding="utf-8",
    )

    csv_lines = [
        "ep_size,num_replicas_eplb,var_before,var_ours,var_eplb,max_mem_before_gib,max_mem_ours_gib,max_mem_eplb_gib"
    ]
    for ep in sorted(results_by_ep.keys()):
        item = results_by_ep[ep]
        csv_lines.append(
            ",".join(
                [
                    str(ep),
                    str(item["num_replicas_eplb"]),
                    f"{item['before']['global_variance']:.6f}",
                    f"{item['ours']['global_variance']:.6f}",
                    f"{item['eplb']['global_variance']:.6f}",
                    f"{item['memory']['before']['max_total_gib']:.6f}",
                    f"{item['memory']['ours']['max_total_gib']:.6f}",
                    f"{item['memory']['eplb']['max_total_gib']:.6f}",
                ]
            )
        )
    (output_dir / "compare_with_eplb_summary.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )

    print(f"Saved outputs to: {output_dir}")
    for ep in sorted(results_by_ep.keys()):
        item = results_by_ep[ep]
        before_tokens = [float(x) for x in item["before"]["global_gpu_tokens"]]
        ours_tokens = [float(x) for x in item["ours"]["global_gpu_tokens"]]
        eplb_tokens = [float(x) for x in item["eplb"]["global_gpu_tokens"]]
        before_var = variance_breakdown(before_tokens)
        ours_var = variance_breakdown(ours_tokens)
        eplb_var = variance_breakdown(eplb_tokens)
        print(
            "EP={ep} | var before/ours/eplb: {vb:.2f}/{vo:.2f}/{ve:.2f} | max mem GiB before/ours/eplb: {mb:.3f}/{mo:.3f}/{me:.3f}".format(
                ep=ep,
                vb=item["before"]["global_variance"],
                vo=item["ours"]["global_variance"],
                ve=item["eplb"]["global_variance"],
                mb=item["memory"]["before"]["max_total_gib"],
                mo=item["memory"]["ours"]["max_total_gib"],
                me=item["memory"]["eplb"]["max_total_gib"],
            )
        )
        print(f"  Before sum tokens per GPU: {before_tokens}")
        print(
            "  Before variance: mean={m:.6f}, squared_diffs={sd}, variance={v:.6f}".format(
                m=before_var["mean"],
                sd=before_var["squared_diffs"],
                v=before_var["variance"],
            )
        )
        print(f"  Our method sum tokens per GPU: {ours_tokens}")
        print(
            "  Our method variance: mean={m:.6f}, squared_diffs={sd}, variance={v:.6f}".format(
                m=ours_var["mean"], sd=ours_var["squared_diffs"], v=ours_var["variance"]
            )
        )
        print(f"  EPLB sum tokens per GPU: {eplb_tokens}")
        print(
            "  EPLB variance: mean={m:.6f}, squared_diffs={sd}, variance={v:.6f}".format(
                m=eplb_var["mean"], sd=eplb_var["squared_diffs"], v=eplb_var["variance"]
            )
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

from AutoTuner.Profiler.activations import (
    ActivationProfilingConfig,
    ActivationStrategyDecider,
    TransferBandwidthModel,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_activation_strategy_decider_uses_group_coherent_offload(tmp_path: Path) -> None:
    output_dir = tmp_path / "activation_strategy"
    _write_json(
        output_dir / "activation_profile.json",
        {
            "model_name": "Qwen/Qwen3-0.6B",
            "tp_size": 2,
            "cp_size": 2,
            "seqlen": 4096,
            "max_token_len": 4096,
            "batch_size": 128,
            "micro_batch_size": 2,
            "shape": "thd",
            "system": "megatron",
            "forward_time_s": 0.0035,
            "backward_time_s": 0.0035,
            "total_time_s": 0.007,
            "activation_bytes_by_part": {
                "attn_norm": 48 * 1024 * 1024,
                "core_attn": 96 * 1024 * 1024,
                "attn_proj": 48 * 1024 * 1024,
                "mlp_norm": 16 * 1024 * 1024,
                "expert_fc1": 0,
                "moe_act": 0,
            },
            "activation_present_by_part": {
                "attn_norm": True,
                "core_attn": True,
                "attn_proj": True,
                "mlp_norm": True,
                "expert_fc1": False,
                "moe_act": False,
            },
            "rank_summaries": [],
        },
    )

    report = ActivationStrategyDecider(
        ActivationProfilingConfig(
            model_name="Qwen/Qwen3-0.6B",
            output_dir=str(output_dir),
            tp_size=2,
            cp_size=2,
            seqlen=4096,
            max_token_len=4096,
            batch_size=128,
            micro_batch_size=2,
            transfer_model=TransferBandwidthModel(bandwidth_gbps=64.0),
        )
    ).run(skip_profiling=True)

    assert report.offload.modules == ["core_attn", "attn_proj", "attn_norm", "mlp_norm"]
    assert report.recompute.kind == "none"


def test_activation_strategy_decider_recomputes_leftover_groups(tmp_path: Path) -> None:
    output_dir = tmp_path / "activation_strategy"
    _write_json(
        output_dir / "activation_profile.json",
        {
            "model_name": "Qwen/Qwen3-0.6B",
            "tp_size": 2,
            "cp_size": 2,
            "seqlen": 4096,
            "max_token_len": 4096,
            "batch_size": 128,
            "micro_batch_size": 2,
            "shape": "thd",
            "system": "megatron",
            "forward_time_s": 0.0025,
            "backward_time_s": 0.0025,
            "total_time_s": 0.005,
            "activation_bytes_by_part": {
                "attn_norm": 64 * 1024 * 1024,
                "core_attn": 96 * 1024 * 1024,
                "attn_proj": 48 * 1024 * 1024,
                "mlp_norm": 32 * 1024 * 1024,
                "expert_fc1": 0,
                "moe_act": 0,
            },
            "activation_present_by_part": {
                "attn_norm": True,
                "core_attn": True,
                "attn_proj": True,
                "mlp_norm": True,
                "expert_fc1": False,
                "moe_act": False,
            },
            "rank_summaries": [],
        },
    )

    report = ActivationStrategyDecider(
        ActivationProfilingConfig(
            model_name="Qwen/Qwen3-0.6B",
            output_dir=str(output_dir),
            tp_size=2,
            cp_size=2,
            seqlen=4096,
            max_token_len=4096,
            batch_size=128,
            micro_batch_size=2,
            transfer_model=TransferBandwidthModel(bandwidth_gbps=64.0),
        )
    ).run(skip_profiling=True)

    assert report.offload.modules == ["core_attn", "attn_proj"]
    assert report.recompute.kind == "selective"
    assert report.recompute.modules == ["layernorm"]

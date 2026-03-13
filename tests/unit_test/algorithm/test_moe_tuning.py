from __future__ import annotations

import json
from pathlib import Path

from AutoTuner.algorithm.moe_tuning import MoETuningAlgorithm, MoETuningConfig


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_moe_tuning_runs_from_existing_reports(tmp_path: Path) -> None:
    tp_dir = tmp_path / "tp_overlap"
    cp_dir = tmp_path / "cp_overlap"
    ep_dir = tmp_path / "expert_parallel"
    output_dir = tmp_path / "moe_tuning"

    _write_json(
        tp_dir / "tuning_report.json",
        {
            "tuner_config": {"max_token_len": 4096},
            "tp_scaling": {"optimal_tp_size": 2},
            "analyses": [
                {
                    "config_id": "tp2_fc1_wgrad_bulk_sm4",
                    "tp_size": 2,
                    "operator": "fc1",
                    "phase": "wgrad",
                    "total_overlap_ratio": 0.82,
                    "operator_e2e_time_us": 130.0,
                },
                {
                    "config_id": "tp4_qkv_fprop_ring_agg1",
                    "tp_size": 4,
                    "operator": "qkv",
                    "phase": "fprop",
                    "total_overlap_ratio": 0.95,
                    "operator_e2e_time_us": 160.0,
                },
            ],
        },
    )
    _write_json(
        cp_dir / "cp_overlap_report.json",
        {
            "cp_size": 2,
            "results": [
                {
                    "case_id": "seq2048_tok2048_bs128_mbs2_thd",
                    "case": {"seqlen": 2048, "max_token_len": 2048},
                    "success": True,
                    "analysis": {
                        "overlap_ratio": 0.90,
                        "compute_time_us": 50.0,
                        "comm_time_us": 20.0,
                        "overlapped_comm_time_us": 18.0,
                    },
                }
            ],
        },
    )
    _write_json(
        ep_dir / "expert_parallel_profile.json",
        {
            "model_name": "Qwen/Qwen3-30B-A3B",
            "tp_size": 2,
            "cp_size": 3,
            "seqlen": 6000,
            "max_token_len": 6144,
            "nprocs_per_node": 16,
            "batch_size": 128,
            "micro_batch_size": 2,
            "shape": "thd",
            "system": "megatron",
            "hidden_size": 4096,
            "num_moe_experts": 8,
            "moe_router_topk": 2,
            "dtype_bytes": 2,
            "profiled_ep_size": 1,
            "attention_forward_time_s": 0.003,
            "attention_backward_time_s": 0.006,
            "attention_wgrad_time_s": 0.0005,
            "mlp_forward_time_s": 0.004,
            "mlp_backward_time_s": 0.006,
            "mlp_wgrad_time_s": 0.003,
            "notes": [],
        },
    )

    config = MoETuningConfig(
        model_name="Qwen/Qwen3-30B-A3B",
        prompt_len=4096,
        response_len=1904,
        nprocs_per_node=16,
        output_dir=str(output_dir),
        tp_output_dir=str(tp_dir),
        cp_output_dir=str(cp_dir),
        ep_output_dir=str(ep_dir),
        skip_tp_profiling=True,
        skip_cp_profiling=True,
        skip_ep_profiling=True,
    )

    report = MoETuningAlgorithm(config).run()

    assert report.step1.selected_tp_size == 2
    assert report.step2.cp_size == 3
    assert report.step2.max_token_len == 6144
    assert report.step3.selected_ep_size == 2
    assert report.step3.selected_local_experts == 4
    assert report.step3.selected_ep_fully_overlapped is True
    assert (output_dir / "moe_tuning_report.json").exists()
    assert (output_dir / "summary.txt").exists()
    assert (ep_dir / "expert_parallel_strategy_report.json").exists()

from __future__ import annotations

import json
from pathlib import Path

import pytest

from AutoTuner.algorithm.dense_tuning import (
    DenseTuningAlgorithm,
    DenseTuningConfig,
    DenseTuningStep1Result,
    Step1TaskSummary,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_tp_cp_max_token_len_decider_runs_from_existing_reports(
    tmp_path: Path,
) -> None:
    tp_dir = tmp_path / "tp_overlap"
    cp_dir = tmp_path / "cp_overlap"
    activation_dir = tmp_path / "activation_strategy"
    runtime_dir = tmp_path / "runtime_baseline"
    output_dir = tmp_path / "dense_tuning"

    _write_json(
        tp_dir / "tuning_report.json",
        {
            "tuner_config": {"max_token_len": 4096},
            "tp_scaling": {"optimal_tp_size": 2},
            "analyses": [
                {
                    "config_id": "tp2_fc1_dgrad_bulk_sm4",
                    "tp_size": 2,
                    "operator": "fc1",
                    "phase": "dgrad",
                    "total_overlap_ratio": 0.80,
                    "forward_overlap_ratio": 0.0,
                    "backward_overlap_ratio": 0.80,
                    "operator_e2e_time_us": 120.0,
                },
                {
                    "config_id": "tp4_qkv_fprop_ring_agg1",
                    "tp_size": 4,
                    "operator": "qkv",
                    "phase": "fprop",
                    "total_overlap_ratio": 0.95,
                    "forward_overlap_ratio": 0.95,
                    "backward_overlap_ratio": 0.0,
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
                },
                {
                    "case_id": "seq1024_tok1024_bs128_mbs2_thd",
                    "case": {"seqlen": 1024, "max_token_len": 1024},
                    "success": True,
                    "analysis": {
                        "overlap_ratio": 0.75,
                        "compute_time_us": 30.0,
                        "comm_time_us": 10.0,
                        "overlapped_comm_time_us": 7.5,
                    },
                },
            ],
        },
    )
    _write_json(
        activation_dir / "activation_profile.json",
        {
            "model_name": "Qwen/Qwen3-0.6B",
            "tp_size": 2,
            "cp_size": 3,
            "seqlen": 6000,
            "max_token_len": 6144,
            "batch_size": 128,
            "micro_batch_size": 2,
            "shape": "thd",
            "system": "megatron",
            "forward_time_s": 0.004,
            "backward_time_s": 0.005,
            "total_time_s": 0.009,
            "activation_bytes_by_part": {
                "attn_norm": 64 * 1024 * 1024,
                "core_attn": 160 * 1024 * 1024,
                "attn_proj": 64 * 1024 * 1024,
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
    _write_json(
        runtime_dir / "runtime_baseline.json",
        {
            "model_name": "Qwen/Qwen3-0.6B",
            "original_total_layers": 28,
            "simulation": {
                "stage_param_stats": [
                    {
                        "pp_rank": 0,
                        "runtime_layer_count": 1,
                        "runtime_total_param_bytes": 1_200,
                        "runtime_decoder_param_bytes": 1_000,
                    }
                ]
            },
            "test_cases": [
                {
                    "summary": {
                        "memory_by_rank": [
                            {
                                "rank": 0,
                                "real_detected_bytes": 1_500,
                                "total_device_bytes": 5_000,
                            },
                            {
                                "rank": 1,
                                "real_detected_bytes": 1_450,
                                "total_device_bytes": 5_000,
                            },
                        ],
                        "simulation": {
                            "stage_param_stats": [
                                {
                                    "pp_rank": 0,
                                    "runtime_layer_count": 1,
                                    "runtime_total_param_bytes": 1_200,
                                    "runtime_decoder_param_bytes": 1_000,
                                }
                            ]
                        },
                    }
                }
            ],
        },
    )

    config = DenseTuningConfig(
        model_name="Qwen/Qwen3-0.6B",
        prompt_len=4096,
        response_len=1904,
        nprocs_per_node=8,
        output_dir=str(output_dir),
        tp_output_dir=str(tp_dir),
        cp_output_dir=str(cp_dir),
        activation_output_dir=str(activation_dir),
        runtime_output_dir=str(runtime_dir),
        skip_tp_profiling=True,
        skip_cp_profiling=True,
        skip_activation_profiling=True,
        skip_runtime_profiling=True,
    )

    report = DenseTuningAlgorithm(config).run()

    assert report.step1.selected_tp_size == 2
    assert report.step1.best_tp_task.task_id == "tp4_qkv_fprop_ring_agg1"
    assert report.step1.best_cp_task.task_id == "seq2048_tok2048_bs128_mbs2_thd"
    assert report.step2.seqlen == 6000
    assert report.step2.cp_size == 3
    assert report.step2.max_token_len == 6144
    assert report.step3.offload_modules == ["core_attn", "attn_proj"]
    assert report.step3.recompute_kind == "selective"
    assert report.step3.recompute_modules == ["layernorm"]
    assert report.step4.pipeline_model_parallel_size == 10
    assert report.step4.layer_distribution == [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
    assert (output_dir / "dense_tuning_report.json").exists()
    assert (output_dir / "summary.txt").exists()


def test_decide_step4_from_runtime_baseline_report_uses_post_process_capacity() -> None:
    report = DenseTuningAlgorithm.decide_step4_from_runtime_baseline_report(
        runtime_baseline_report={
            "original_total_layers": 26,
            "simulation": {
                "stage_param_stats": [
                    {
                        "pp_rank": 0,
                        "runtime_layer_count": 1,
                        "runtime_total_param_bytes": 120,
                        "runtime_decoder_param_bytes": 100,
                    }
                ]
            },
            "test_cases": [
                {
                    "summary": {
                        "memory_by_rank": [
                            {
                                "rank": 0,
                                "real_detected_bytes": 150,
                                "total_device_bytes": 500,
                            }
                        ]
                    }
                }
            ],
        },
        runtime_baseline_report_path="runtime_baseline.json",
        memory_target_fraction=0.9,
    )

    assert report.unit_decoder_param_bytes == 100
    assert report.unit_activation_bytes == 30
    assert report.regular_stage_layer_capacity == 3
    assert report.post_process_stage_layer_capacity == 3
    assert report.pipeline_model_parallel_size == 9
    assert report.layer_distribution == [3, 3, 3, 3, 3, 3, 3, 3, 2]


def test_select_best_tp_task_breaks_ties_by_faster_e2e() -> None:
    task = DenseTuningAlgorithm.select_best_tp_task(
        {
            "tuner_config": {"max_token_len": 8192},
            "analyses": [
                {
                    "config_id": "slow",
                    "tp_size": 2,
                    "operator": "fc1",
                    "phase": "dgrad",
                    "total_overlap_ratio": 0.90,
                    "operator_e2e_time_us": 120.0,
                },
                {
                    "config_id": "fast",
                    "tp_size": 4,
                    "operator": "qkv",
                    "phase": "fprop",
                    "total_overlap_ratio": 0.90,
                    "operator_e2e_time_us": 80.0,
                },
            ],
        }
    )

    assert task.task_id == "fast"
    assert task.max_token_len == 8192


def test_decide_step2_uses_smallest_valid_custom_cp_candidate() -> None:
    step1_result = DenseTuningStep1Result(
        tp_report_path="tp.json",
        cp_report_path="cp.json",
        selected_tp_size=2,
        best_tp_task=Step1TaskSummary(
            source="tp", task_id="tp", overlap_ratio=0.8, max_token_len=4096
        ),
        best_cp_task=Step1TaskSummary(
            source="cp", task_id="cp", overlap_ratio=0.9, max_token_len=2048
        ),
        best_overall_task=Step1TaskSummary(
            source="cp", task_id="cp", overlap_ratio=0.9, max_token_len=2048
        ),
    )

    result = DenseTuningAlgorithm.decide_step2(
        step1_result=step1_result,
        seqlen=5000,
        nprocs_per_node=16,
        cp_candidates=[1, 4, 8],
    )

    assert result.cp_size == 4
    assert result.max_cp_allowed == 7
    assert result.max_token_len == 8192


def test_decide_step2_errors_when_tp_leaves_no_cp_budget() -> None:
    step1_result = DenseTuningStep1Result(
        tp_report_path="tp.json",
        cp_report_path="cp.json",
        selected_tp_size=8,
        best_tp_task=Step1TaskSummary(
            source="tp", task_id="tp", overlap_ratio=0.8, max_token_len=4096
        ),
        best_cp_task=Step1TaskSummary(
            source="cp", task_id="cp", overlap_ratio=0.9, max_token_len=2048
        ),
        best_overall_task=Step1TaskSummary(
            source="cp", task_id="cp", overlap_ratio=0.9, max_token_len=2048
        ),
    )

    with pytest.raises(ValueError, match="No valid CP value"):
        DenseTuningAlgorithm.decide_step2(
            step1_result=step1_result,
            seqlen=4096,
            nprocs_per_node=8,
        )

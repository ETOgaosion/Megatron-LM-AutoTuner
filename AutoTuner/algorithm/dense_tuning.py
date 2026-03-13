#!/usr/bin/env python3
"""Dense-model tuning algorithm for TP/CP/max_token_len/PP decisions."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

from .common import (
    Step1TaskSummary,
    Step12TuningAlgorithmMixin,
    TuningStep1Result,
    TuningStep2Result,
    default_output_dir,
)


def _default_output_dir() -> str:
    return default_output_dir(
        "dense_tuning", datetime.now().strftime("%Y%m%d_%H%M%S")
    )


DenseTuningStep1Result = TuningStep1Result
DenseTuningStep2Result = TuningStep2Result


@dataclass(frozen=True)
class DenseTuningStep3Result:
    """Step 3 activation strategy derived from step 2."""

    activation_profile_path: str
    activation_strategy_path: str
    compute_time_s: float
    activation_bytes_by_part: dict[str, int]
    offload_modules: list[str]
    total_offload_time_s: float
    recompute_kind: str
    recompute_modules: list[str]
    recompute_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DenseTuningStep4Result:
    """Step 4 pipeline-parallel decision derived from runtime baseline."""

    runtime_baseline_report_path: str
    total_num_layers: int
    pipeline_model_parallel_size: int
    layers_per_regular_stage: int
    layers_in_post_process_stage: int
    layer_distribution: list[int]
    regular_stage_layer_capacity: int
    post_process_stage_layer_capacity: int
    unit_decoder_param_bytes: int
    post_process_overhead_param_bytes: int
    unit_activation_bytes: int
    unit_total_bytes: int
    measured_peak_memory_bytes: int
    usable_memory_bytes: int
    memory_target_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DenseTuningReport:
    """Combined step 1 through step 4 output."""

    model_name: str
    prompt_len: int
    response_len: int
    nprocs_per_node: int
    step1: DenseTuningStep1Result
    step2: DenseTuningStep2Result
    step3: DenseTuningStep3Result
    step4: DenseTuningStep4Result
    assumptions: list[str]

    @property
    def seqlen(self) -> int:
        return self.prompt_len + self.response_len

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "prompt_len": self.prompt_len,
            "response_len": self.response_len,
            "seqlen": self.seqlen,
            "nprocs_per_node": self.nprocs_per_node,
            "step1": self.step1.to_dict(),
            "step2": self.step2.to_dict(),
            "step3": self.step3.to_dict(),
            "step4": self.step4.to_dict(),
            "assumptions": self.assumptions,
        }


@dataclass(frozen=True)
class DenseTuningConfig:
    """Configuration for the dense tuning algorithm."""

    model_name: str
    prompt_len: int
    response_len: int
    nprocs_per_node: int
    output_dir: str = field(default_factory=_default_output_dir)
    tp_output_dir: Optional[str] = None
    cp_output_dir: Optional[str] = None
    max_tp_size: int = 8
    tp_max_token_len: int = 8192
    tp_operators: list[str] = field(
        default_factory=lambda: ["fc1", "fc2", "qkv", "proj"]
    )
    tp_min_num_sm: int = 1
    tp_max_num_sm: int = 16
    skip_tp_profiling: bool = False
    cp_seqlen_start: Optional[int] = None
    cp_seqlen_end: Optional[int] = None
    cp_max_token_len: Optional[int] = None
    cp_seqlen_step: int = 1024
    cp_size_for_profiling: int = 2
    cp_batch_size: int = 128
    cp_micro_batch_size: int = 2
    cp_shape: str = "thd"
    cp_system: str = "megatron"
    skip_cp_profiling: bool = False
    cp_candidates: Optional[list[int]] = None
    activation_output_dir: Optional[str] = None
    activation_batch_size: int = 128
    activation_micro_batch_size: int = 2
    activation_shape: str = "thd"
    activation_system: str = "megatron"
    activation_warmup_iters: int = 2
    activation_profile_iters: int = 3
    activation_bandwidth_gbps: float = 64.0
    activation_bandwidth_overhead_ms: float = 0.0
    activation_transfer_round_trips: float = 2.0
    skip_activation_profiling: bool = False
    runtime_output_dir: Optional[str] = None
    runtime_batch_size: int = 128
    runtime_micro_batch_size: int = 2
    runtime_shape: str = "thd"
    runtime_system: str = "megatron"
    runtime_max_iterations: int = 3
    runtime_warmup_iterations: int = 1
    runtime_use_fused_kernels: bool = True
    runtime_no_ddp: bool = True
    runtime_memory_target_fraction: float = 0.9
    runtime_full_recompute_method: str = "uniform"
    skip_runtime_profiling: bool = False

    def __post_init__(self) -> None:
        if self.prompt_len < 0 or self.response_len < 0:
            raise ValueError("prompt_len and response_len must be non-negative.")
        if self.nprocs_per_node <= 0:
            raise ValueError("nprocs_per_node must be positive.")
        if self.max_tp_size <= 0:
            raise ValueError("max_tp_size must be positive.")
        if self.tp_max_token_len <= 0:
            raise ValueError("tp_max_token_len must be positive.")
        if self.cp_seqlen_step <= 0:
            raise ValueError("cp_seqlen_step must be positive.")
        if self.cp_size_for_profiling <= 0:
            raise ValueError("cp_size_for_profiling must be positive.")
        if self.cp_batch_size <= 0 or self.cp_micro_batch_size <= 0:
            raise ValueError("cp batch sizes must be positive.")
        if self.cp_shape not in {"thd", "bshd"}:
            raise ValueError("cp_shape must be either 'thd' or 'bshd'.")
        if self.activation_batch_size <= 0 or self.activation_micro_batch_size <= 0:
            raise ValueError("activation batch sizes must be positive.")
        if self.activation_shape not in {"thd", "bshd"}:
            raise ValueError("activation_shape must be either 'thd' or 'bshd'.")
        if self.runtime_batch_size <= 0 or self.runtime_micro_batch_size <= 0:
            raise ValueError("runtime batch sizes must be positive.")
        if self.runtime_shape not in {"thd", "bshd"}:
            raise ValueError("runtime_shape must be either 'thd' or 'bshd'.")
        if self.runtime_max_iterations <= 0:
            raise ValueError("runtime_max_iterations must be positive.")
        if self.runtime_warmup_iterations < 0:
            raise ValueError("runtime_warmup_iterations must be non-negative.")
        if not (0 < self.runtime_memory_target_fraction <= 1):
            raise ValueError("runtime_memory_target_fraction must be in (0, 1].")
        if self.runtime_full_recompute_method not in {"uniform", "block"}:
            raise ValueError(
                "runtime_full_recompute_method must be either 'uniform' or 'block'."
            )
        if self.activation_warmup_iters < 0 or self.activation_profile_iters <= 0:
            raise ValueError(
                "activation_warmup_iters must be >= 0 and "
                "activation_profile_iters must be > 0."
            )
        if self.activation_bandwidth_gbps <= 0:
            raise ValueError("activation_bandwidth_gbps must be positive.")
        if self.activation_bandwidth_overhead_ms < 0:
            raise ValueError("activation_bandwidth_overhead_ms must be non-negative.")
        if self.activation_transfer_round_trips <= 0:
            raise ValueError("activation_transfer_round_trips must be positive.")
        if self.cp_seqlen_start is not None and self.cp_seqlen_start <= 0:
            raise ValueError("cp_seqlen_start must be positive when provided.")
        if self.cp_seqlen_end is not None and self.cp_seqlen_end <= 0:
            raise ValueError("cp_seqlen_end must be positive when provided.")
        if (
            self.cp_seqlen_start is not None
            and self.cp_seqlen_end is not None
            and self.cp_seqlen_start > self.cp_seqlen_end
        ):
            raise ValueError("cp_seqlen_start must be <= cp_seqlen_end.")
        if self.cp_candidates is not None:
            if not self.cp_candidates:
                raise ValueError("cp_candidates must not be empty.")
            if any(candidate <= 0 for candidate in self.cp_candidates):
                raise ValueError("cp_candidates must contain only positive integers.")

    @property
    def seqlen(self) -> int:
        return self.prompt_len + self.response_len

    @property
    def resolved_tp_output_dir(self) -> str:
        return self.tp_output_dir or os.path.join(self.output_dir, "tp_overlap")

    @property
    def resolved_cp_output_dir(self) -> str:
        return self.cp_output_dir or os.path.join(self.output_dir, "cp_overlap")

    @property
    def resolved_cp_max_token_len(self) -> int:
        return self.cp_max_token_len or self.tp_max_token_len

    @property
    def resolved_cp_seqlen_start(self) -> int:
        return self.cp_seqlen_start or self.resolved_cp_max_token_len

    @property
    def resolved_cp_seqlen_end(self) -> int:
        return self.cp_seqlen_end or self.resolved_cp_max_token_len

    @property
    def resolved_activation_output_dir(self) -> str:
        return self.activation_output_dir or os.path.join(
            self.output_dir, "activation_strategy"
        )

    @property
    def resolved_runtime_output_dir(self) -> str:
        return self.runtime_output_dir or os.path.join(
            self.output_dir, "runtime_baseline"
        )


class DenseTuningAlgorithm(Step12TuningAlgorithmMixin):
    """Implements dense-model tuning steps 1 through 4."""

    def __init__(self, config: DenseTuningConfig):
        self.config = config

    def run(self) -> DenseTuningReport:
        os.makedirs(self.config.output_dir, exist_ok=True)

        step1_result = self.run_step1()
        step2_result = self.run_step2(step1_result)
        step3_result = self.run_step3(step2_result)
        step4_result = self.run_step4(step2_result, step3_result)
        report = DenseTuningReport(
            model_name=self.config.model_name,
            prompt_len=self.config.prompt_len,
            response_len=self.config.response_len,
            nprocs_per_node=self.config.nprocs_per_node,
            step1=step1_result,
            step2=step2_result,
            step3=step3_result,
            step4=step4_result,
            assumptions=[
                "Step 2 picks the smallest valid integer CP that satisfies "
                "seqlen <= CP * step1_max_token_len.",
                "Default CP candidates are all positive integers because the "
                "requirement only constrains CP by CP < nprocs_per_node / TP.",
                "Step 3 compares estimated activation offload round-trip time "
                "against the measured TransformerLayer forward+backward time.",
                "Step 4 treats runtime decoder parameters and runtime activation peak "
                "as linear per-layer unit costs.",
                "Step 4 assigns the runtime baseline non-decoder parameter overhead "
                "to the post-process PP stage when sizing the irregular final stage.",
                "Step 4 uses runtime_memory_target_fraction to leave memory headroom "
                "when translating the runtime probe into a PP recommendation.",
            ],
        )
        self._save_report(report)
        return report

    def run_step3(
        self, step2_result: DenseTuningStep2Result
    ) -> DenseTuningStep3Result:
        from AutoTuner.Profiler.activations import (
            ActivationProfilingConfig,
            ActivationStrategyDecider,
            TransferBandwidthModel,
        )

        strategy = ActivationStrategyDecider(
            ActivationProfilingConfig(
                model_name=self.config.model_name,
                output_dir=self.config.resolved_activation_output_dir,
                tp_size=step2_result.tp_size,
                cp_size=step2_result.cp_size,
                seqlen=step2_result.seqlen,
                max_token_len=step2_result.max_token_len,
                batch_size=self.config.activation_batch_size,
                micro_batch_size=self.config.activation_micro_batch_size,
                shape=self.config.activation_shape,
                system=self.config.activation_system,
                warmup_iters=self.config.activation_warmup_iters,
                profile_iters=self.config.activation_profile_iters,
                transfer_model=TransferBandwidthModel(
                    bandwidth_gbps=self.config.activation_bandwidth_gbps,
                    overhead_seconds=self.config.activation_bandwidth_overhead_ms
                    / 1000.0,
                    round_trip_factor=self.config.activation_transfer_round_trips,
                ),
            )
        ).run(skip_profiling=self.config.skip_activation_profiling)

        return DenseTuningStep3Result(
            activation_profile_path=strategy.profile_path,
            activation_strategy_path=os.path.join(
                self.config.resolved_activation_output_dir,
                "activation_strategy_report.json",
            ),
            compute_time_s=strategy.profile.total_time_s,
            activation_bytes_by_part=strategy.profile.activation_bytes_by_part,
            offload_modules=list(strategy.offload.modules),
            total_offload_time_s=strategy.offload.total_time_s,
            recompute_kind=strategy.recompute.kind,
            recompute_modules=list(strategy.recompute.modules),
            recompute_reason=strategy.recompute.reason,
        )

    def run_step4(
        self,
        step2_result: DenseTuningStep2Result,
        step3_result: DenseTuningStep3Result,
    ) -> DenseTuningStep4Result:
        runtime_baseline_report_path = self._ensure_runtime_baseline_report(
            step2_result=step2_result,
            step3_result=step3_result,
        )
        runtime_baseline_report = self._load_json(runtime_baseline_report_path)
        return self.decide_step4_from_runtime_baseline_report(
            runtime_baseline_report=runtime_baseline_report,
            runtime_baseline_report_path=runtime_baseline_report_path,
            memory_target_fraction=self.config.runtime_memory_target_fraction,
        )

    def _ensure_runtime_baseline_report(
        self,
        step2_result: DenseTuningStep2Result,
        step3_result: DenseTuningStep3Result,
    ) -> str:
        report_path = self._find_runtime_baseline_report(
            self.config.resolved_runtime_output_dir
        )
        if self.config.skip_runtime_profiling:
            if report_path is None:
                raise FileNotFoundError(
                    "Runtime baseline report not found under "
                    f"{self.config.resolved_runtime_output_dir}. "
                    "Pass an existing --runtime-output-dir or disable "
                    "--skip-runtime-profiling."
                )
            return report_path

        input_dir = os.path.join(self.config.resolved_runtime_output_dir, "inputs")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(self.config.resolved_runtime_output_dir, exist_ok=True)

        test_cases_payload = {
            "cases": [
                {
                    "batch_size": self.config.runtime_batch_size,
                    "micro_batch_size": self.config.runtime_micro_batch_size,
                    "seqlen": step2_result.seqlen,
                    "max_token_len": step2_result.max_token_len,
                    "shape": self.config.runtime_shape,
                    "system": self.config.runtime_system,
                }
            ]
        }
        with open(
            os.path.join(input_dir, "dense_tuning_step4_test_cases.json"), "w"
        ) as handle:
            json.dump(test_cases_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

        with open(
            os.path.join(input_dir, "override_model_config.json"), "w"
        ) as handle:
            json.dump({}, handle, indent=2, sort_keys=True)
            handle.write("\n")

        with open(
            os.path.join(input_dir, "override_tf_config.json"), "w"
        ) as handle:
            json.dump(
                self._build_runtime_override_tf_config(step3_result),
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        with open(
            os.path.join(input_dir, "ddp_simulate_config.json"), "w"
        ) as handle:
            json.dump(
                {
                    "dp_allreduce_bandwidth_gbps": 50,
                    "dp_allreduce_latency_us": 30,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        cmd = [
            "torchrun",
            f"--nproc_per_node={step2_result.tp_size * step2_result.cp_size}",
            "-m",
            "AutoTuner.runtime.baseline.main",
            "--model-name",
            self.config.model_name,
            "--test-cases-dir",
            input_dir,
            "--config-dir",
            input_dir,
            "--test-cases-file",
            "dense_tuning_step4_test_cases.json",
            "--override-model-config-file",
            "override_model_config.json",
            "--override-tf-config-file",
            "override_tf_config.json",
            "--ddp-simulate-config-file",
            "ddp_simulate_config.json",
            "--tensor-model-parallel-size",
            str(step2_result.tp_size),
            "--context-parallel-size",
            str(step2_result.cp_size),
            "--pipeline-model-parallel-size",
            "1",
            "--max-iterations",
            str(self.config.runtime_max_iterations),
            "--warmup-iterations",
            str(self.config.runtime_warmup_iterations),
            "--use-fused-kernels",
            "true" if self.config.runtime_use_fused_kernels else "false",
            "--output-dir",
            self.config.resolved_runtime_output_dir,
        ]
        if self.config.runtime_no_ddp:
            cmd.append("--no-ddp")

        env = os.environ.copy()
        env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        env.setdefault("NVTE_FLASH_ATTN", "1")
        env.setdefault("NVTE_FUSED_ATTN", "0")
        env.setdefault("NVTE_NVTX_ENABLED", "1")
        try:
            subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - env dependent
            error = exc.stderr or exc.stdout or str(exc)
            raise RuntimeError(f"Runtime baseline profiling failed: {error}") from exc

        report_path = self._find_runtime_baseline_report(
            self.config.resolved_runtime_output_dir
        )
        if report_path is None:  # pragma: no cover - env dependent
            raise FileNotFoundError(
                "Runtime baseline finished without producing runtime_baseline.json under "
                f"{self.config.resolved_runtime_output_dir}."
            )
        return report_path

    def _build_runtime_override_tf_config(
        self, step3_result: DenseTuningStep3Result
    ) -> dict[str, Any]:
        override_tf_config: dict[str, Any] = {"tp_comm_overlap": False}
        if step3_result.recompute_kind == "full":
            override_tf_config.update(
                {
                    "recompute_granularity": "full",
                    "recompute_method": self.config.runtime_full_recompute_method,
                    "recompute_num_layers": 1,
                    "recompute_modules": None,
                }
            )
        elif step3_result.recompute_kind == "selective":
            override_tf_config.update(
                {
                    "recompute_granularity": "selective",
                    "recompute_method": None,
                    "recompute_num_layers": None,
                    "recompute_modules": list(step3_result.recompute_modules),
                }
            )
        else:
            override_tf_config.update(
                {
                    "recompute_granularity": None,
                    "recompute_method": None,
                    "recompute_num_layers": None,
                    "recompute_modules": None,
                }
            )
        return override_tf_config

    @classmethod
    def decide_step4_from_runtime_baseline_report(
        cls,
        runtime_baseline_report: Mapping[str, Any],
        runtime_baseline_report_path: str,
        memory_target_fraction: float = 0.9,
    ) -> DenseTuningStep4Result:
        if not (0 < memory_target_fraction <= 1):
            raise ValueError("memory_target_fraction must be in (0, 1].")

        total_num_layers = int(runtime_baseline_report.get("original_total_layers", 0))
        if total_num_layers <= 0:
            raise ValueError(
                "Runtime baseline report does not contain a positive original_total_layers."
            )

        probe_summary = cls._extract_runtime_probe_summary(runtime_baseline_report)
        simulation = (runtime_baseline_report.get("simulation") or {}).copy()
        if "stage_param_stats" not in simulation:
            simulation["stage_param_stats"] = (probe_summary.get("simulation") or {}).get(
                "stage_param_stats"
            )
        stage_param_stats = simulation.get("stage_param_stats")
        if not isinstance(stage_param_stats, list) or len(stage_param_stats) != 1:
            raise ValueError(
                "Step 4 expects a PP=1 runtime baseline report with exactly one stage_param_stats entry."
            )

        stage_stats = stage_param_stats[0]
        runtime_layer_count = int(stage_stats.get("runtime_layer_count", 0))
        runtime_total_param_bytes = int(stage_stats.get("runtime_total_param_bytes", 0))
        runtime_decoder_param_bytes = int(
            stage_stats.get("runtime_decoder_param_bytes", 0)
        )
        if runtime_layer_count <= 0:
            raise ValueError("runtime_layer_count must be positive in stage_param_stats.")
        if runtime_decoder_param_bytes <= 0 or runtime_total_param_bytes <= 0:
            raise ValueError("Runtime stage parameter bytes must be positive.")
        if runtime_total_param_bytes < runtime_decoder_param_bytes:
            raise ValueError(
                "runtime_total_param_bytes must be >= runtime_decoder_param_bytes."
            )

        memory_by_rank = probe_summary.get("memory_by_rank")
        if not isinstance(memory_by_rank, list) or not memory_by_rank:
            raise ValueError(
                "Runtime baseline probe summary does not contain memory_by_rank."
            )

        measured_peak_memory_bytes = max(
            int(rank_stats.get("real_detected_bytes", 0)) for rank_stats in memory_by_rank
        )
        total_device_bytes = min(
            int(rank_stats.get("total_device_bytes", 0)) for rank_stats in memory_by_rank
        )
        if measured_peak_memory_bytes <= 0 or total_device_bytes <= 0:
            raise ValueError("Runtime baseline memory summary is incomplete.")

        unit_decoder_param_bytes = math.ceil(
            runtime_decoder_param_bytes / float(runtime_layer_count)
        )
        post_process_overhead_param_bytes = max(
            0, runtime_total_param_bytes - runtime_decoder_param_bytes
        )
        unit_activation_bytes = max(
            0,
            math.ceil(
                max(0, measured_peak_memory_bytes - runtime_total_param_bytes)
                / float(runtime_layer_count)
            ),
        )
        unit_total_bytes = unit_decoder_param_bytes + unit_activation_bytes
        if unit_total_bytes <= 0:
            raise ValueError("Derived unit_total_bytes must be positive.")

        usable_memory_bytes = math.floor(total_device_bytes * memory_target_fraction)
        regular_stage_layer_capacity = usable_memory_bytes // unit_total_bytes
        post_process_stage_layer_capacity = max(
            0,
            (usable_memory_bytes - post_process_overhead_param_bytes) // unit_total_bytes,
        )
        if regular_stage_layer_capacity < 1:
            raise ValueError(
                "Even one decoder layer does not fit within the usable memory budget."
            )
        if post_process_stage_layer_capacity < 1:
            raise ValueError(
                "The post-process stage cannot hold one decoder layer within the usable memory budget."
            )

        chosen_pp_size = None
        layers_per_regular_stage = None
        layers_in_post_process_stage = None
        layer_distribution = None
        for pp_size in range(1, total_num_layers + 1):
            candidate_regular_layers = math.ceil(total_num_layers / float(pp_size))
            candidate_post_layers = total_num_layers - (
                (pp_size - 1) * candidate_regular_layers
            )
            if candidate_post_layers <= 0:
                continue
            if candidate_regular_layers > regular_stage_layer_capacity:
                continue
            if candidate_post_layers > post_process_stage_layer_capacity:
                continue
            chosen_pp_size = pp_size
            layers_per_regular_stage = candidate_regular_layers
            layers_in_post_process_stage = candidate_post_layers
            layer_distribution = [candidate_regular_layers] * max(0, pp_size - 1)
            layer_distribution.append(candidate_post_layers)
            break

        if (
            chosen_pp_size is None
            or layers_per_regular_stage is None
            or layers_in_post_process_stage is None
            or layer_distribution is None
        ):
            raise ValueError(
                "Could not find a PP size that fits the runtime-derived per-layer and post-process memory limits."
            )

        return DenseTuningStep4Result(
            runtime_baseline_report_path=runtime_baseline_report_path,
            total_num_layers=total_num_layers,
            pipeline_model_parallel_size=chosen_pp_size,
            layers_per_regular_stage=layers_per_regular_stage,
            layers_in_post_process_stage=layers_in_post_process_stage,
            layer_distribution=layer_distribution,
            regular_stage_layer_capacity=regular_stage_layer_capacity,
            post_process_stage_layer_capacity=post_process_stage_layer_capacity,
            unit_decoder_param_bytes=unit_decoder_param_bytes,
            post_process_overhead_param_bytes=post_process_overhead_param_bytes,
            unit_activation_bytes=unit_activation_bytes,
            unit_total_bytes=unit_total_bytes,
            measured_peak_memory_bytes=measured_peak_memory_bytes,
            usable_memory_bytes=usable_memory_bytes,
            memory_target_fraction=memory_target_fraction,
        )

    @staticmethod
    def _extract_runtime_probe_summary(
        runtime_baseline_report: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        test_cases = runtime_baseline_report.get("test_cases")
        if not isinstance(test_cases, list) or not test_cases:
            raise ValueError("Runtime baseline report does not contain any test_cases.")
        summary = test_cases[0].get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError(
                "Runtime baseline report test case does not contain a summary block."
            )
        return summary

    @staticmethod
    def _find_runtime_baseline_report(base_dir: str) -> Optional[str]:
        if os.path.isfile(base_dir):
            return base_dir

        direct_candidate = os.path.join(base_dir, "runtime_baseline.json")
        if os.path.exists(direct_candidate):
            return direct_candidate

        candidates = glob.glob(
            os.path.join(base_dir, "**", "runtime_baseline.json"), recursive=True
        )
        if not candidates:
            return None
        return max(candidates, key=os.path.getmtime)

    def _save_report(self, report: DenseTuningReport) -> None:
        report_path = os.path.join(self.config.output_dir, "dense_tuning_report.json")
        with open(report_path, "w") as handle:
            json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

        summary_path = os.path.join(self.config.output_dir, "summary.txt")
        lines = [
            "DENSE TUNING REPORT",
            f"Model: {report.model_name}",
            f"Prompt/Response/Seqlen: {report.prompt_len}/{report.response_len}/{report.seqlen}",
            f"nprocs_per_node: {report.nprocs_per_node}",
            "",
            "STEP 1",
            f"Selected TP size: {report.step1.selected_tp_size}",
            (
                "Best TP task: "
                f"{report.step1.best_tp_task.task_id} "
                f"(ratio={report.step1.best_tp_task.overlap_ratio:.4f})"
            ),
            (
                "Best CP task: "
                f"{report.step1.best_cp_task.task_id} "
                f"(ratio={report.step1.best_cp_task.overlap_ratio:.4f}, "
                f"max_token_len={report.step1.best_cp_task.max_token_len})"
            ),
            (
                "Best overall task: "
                f"{report.step1.best_overall_task.source}:{report.step1.best_overall_task.task_id} "
                f"(ratio={report.step1.best_overall_task.overlap_ratio:.4f})"
            ),
            "",
            "STEP 2",
            (
                f"TP={report.step2.tp_size}, CP={report.step2.cp_size}, "
                f"max_token_len={report.step2.max_token_len}"
            ),
            (
                "Derived from step1_max_token_len="
                f"{report.step2.step1_max_token_len}, "
                f"max_cp_allowed={report.step2.max_cp_allowed}"
            ),
            "",
            "STEP 3",
            f"TransformerLayer compute time: {report.step3.compute_time_s:.6f}s",
            (
                "Activation offload modules: "
                + (
                    ", ".join(report.step3.offload_modules)
                    if report.step3.offload_modules
                    else "none"
                )
            ),
            (
                "Activation recompute: "
                f"{report.step3.recompute_kind}"
                + (
                    f" ({', '.join(report.step3.recompute_modules)})"
                    if report.step3.recompute_modules
                    else ""
                )
            ),
            "",
            "STEP 4",
            (
                f"PP={report.step4.pipeline_model_parallel_size}, "
                f"layer_distribution={report.step4.layer_distribution}"
            ),
            (
                "Unit bytes: "
                f"decoder_param={report.step4.unit_decoder_param_bytes}, "
                f"activation={report.step4.unit_activation_bytes}, "
                f"post_process_overhead={report.step4.post_process_overhead_param_bytes}"
            ),
            (
                "Usable memory bytes: "
                f"{report.step4.usable_memory_bytes} "
                f"(target_fraction={report.step4.memory_target_fraction:.2f})"
            ),
            "",
            "ASSUMPTIONS",
            *report.assumptions,
        ]
        with open(summary_path, "w") as handle:
            handle.write("\n".join(lines))
            handle.write("\n")

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dense-tuning-algorithm",
        description="Run dense-model tuning steps 1 through 4.",
    )
    parser.add_argument("--model-name", required=True, help="Model name from HuggingFace.")
    parser.add_argument("--prompt-len", type=int, required=True, help="Prompt length.")
    parser.add_argument("--response-len", type=int, required=True, help="Response length.")
    parser.add_argument(
        "--nprocs-per-node",
        type=int,
        required=True,
        help="Visible process slots per node used for TP/CP sizing.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Dense tuning output directory. Defaults to outputs/algorithm/dense_tuning/<timestamp>.",
    )
    parser.add_argument(
        "--tp-output-dir",
        type=str,
        default=None,
        help="Existing or target TP overlap output directory.",
    )
    parser.add_argument(
        "--cp-output-dir",
        type=str,
        default=None,
        help="Existing or target CP overlap output directory.",
    )
    parser.add_argument("--max-tp-size", type=int, default=8)
    parser.add_argument("--tp-max-token-len", type=int, default=8192)
    parser.add_argument(
        "--tp-operators",
        nargs="+",
        default=["fc1", "fc2", "qkv", "proj"],
        choices=["fc1", "fc2", "qkv", "proj"],
    )
    parser.add_argument("--tp-min-num-sm", type=int, default=1)
    parser.add_argument("--tp-max-num-sm", type=int, default=16)
    parser.add_argument("--skip-tp-profiling", action="store_true")
    parser.add_argument(
        "--cp-seqlen-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Optional seqlen range for CP step 1 profiling.",
    )
    parser.add_argument(
        "--cp-max-token-len",
        type=int,
        default=None,
        help="max_token_len used by CP step 1 profiling. Defaults to --tp-max-token-len.",
    )
    parser.add_argument("--cp-seqlen-step", type=int, default=1024)
    parser.add_argument("--cp-size-for-profiling", type=int, default=2)
    parser.add_argument("--cp-batch-size", type=int, default=128)
    parser.add_argument("--cp-micro-batch-size", type=int, default=2)
    parser.add_argument(
        "--cp-shape",
        type=str,
        default="thd",
        choices=["thd", "bshd"],
    )
    parser.add_argument("--cp-system", type=str, default="megatron")
    parser.add_argument("--skip-cp-profiling", action="store_true")
    parser.add_argument(
        "--cp-candidates",
        type=int,
        nargs="+",
        default=None,
        help="Optional CP candidates to consider in step 2. Defaults to all positive integers up to the TP limit.",
    )
    parser.add_argument(
        "--activation-output-dir",
        type=str,
        default=None,
        help="Existing or target activation step-3 output directory.",
    )
    parser.add_argument("--activation-batch-size", type=int, default=128)
    parser.add_argument("--activation-micro-batch-size", type=int, default=2)
    parser.add_argument(
        "--activation-shape",
        type=str,
        default="thd",
        choices=["thd", "bshd"],
    )
    parser.add_argument("--activation-system", type=str, default="megatron")
    parser.add_argument("--activation-warmup-iters", type=int, default=2)
    parser.add_argument("--activation-profile-iters", type=int, default=3)
    parser.add_argument("--activation-bandwidth-gbps", type=float, default=64.0)
    parser.add_argument(
        "--activation-bandwidth-overhead-ms", type=float, default=0.0
    )
    parser.add_argument(
        "--activation-transfer-round-trips", type=float, default=2.0
    )
    parser.add_argument("--skip-activation-profiling", action="store_true")
    parser.add_argument(
        "--runtime-output-dir",
        type=str,
        default=None,
        help="Existing or target runtime-baseline step-4 output directory.",
    )
    parser.add_argument("--runtime-batch-size", type=int, default=128)
    parser.add_argument("--runtime-micro-batch-size", type=int, default=2)
    parser.add_argument(
        "--runtime-shape",
        type=str,
        default="thd",
        choices=["thd", "bshd"],
    )
    parser.add_argument("--runtime-system", type=str, default="megatron")
    parser.add_argument("--runtime-max-iterations", type=int, default=3)
    parser.add_argument("--runtime-warmup-iterations", type=int, default=1)
    parser.add_argument(
        "--runtime-memory-target-fraction", type=float, default=0.9
    )
    parser.add_argument(
        "--runtime-full-recompute-method",
        type=str,
        default="uniform",
        choices=["uniform", "block"],
    )
    parser.add_argument(
        "--runtime-disable-fused-kernels",
        action="store_true",
        help="Disable fused runtime baseline forward path for step 4.",
    )
    parser.add_argument(
        "--runtime-enable-ddp",
        action="store_true",
        help="Wrap the step-4 runtime baseline probe with DDP.",
    )
    parser.add_argument("--skip-runtime-profiling", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> DenseTuningConfig:
    cp_seqlen_start = None
    cp_seqlen_end = None
    if args.cp_seqlen_range is not None:
        cp_seqlen_start, cp_seqlen_end = args.cp_seqlen_range

    return DenseTuningConfig(
        model_name=args.model_name,
        prompt_len=args.prompt_len,
        response_len=args.response_len,
        nprocs_per_node=args.nprocs_per_node,
        output_dir=args.output_dir or _default_output_dir(),
        tp_output_dir=args.tp_output_dir,
        cp_output_dir=args.cp_output_dir,
        max_tp_size=args.max_tp_size,
        tp_max_token_len=args.tp_max_token_len,
        tp_operators=list(args.tp_operators),
        tp_min_num_sm=args.tp_min_num_sm,
        tp_max_num_sm=args.tp_max_num_sm,
        skip_tp_profiling=args.skip_tp_profiling,
        cp_seqlen_start=cp_seqlen_start,
        cp_seqlen_end=cp_seqlen_end,
        cp_max_token_len=args.cp_max_token_len,
        cp_seqlen_step=args.cp_seqlen_step,
        cp_size_for_profiling=args.cp_size_for_profiling,
        cp_batch_size=args.cp_batch_size,
        cp_micro_batch_size=args.cp_micro_batch_size,
        cp_shape=args.cp_shape,
        cp_system=args.cp_system,
        skip_cp_profiling=args.skip_cp_profiling,
        cp_candidates=list(args.cp_candidates) if args.cp_candidates else None,
        activation_output_dir=args.activation_output_dir,
        activation_batch_size=args.activation_batch_size,
        activation_micro_batch_size=args.activation_micro_batch_size,
        activation_shape=args.activation_shape,
        activation_system=args.activation_system,
        activation_warmup_iters=args.activation_warmup_iters,
        activation_profile_iters=args.activation_profile_iters,
        activation_bandwidth_gbps=args.activation_bandwidth_gbps,
        activation_bandwidth_overhead_ms=args.activation_bandwidth_overhead_ms,
        activation_transfer_round_trips=args.activation_transfer_round_trips,
        skip_activation_profiling=args.skip_activation_profiling,
        runtime_output_dir=args.runtime_output_dir,
        runtime_batch_size=args.runtime_batch_size,
        runtime_micro_batch_size=args.runtime_micro_batch_size,
        runtime_shape=args.runtime_shape,
        runtime_system=args.runtime_system,
        runtime_max_iterations=args.runtime_max_iterations,
        runtime_warmup_iterations=args.runtime_warmup_iterations,
        runtime_use_fused_kernels=not args.runtime_disable_fused_kernels,
        runtime_no_ddp=not args.runtime_enable_ddp,
        runtime_memory_target_fraction=args.runtime_memory_target_fraction,
        runtime_full_recompute_method=args.runtime_full_recompute_method,
        skip_runtime_profiling=args.skip_runtime_profiling,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    algorithm = DenseTuningAlgorithm(config)
    report = algorithm.run()
    print("=" * 70)
    print("DENSE TUNING")
    print("=" * 70)
    print(f"Model: {report.model_name}")
    print(
        f"Prompt/Response/Seqlen: {report.prompt_len}/{report.response_len}/{report.seqlen}"
    )
    print(f"Selected TP: {report.step1.selected_tp_size}")
    print(
        "Best TP task: "
        f"{report.step1.best_tp_task.task_id} "
        f"(ratio={report.step1.best_tp_task.overlap_ratio:.4f})"
    )
    print(
        "Best CP task: "
        f"{report.step1.best_cp_task.task_id} "
        f"(ratio={report.step1.best_cp_task.overlap_ratio:.4f}, "
        f"step1_max_token_len={report.step1.best_cp_task.max_token_len})"
    )
    print(
        "Step 2 config: "
        f"TP={report.step2.tp_size}, CP={report.step2.cp_size}, "
        f"max_token_len={report.step2.max_token_len}"
    )
    print(
        "Step 3 activation strategy: "
        f"offload={report.step3.offload_modules or ['none']}, "
        f"recompute={report.step3.recompute_kind}"
        + (
            f" ({', '.join(report.step3.recompute_modules)})"
            if report.step3.recompute_modules
            else ""
        )
    )
    print(
        "Step 4 PP strategy: "
        f"pp={report.step4.pipeline_model_parallel_size}, "
        f"layer_distribution={report.step4.layer_distribution}"
    )
    print(f"JSON report: {os.path.join(config.output_dir, 'dense_tuning_report.json')}")
    print(f"Summary: {os.path.join(config.output_dir, 'summary.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

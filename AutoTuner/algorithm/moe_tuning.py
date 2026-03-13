#!/usr/bin/env python3
"""MoE-model tuning algorithm for TP/CP/max_token_len/EP decisions."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence

from AutoTuner.Profiler.expert_parallel import (
    AllToAllBandwidthModel,
    ExpertParallelProfilingConfig,
    ExpertParallelStrategyDecider,
)

from .common import (
    Step12TuningAlgorithmMixin,
    TuningStep1Result,
    TuningStep2Result,
    default_output_dir,
)


def _default_output_dir() -> str:
    return default_output_dir("moe_tuning", datetime.now().strftime("%Y%m%d_%H%M%S"))


MoETuningStep1Result = TuningStep1Result
MoETuningStep2Result = TuningStep2Result


@dataclass(frozen=True)
class MoETuningStep3Result:
    """Step 3 EP strategy derived from step 2."""

    expert_parallel_profile_path: str
    expert_parallel_strategy_path: str
    selected_ep_size: int
    selected_local_experts: int
    selected_bandwidth_tier: str
    selected_ep_fully_overlapped: bool
    comm_payload_bytes_per_phase: int
    comm_time_s_by_phase: dict[str, float]
    overlap_window_s_by_phase: dict[str, float]
    exposed_comm_time_s: float
    objective_time_s: float
    candidate_evaluations: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MoETuningReport:
    """Combined step 1 through step 3 output."""

    model_name: str
    prompt_len: int
    response_len: int
    nprocs_per_node: int
    step1: MoETuningStep1Result
    step2: MoETuningStep2Result
    step3: MoETuningStep3Result
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
            "assumptions": self.assumptions,
        }


@dataclass(frozen=True)
class MoETuningConfig:
    """Configuration for the MoE tuning algorithm."""

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
    ep_output_dir: Optional[str] = None
    ep_batch_size: int = 128
    ep_micro_batch_size: int = 2
    ep_shape: str = "thd"
    ep_system: str = "megatron"
    ep_warmup_iters: int = 2
    ep_profile_iters: int = 3
    ep_dtype_bytes: int = 2
    ep_intra_node_bandwidth_gbps: float = 300.0
    ep_inter_node_bandwidth_gbps: float = 50.0
    ep_latency_ms: float = 0.0
    ep_candidates: Optional[list[int]] = None
    skip_ep_profiling: bool = False

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
        if self.ep_batch_size <= 0 or self.ep_micro_batch_size <= 0:
            raise ValueError("ep batch sizes must be positive.")
        if self.ep_shape not in {"thd", "bshd"}:
            raise ValueError("ep_shape must be either 'thd' or 'bshd'.")
        if self.ep_warmup_iters < 0 or self.ep_profile_iters <= 0:
            raise ValueError("ep_warmup_iters must be >= 0 and ep_profile_iters > 0.")
        if self.ep_dtype_bytes <= 0:
            raise ValueError("ep_dtype_bytes must be positive.")
        if self.ep_intra_node_bandwidth_gbps <= 0:
            raise ValueError("ep_intra_node_bandwidth_gbps must be positive.")
        if self.ep_inter_node_bandwidth_gbps <= 0:
            raise ValueError("ep_inter_node_bandwidth_gbps must be positive.")
        if self.ep_latency_ms < 0:
            raise ValueError("ep_latency_ms must be non-negative.")
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
        if self.ep_candidates is not None:
            if not self.ep_candidates:
                raise ValueError("ep_candidates must not be empty.")
            if any(candidate <= 0 for candidate in self.ep_candidates):
                raise ValueError("ep_candidates must contain only positive integers.")

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
    def resolved_ep_output_dir(self) -> str:
        return self.ep_output_dir or os.path.join(self.output_dir, "expert_parallel")


class MoETuningAlgorithm(Step12TuningAlgorithmMixin):
    """Implements MoE-model tuning steps 1 through 3."""

    def __init__(self, config: MoETuningConfig):
        self.config = config

    def run(self) -> MoETuningReport:
        os.makedirs(self.config.output_dir, exist_ok=True)

        step1_result = self.run_step1()
        step2_result = self.run_step2(step1_result)
        step3_result = self.run_step3(step2_result)
        report = MoETuningReport(
            model_name=self.config.model_name,
            prompt_len=self.config.prompt_len,
            response_len=self.config.response_len,
            nprocs_per_node=self.config.nprocs_per_node,
            step1=step1_result,
            step2=step2_result,
            step3=step3_result,
            assumptions=[
                "Step 2 picks the smallest valid integer CP that satisfies seqlen <= CP * step1_max_token_len.",
                "Default CP candidates are all positive integers because the requirement only constrains CP by CP < nprocs_per_node / TP.",
                "Step 3 profiles attention and TEGroupedMLP overlap windows at EP=1, then scales TEGroupedMLP timings by local expert count.",
                "Step 3 prefers the lowest estimated EP objective among candidates whose dispatch/combine communication can be fully overlapped.",
            ],
        )
        self._save_report(report)
        return report

    def run_step3(self, step2_result: MoETuningStep2Result) -> MoETuningStep3Result:
        strategy = ExpertParallelStrategyDecider(
            ExpertParallelProfilingConfig(
                model_name=self.config.model_name,
                output_dir=self.config.resolved_ep_output_dir,
                tp_size=step2_result.tp_size,
                cp_size=step2_result.cp_size,
                seqlen=step2_result.seqlen,
                max_token_len=step2_result.max_token_len,
                nprocs_per_node=self.config.nprocs_per_node,
                batch_size=self.config.ep_batch_size,
                micro_batch_size=self.config.ep_micro_batch_size,
                shape=self.config.ep_shape,
                system=self.config.ep_system,
                warmup_iters=self.config.ep_warmup_iters,
                profile_iters=self.config.ep_profile_iters,
                dtype_bytes=self.config.ep_dtype_bytes,
                bandwidth_model=AllToAllBandwidthModel(
                    intra_node_bandwidth_gbps=self.config.ep_intra_node_bandwidth_gbps,
                    inter_node_bandwidth_gbps=self.config.ep_inter_node_bandwidth_gbps,
                    latency_seconds=self.config.ep_latency_ms / 1000.0,
                ),
                ep_candidates=self.config.ep_candidates,
            )
        ).run(skip_profiling=self.config.skip_ep_profiling)

        return MoETuningStep3Result(
            expert_parallel_profile_path=strategy.profile_path,
            expert_parallel_strategy_path=os.path.join(
                self.config.resolved_ep_output_dir,
                "expert_parallel_strategy_report.json",
            ),
            selected_ep_size=strategy.decision.ep_size,
            selected_local_experts=strategy.decision.local_experts,
            selected_bandwidth_tier=strategy.decision.bandwidth_tier,
            selected_ep_fully_overlapped=strategy.decision.can_fully_overlap,
            comm_payload_bytes_per_phase=strategy.decision.comm_payload_bytes_per_phase,
            comm_time_s_by_phase=strategy.decision.comm_time_s_by_phase,
            overlap_window_s_by_phase=strategy.decision.overlap_window_s_by_phase,
            exposed_comm_time_s=strategy.decision.exposed_comm_time_s,
            objective_time_s=strategy.decision.objective_time_s,
            candidate_evaluations=[
                candidate.to_dict() for candidate in strategy.candidate_evaluations
            ],
        )

    def _save_report(self, report: MoETuningReport) -> None:
        report_path = os.path.join(self.config.output_dir, "moe_tuning_report.json")
        with open(report_path, "w") as handle:
            json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

        summary_path = os.path.join(self.config.output_dir, "summary.txt")
        lines = [
            "MOE TUNING REPORT",
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
            "",
            "STEP 2",
            (
                f"TP={report.step2.tp_size}, CP={report.step2.cp_size}, "
                f"max_token_len={report.step2.max_token_len}"
            ),
            "",
            "STEP 3",
            (
                f"Selected EP={report.step3.selected_ep_size}, "
                f"local_experts={report.step3.selected_local_experts}, "
                f"bandwidth_tier={report.step3.selected_bandwidth_tier}"
            ),
            (
                "Full overlap: "
                f"{report.step3.selected_ep_fully_overlapped}, "
                f"exposed_comm_time_s={report.step3.exposed_comm_time_s:.6f}, "
                f"objective_time_s={report.step3.objective_time_s:.6f}"
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
        prog="moe-tuning-algorithm",
        description="Run MoE-model tuning steps 1 through 3.",
    )
    parser.add_argument("--model-name", required=True, help="Model name from HuggingFace.")
    parser.add_argument("--prompt-len", type=int, required=True, help="Prompt length.")
    parser.add_argument("--response-len", type=int, required=True, help="Response length.")
    parser.add_argument(
        "--nprocs-per-node",
        type=int,
        required=True,
        help="Visible process slots per node used for TP/CP/EP sizing.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="MoE tuning output directory. Defaults to outputs/algorithm/moe_tuning/<timestamp>.",
    )
    parser.add_argument("--tp-output-dir", type=str, default=None)
    parser.add_argument("--cp-output-dir", type=str, default=None)
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
    )
    parser.add_argument("--cp-max-token-len", type=int, default=None)
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
    parser.add_argument("--cp-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--ep-output-dir", type=str, default=None)
    parser.add_argument("--ep-batch-size", type=int, default=128)
    parser.add_argument("--ep-micro-batch-size", type=int, default=2)
    parser.add_argument(
        "--ep-shape",
        type=str,
        default="thd",
        choices=["thd", "bshd"],
    )
    parser.add_argument("--ep-system", type=str, default="megatron")
    parser.add_argument("--ep-warmup-iters", type=int, default=2)
    parser.add_argument("--ep-profile-iters", type=int, default=3)
    parser.add_argument("--ep-dtype-bytes", type=int, default=2)
    parser.add_argument("--ep-intra-node-bandwidth-gbps", type=float, default=300.0)
    parser.add_argument("--ep-inter-node-bandwidth-gbps", type=float, default=50.0)
    parser.add_argument("--ep-latency-ms", type=float, default=0.0)
    parser.add_argument("--ep-candidates", type=int, nargs="+", default=None)
    parser.add_argument("--skip-ep-profiling", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> MoETuningConfig:
    cp_seqlen_start = None
    cp_seqlen_end = None
    if args.cp_seqlen_range is not None:
        cp_seqlen_start, cp_seqlen_end = args.cp_seqlen_range

    return MoETuningConfig(
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
        ep_output_dir=args.ep_output_dir,
        ep_batch_size=args.ep_batch_size,
        ep_micro_batch_size=args.ep_micro_batch_size,
        ep_shape=args.ep_shape,
        ep_system=args.ep_system,
        ep_warmup_iters=args.ep_warmup_iters,
        ep_profile_iters=args.ep_profile_iters,
        ep_dtype_bytes=args.ep_dtype_bytes,
        ep_intra_node_bandwidth_gbps=args.ep_intra_node_bandwidth_gbps,
        ep_inter_node_bandwidth_gbps=args.ep_inter_node_bandwidth_gbps,
        ep_latency_ms=args.ep_latency_ms,
        ep_candidates=list(args.ep_candidates) if args.ep_candidates else None,
        skip_ep_profiling=args.skip_ep_profiling,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    algorithm = MoETuningAlgorithm(config)
    report = algorithm.run()
    print("=" * 70)
    print("MOE TUNING")
    print("=" * 70)
    print(f"Model: {report.model_name}")
    print(
        f"Prompt/Response/Seqlen: {report.prompt_len}/{report.response_len}/{report.seqlen}"
    )
    print(
        "Step 2 config: "
        f"TP={report.step2.tp_size}, CP={report.step2.cp_size}, "
        f"max_token_len={report.step2.max_token_len}"
    )
    print(
        "Step 3 EP strategy: "
        f"ep={report.step3.selected_ep_size}, "
        f"local_experts={report.step3.selected_local_experts}, "
        f"full_overlap={report.step3.selected_ep_fully_overlapped}"
    )
    print(f"JSON report: {os.path.join(config.output_dir, 'moe_tuning_report.json')}")
    print(f"Summary: {os.path.join(config.output_dir, 'summary.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

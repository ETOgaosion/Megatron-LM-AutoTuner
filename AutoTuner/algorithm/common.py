"""Shared tuning helpers for step-1 TP/CP analysis and step-2 TP/CP sizing."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence


def default_output_dir(step_name: str, timestamp: str) -> str:
    return os.path.join("outputs", "algorithm", step_name, timestamp)


@dataclass(frozen=True)
class Step1TaskSummary:
    """Normalized summary for the best task found in one overlap domain."""

    source: str
    task_id: str
    overlap_ratio: float
    max_token_len: int
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "task_id": self.task_id,
            "overlap_ratio": self.overlap_ratio,
            "max_token_len": self.max_token_len,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class TuningStep1Result:
    """Shared step-1 result for TP/CP overlap selection."""

    tp_report_path: str
    cp_report_path: str
    selected_tp_size: int
    best_tp_task: Step1TaskSummary
    best_cp_task: Step1TaskSummary
    best_overall_task: Step1TaskSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp_report_path": self.tp_report_path,
            "cp_report_path": self.cp_report_path,
            "selected_tp_size": self.selected_tp_size,
            "best_tp_task": self.best_tp_task.to_dict(),
            "best_cp_task": self.best_cp_task.to_dict(),
            "best_overall_task": self.best_overall_task.to_dict(),
        }


@dataclass(frozen=True)
class TuningStep2Result:
    """Shared step-2 decision derived from step 1 and user seqlen."""

    seqlen: int
    tp_size: int
    cp_size: int
    step1_max_token_len: int
    max_token_len: int
    max_cp_allowed: int
    cp_candidates_considered: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Step12TuningAlgorithmMixin:
    """Shared implementation for steps 1 and 2 across dense and MoE tuning."""

    def run_step1(self) -> TuningStep1Result:
        tp_report_path = self._ensure_tp_report()
        cp_report_path = self._ensure_cp_report()
        tp_report = self._load_json(tp_report_path)
        cp_report = self._load_json(cp_report_path)

        best_tp_task = self.select_best_tp_task(tp_report)
        best_cp_task = self.select_best_cp_task(cp_report)
        selected_tp_size = self.select_tp_size(tp_report, best_tp_task)
        best_overall_task = max(
            [best_tp_task, best_cp_task],
            key=lambda task: (task.overlap_ratio, task.max_token_len, task.task_id),
        )
        return TuningStep1Result(
            tp_report_path=tp_report_path,
            cp_report_path=cp_report_path,
            selected_tp_size=selected_tp_size,
            best_tp_task=best_tp_task,
            best_cp_task=best_cp_task,
            best_overall_task=best_overall_task,
        )

    def run_step2(self, step1_result: TuningStep1Result) -> TuningStep2Result:
        return self.decide_step2(
            step1_result=step1_result,
            seqlen=self.config.seqlen,
            nprocs_per_node=self.config.nprocs_per_node,
            cp_candidates=self.config.cp_candidates,
        )

    def _ensure_tp_report(self) -> str:
        report_path = os.path.join(
            self.config.resolved_tp_output_dir, "tuning_report.json"
        )
        if self.config.skip_tp_profiling:
            if not os.path.exists(report_path):
                raise FileNotFoundError(
                    f"TP report not found at {report_path}. "
                    "Pass an existing --tp-output-dir or disable --skip-tp-profiling."
                )
            return report_path

        from AutoTuner.Profiler.overlap.tp import TPOverlapTuner, TPOverlapTunerConfig

        tuner_config = TPOverlapTunerConfig(
            model_name=self.config.model_name,
            max_tp_size=self.config.max_tp_size,
            max_token_len=self.config.tp_max_token_len,
            operators=self.config.tp_operators,
            output_dir=self.config.resolved_tp_output_dir,
            min_num_sm=self.config.tp_min_num_sm,
            max_num_sm=self.config.tp_max_num_sm,
        )
        TPOverlapTuner(tuner_config=tuner_config).run(skip_profiling=False)
        return report_path

    def _ensure_cp_report(self) -> str:
        report_path = os.path.join(
            self.config.resolved_cp_output_dir, "cp_overlap_report.json"
        )
        if self.config.skip_cp_profiling:
            if not os.path.exists(report_path):
                raise FileNotFoundError(
                    f"CP report not found at {report_path}. "
                    "Pass an existing --cp-output-dir or disable --skip-cp-profiling."
                )
            return report_path

        from AutoTuner.Profiler.overlap.cp import CPOverlapRunner, CPOverlapRunnerConfig

        runner = CPOverlapRunner(
            CPOverlapRunnerConfig(
                model_name=self.config.model_name,
                output_dir=self.config.resolved_cp_output_dir,
                seqlen_start=self.config.resolved_cp_seqlen_start,
                seqlen_end=self.config.resolved_cp_seqlen_end,
                max_token_len=self.config.resolved_cp_max_token_len,
                cp_size=self.config.cp_size_for_profiling,
                batch_size=self.config.cp_batch_size,
                micro_batch_size=self.config.cp_micro_batch_size,
                shape=self.config.cp_shape,
                system=self.config.cp_system,
                seqlen_step=self.config.cp_seqlen_step,
            )
        )
        runner.run(skip_profiling=False)
        return report_path

    @staticmethod
    def select_best_tp_task(tp_report: Mapping[str, Any]) -> Step1TaskSummary:
        analyses = tp_report.get("analyses")
        if not isinstance(analyses, list) or not analyses:
            raise ValueError("TP report does not contain any analyses.")

        best = max(
            analyses,
            key=lambda item: (
                float(item.get("total_overlap_ratio", 0.0)),
                -float(item.get("operator_e2e_time_us", float("inf"))),
                int(item.get("tp_size", 0)),
                str(item.get("config_id", "")),
            ),
        )
        metrics = {
            "tp_size": int(best["tp_size"]),
            "operator": best.get("operator"),
            "phase": best.get("phase"),
            "total_overlap_ratio": float(best.get("total_overlap_ratio", 0.0)),
            "forward_overlap_ratio": float(best.get("forward_overlap_ratio", 0.0)),
            "backward_overlap_ratio": float(best.get("backward_overlap_ratio", 0.0)),
            "operator_e2e_time_us": float(best.get("operator_e2e_time_us", 0.0)),
        }
        tuner_config = tp_report.get("tuner_config") or {}
        return Step1TaskSummary(
            source="tp",
            task_id=str(best.get("config_id") or ""),
            overlap_ratio=float(best.get("total_overlap_ratio", 0.0)),
            max_token_len=int(tuner_config.get("max_token_len", 0)),
            metrics=metrics,
        )

    @staticmethod
    def select_tp_size(
        tp_report: Mapping[str, Any], best_tp_task: Step1TaskSummary
    ) -> int:
        tp_scaling = tp_report.get("tp_scaling")
        if isinstance(tp_scaling, Mapping):
            optimal_tp_size = tp_scaling.get("optimal_tp_size")
            if isinstance(optimal_tp_size, int) and optimal_tp_size > 0:
                return optimal_tp_size
        tp_size = int(best_tp_task.metrics.get("tp_size", 0))
        if tp_size <= 0:
            raise ValueError("Could not determine TP size from TP step 1 report.")
        return tp_size

    @staticmethod
    def select_best_cp_task(cp_report: Mapping[str, Any]) -> Step1TaskSummary:
        results = cp_report.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError("CP report does not contain any results.")

        successful_results = [
            result
            for result in results
            if result.get("success") and isinstance(result.get("analysis"), Mapping)
        ]
        if not successful_results:
            raise ValueError("CP report does not contain any successful analyses.")

        best = max(
            successful_results,
            key=lambda item: (
                float(item["analysis"].get("overlap_ratio", 0.0)),
                int((item.get("case") or {}).get("max_token_len", 0)),
                int((item.get("case") or {}).get("seqlen", 0)),
                str(item.get("case_id", "")),
            ),
        )
        case = best["case"]
        analysis = best["analysis"]
        metrics = {
            "seqlen": int(case["seqlen"]),
            "cp_size": int(cp_report.get("cp_size", 0)),
            "overlap_ratio": float(analysis.get("overlap_ratio", 0.0)),
            "compute_time_us": float(analysis.get("compute_time_us", 0.0)),
            "comm_time_us": float(analysis.get("comm_time_us", 0.0)),
            "overlapped_comm_time_us": float(
                analysis.get("overlapped_comm_time_us", 0.0)
            ),
        }
        return Step1TaskSummary(
            source="cp",
            task_id=str(best.get("case_id") or ""),
            overlap_ratio=float(analysis.get("overlap_ratio", 0.0)),
            max_token_len=int(case["max_token_len"]),
            metrics=metrics,
        )

    @classmethod
    def decide_step2(
        cls,
        step1_result: TuningStep1Result,
        seqlen: int,
        nprocs_per_node: int,
        cp_candidates: Optional[Sequence[int]] = None,
    ) -> TuningStep2Result:
        if seqlen <= 0:
            raise ValueError("seqlen must be positive.")
        if nprocs_per_node <= 0:
            raise ValueError("nprocs_per_node must be positive.")

        tp_size = step1_result.selected_tp_size
        if tp_size <= 0:
            raise ValueError("selected TP size must be positive.")

        step1_max_token_len = step1_result.best_cp_task.max_token_len
        if step1_max_token_len <= 0:
            raise ValueError("Step 1 max_token_len must be positive.")

        max_cp_allowed = cls._max_cp_allowed(tp_size, nprocs_per_node)
        if max_cp_allowed < 1:
            raise ValueError(
                "No valid CP value satisfies CP < nprocs_per_node / TP with "
                f"nprocs_per_node={nprocs_per_node} and TP={tp_size}."
            )

        required_cp = max(1, math.ceil(seqlen / step1_max_token_len))
        candidate_pool = sorted(set(cp_candidates or range(1, max_cp_allowed + 1)))
        valid_candidates = [
            candidate
            for candidate in candidate_pool
            if candidate >= required_cp and candidate <= max_cp_allowed
        ]
        if not valid_candidates:
            raise ValueError(
                "No CP candidate can cover the requested seqlen under the TP limit: "
                f"required_cp={required_cp}, max_cp_allowed={max_cp_allowed}, "
                f"candidates={candidate_pool}"
            )

        cp_size = valid_candidates[0]
        return TuningStep2Result(
            seqlen=seqlen,
            tp_size=tp_size,
            cp_size=cp_size,
            step1_max_token_len=step1_max_token_len,
            max_token_len=cp_size * step1_max_token_len,
            max_cp_allowed=max_cp_allowed,
            cp_candidates_considered=candidate_pool,
        )

    @staticmethod
    def _max_cp_allowed(tp_size: int, nprocs_per_node: int) -> int:
        ratio = nprocs_per_node / tp_size
        max_cp = math.ceil(ratio) - 1
        while max_cp > 0 and not (max_cp < ratio):
            max_cp -= 1
        return max_cp

    @staticmethod
    def _load_json(path: str) -> dict[str, Any]:
        with open(path, "r") as handle:
            return json.load(handle)

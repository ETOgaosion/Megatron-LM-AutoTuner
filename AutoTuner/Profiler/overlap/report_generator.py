"""
Report Generator for TP Overlap Tuning.

This module generates tuning reports and optimal YAML configurations
based on overlap analysis results.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

from .config_generator import (
    DEFAULT_CONFIGS,
    OverlapMethod,
    TPOverlapTestConfig,
    TPOverlapTunerConfig,
)
from .overlap_detector import OverlapAnalysis


@dataclass
class OperatorAnalysisSummary:
    """Summary of analysis results for a single operator."""

    operator: str
    tp_size: int
    best_fprop_config: Optional[TPOverlapTestConfig] = None
    best_dgrad_config: Optional[TPOverlapTestConfig] = None
    best_wgrad_config: Optional[TPOverlapTestConfig] = None
    best_fprop_analysis: Optional[OverlapAnalysis] = None
    best_dgrad_analysis: Optional[OverlapAnalysis] = None
    best_wgrad_analysis: Optional[OverlapAnalysis] = None
    all_analyses: List[OverlapAnalysis] = field(default_factory=list)

    @property
    def has_effective_overlap(self) -> bool:
        """Check if any phase has effective overlap (> 50%)."""
        threshold = 0.5
        if (
            self.best_fprop_analysis
            and self.best_fprop_analysis.forward_overlap_ratio >= threshold
        ):
            return True
        if (
            self.best_dgrad_analysis
            and self.best_dgrad_analysis.backward_overlap_ratio >= threshold
        ):
            return True
        if (
            self.best_wgrad_analysis
            and self.best_wgrad_analysis.backward_overlap_ratio >= threshold
        ):
            return True
        return False


@dataclass
class TPScalingResult:
    """Result of TP scaling efficiency analysis for an operator."""

    operator: str
    optimal_tp_size: int
    scaling_efficient: Dict[int, bool]  # tp_size -> whether scaling is efficient
    scaling_ratios: Dict[int, float]  # tp_size -> actual_time / expected_time
    e2e_times: Dict[int, float]  # tp_size -> e2e time
    reason: str  # Human-readable explanation


@dataclass
class TuningReport:
    """Complete tuning report."""

    tuner_config: TPOverlapTunerConfig
    operator_summaries: Dict[str, Dict[int, OperatorAnalysisSummary]] = field(
        default_factory=dict
    )  # operator -> tp_size -> summary
    all_analyses: List[OverlapAnalysis] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    recommendations: List[str] = field(default_factory=list)
    tp_scaling_results: Dict[str, TPScalingResult] = field(
        default_factory=dict
    )  # operator -> scaling result
    optimal_tp_size: int = 2  # Overall optimal TP size based on scaling efficiency

    def get_best_config_for_operator(
        self, operator: str, tp_size: int, phase: str
    ) -> Optional[TPOverlapTestConfig]:
        """Get the best config for a specific operator, TP size, and phase."""
        if operator not in self.operator_summaries:
            return None
        if tp_size not in self.operator_summaries[operator]:
            return None

        summary = self.operator_summaries[operator][tp_size]
        if phase == "fprop":
            return summary.best_fprop_config
        elif phase == "dgrad":
            return summary.best_dgrad_config
        elif phase == "wgrad":
            return summary.best_wgrad_config
        return None


class ReportGenerator:
    """Generates reports and optimal configurations from overlap analysis."""

    # Tolerance for TP scaling efficiency check.
    # If actual_time <= expected_time * (1 + tolerance), scaling is considered efficient.
    TP_SCALING_TOLERANCE = 0.2  # 20% tolerance

    def __init__(self, overlap_threshold: float = 0.5):
        """Initialize the report generator.

        Args:
            overlap_threshold: Minimum overlap ratio to consider overlap effective.
        """
        self.overlap_threshold = overlap_threshold

    def generate(
        self,
        results: List[OverlapAnalysis],
        tuner_config: TPOverlapTunerConfig,
    ) -> TuningReport:
        """Generate a tuning report from analysis results.

        Args:
            results: List of OverlapAnalysis results.
            tuner_config: The tuner configuration used.

        Returns:
            TuningReport with analysis summaries and recommendations.
        """
        report = TuningReport(tuner_config=tuner_config, all_analyses=results)

        # Group results by operator and TP size
        grouped = self._group_by_operator_and_tp(results)

        # Find best configs for each operator/phase
        for operator, tp_dict in grouped.items():
            report.operator_summaries[operator] = {}
            for tp_size, analyses in tp_dict.items():
                summary = self._analyze_operator(operator, tp_size, analyses)
                report.operator_summaries[operator][tp_size] = summary

        # Analyze TP scaling efficiency
        report.tp_scaling_results = self._analyze_tp_scaling(report)
        report.optimal_tp_size = self._determine_overall_optimal_tp(
            report.tp_scaling_results
        )

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _group_by_operator_and_tp(
        self, results: List[OverlapAnalysis]
    ) -> Dict[str, Dict[int, List[OverlapAnalysis]]]:
        """Group results by operator and TP size."""
        grouped: Dict[str, Dict[int, List[OverlapAnalysis]]] = {}

        for analysis in results:
            operator = analysis.config.operator
            tp_size = analysis.config.tp_size

            if operator not in grouped:
                grouped[operator] = {}
            if tp_size not in grouped[operator]:
                grouped[operator][tp_size] = []

            grouped[operator][tp_size].append(analysis)

        return grouped

    def _analyze_operator(
        self, operator: str, tp_size: int, analyses: List[OverlapAnalysis]
    ) -> OperatorAnalysisSummary:
        """Analyze results for a single operator and find best configs."""
        summary = OperatorAnalysisSummary(
            operator=operator, tp_size=tp_size, all_analyses=analyses
        )

        # Group by phase
        fprop_analyses = [a for a in analyses if a.config.phase == "fprop"]
        dgrad_analyses = [a for a in analyses if a.config.phase == "dgrad"]
        wgrad_analyses = [a for a in analyses if a.config.phase == "wgrad"]

        # Find best for each phase (highest overlap ratio)
        if fprop_analyses:
            best_fprop = max(fprop_analyses, key=lambda a: a.forward_overlap_ratio)
            summary.best_fprop_analysis = best_fprop
            summary.best_fprop_config = best_fprop.config

        if dgrad_analyses:
            best_dgrad = max(dgrad_analyses, key=lambda a: a.backward_overlap_ratio)
            summary.best_dgrad_analysis = best_dgrad
            summary.best_dgrad_config = best_dgrad.config

        if wgrad_analyses:
            best_wgrad = max(wgrad_analyses, key=lambda a: a.backward_overlap_ratio)
            summary.best_wgrad_analysis = best_wgrad
            summary.best_wgrad_config = best_wgrad.config

        return summary

    def _analyze_tp_scaling(self, report: TuningReport) -> Dict[str, TPScalingResult]:
        """Analyze TP scaling efficiency for each operator.

        Uses TP=1 (no tensor parallelism) as the baseline for comparison.
        For each operator, checks if TP sizes achieve expected speedup:
        - If TP=n, then Time(TP=n) should be ≈ Time(TP=1) / n
        - If TP doesn't provide expected speedup, recommend smaller TP or no TP

        Args:
            report: The tuning report with operator summaries.

        Returns:
            Dict mapping operator to TPScalingResult.
        """
        results = {}

        for operator, tp_dict in report.operator_summaries.items():
            # Collect e2e times for each TP size (using operator_e2e_time)
            e2e_times: Dict[int, float] = {}
            for tp_size, summary in tp_dict.items():
                # Get the e2e time from the best analysis (any phase)
                best_analysis = (
                    summary.best_fprop_analysis
                    or summary.best_dgrad_analysis
                    or summary.best_wgrad_analysis
                )
                if best_analysis and best_analysis.operator_e2e_time > 0:
                    e2e_times[tp_size] = best_analysis.operator_e2e_time

            if not e2e_times:
                continue

            # Sort TP sizes
            sorted_tp_sizes = sorted(e2e_times.keys())
            if len(sorted_tp_sizes) < 1:
                continue

            # Use TP=1 as the baseline if available, otherwise use smallest TP
            # TP=1 represents no tensor parallelism (pure computation, no comm overhead)
            if 1 in e2e_times:
                base_tp = 1
                base_time = e2e_times[1]
            else:
                base_tp = sorted_tp_sizes[0]
                base_time = e2e_times[base_tp]

            scaling_efficient: Dict[int, bool] = {base_tp: True}
            scaling_ratios: Dict[int, float] = {base_tp: 1.0}
            optimal_tp = base_tp
            reasons = []

            # Check scaling efficiency for each TP size (skip the base)
            tp_sizes_to_check = [tp for tp in sorted_tp_sizes if tp != base_tp]
            for tp_size in tp_sizes_to_check:
                # Expected time: base_time / tp_size (for TP=1 baseline)
                # or base_time * (base_tp / tp_size) for non-TP=1 baseline
                # If TP=2, time should be 1/2 of TP=1
                # If TP=4, time should be 1/4 of TP=1
                if base_tp == 1:
                    expected_time = base_time / tp_size
                else:
                    scale_factor = base_tp / tp_size
                    expected_time = base_time * scale_factor

                actual_time = e2e_times[tp_size]

                # Calculate ratio: actual / expected
                # ratio <= 1.0 means better than expected
                # ratio > 1.0 + tolerance means worse than expected (communication overhead)
                ratio = (
                    actual_time / expected_time if expected_time > 0 else float("inf")
                )
                scaling_ratios[tp_size] = ratio

                # Check if scaling is efficient (within tolerance)
                is_efficient = ratio <= (1.0 + self.TP_SCALING_TOLERANCE)
                scaling_efficient[tp_size] = is_efficient

                if is_efficient:
                    optimal_tp = tp_size
                    reasons.append(
                        f"TP={tp_size}: {actual_time:.1f}us vs expected {expected_time:.1f}us "
                        f"(ratio={ratio:.2f}, EFFICIENT)"
                    )
                else:
                    reasons.append(
                        f"TP={tp_size}: {actual_time:.1f}us vs expected {expected_time:.1f}us "
                        f"(ratio={ratio:.2f}, NOT efficient - use TP={optimal_tp})"
                    )
                    # Stop checking larger TP sizes once we find inefficient scaling
                    break

            # Build reason string
            if base_tp == 1:
                reason_prefix = f"Baseline: TP=1 (no TP) @ {base_time:.1f}us. "
            else:
                reason_prefix = f"Base: TP={base_tp} @ {base_time:.1f}us. "

            reason = (
                reason_prefix + "; ".join(reasons)
                if reasons
                else f"Only TP={base_tp} tested @ {base_time:.1f}us"
            )

            results[operator] = TPScalingResult(
                operator=operator,
                optimal_tp_size=optimal_tp,
                scaling_efficient=scaling_efficient,
                scaling_ratios=scaling_ratios,
                e2e_times=e2e_times,
                reason=reason,
            )

        return results

    def _determine_overall_optimal_tp(
        self, tp_scaling_results: Dict[str, TPScalingResult]
    ) -> int:
        """Determine the overall optimal TP size based on all operators.

        Uses the minimum optimal TP size across all operators to ensure
        all operators scale efficiently.

        Args:
            tp_scaling_results: TP scaling results for each operator.

        Returns:
            Overall optimal TP size.
        """
        if not tp_scaling_results:
            return 2  # Default to smallest TP

        optimal_tp_sizes = [r.optimal_tp_size for r in tp_scaling_results.values()]
        # Use minimum to ensure all operators are efficient
        return min(optimal_tp_sizes)

    def _generate_recommendations(self, report: TuningReport) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []

        # Add TP scaling efficiency recommendations first
        if report.tp_scaling_results:
            recommendations.append(f"=== TP SCALING ANALYSIS ===")
            recommendations.append(
                f"Overall optimal TP size: {report.optimal_tp_size} "
                f"(based on scaling efficiency)"
            )
            for operator, result in report.tp_scaling_results.items():
                recommendations.append(
                    f"{operator}: optimal TP={result.optimal_tp_size} - {result.reason}"
                )
            recommendations.append("")
            recommendations.append("=== OVERLAP ANALYSIS ===")

        for operator, tp_dict in report.operator_summaries.items():
            for tp_size, summary in tp_dict.items():
                prefix = f"TP={tp_size}, {operator}"

                # Check fprop
                if summary.best_fprop_analysis:
                    ratio = summary.best_fprop_analysis.forward_overlap_ratio
                    if ratio >= self.overlap_threshold:
                        cfg = summary.best_fprop_config
                        recommendations.append(
                            f"{prefix} fprop: Use {cfg.overlap_method.value} "
                            f"(overlap ratio: {ratio:.2%})"
                        )
                    else:
                        recommendations.append(
                            f"{prefix} fprop: Overlap not effective "
                            f"(ratio: {ratio:.2%} < {self.overlap_threshold:.0%})"
                        )

                # Check dgrad
                if summary.best_dgrad_analysis:
                    ratio = summary.best_dgrad_analysis.backward_overlap_ratio
                    if ratio >= self.overlap_threshold:
                        cfg = summary.best_dgrad_config
                        method_info = cfg.overlap_method.value
                        if cfg.overlap_method == OverlapMethod.BULK:
                            method_info += f" (num_sm={cfg.num_sm})"
                        recommendations.append(
                            f"{prefix} dgrad: Use {method_info} "
                            f"(overlap ratio: {ratio:.2%})"
                        )
                    else:
                        recommendations.append(
                            f"{prefix} dgrad: Overlap not effective "
                            f"(ratio: {ratio:.2%} < {self.overlap_threshold:.0%})"
                        )

                # Check wgrad
                if summary.best_wgrad_analysis:
                    ratio = summary.best_wgrad_analysis.backward_overlap_ratio
                    if ratio >= self.overlap_threshold:
                        cfg = summary.best_wgrad_config
                        method_info = cfg.overlap_method.value
                        if cfg.overlap_method == OverlapMethod.BULK:
                            method_info += f" (num_sm={cfg.num_sm})"
                        recommendations.append(
                            f"{prefix} wgrad: Use {method_info} "
                            f"(overlap ratio: {ratio:.2%})"
                        )
                    else:
                        recommendations.append(
                            f"{prefix} wgrad: Overlap not effective "
                            f"(ratio: {ratio:.2%} < {self.overlap_threshold:.0%})"
                        )

        return recommendations

    def generate_optimal_yaml(
        self, report: TuningReport, tp_size: int
    ) -> Dict[str, Dict[str, Any]]:
        """Generate optimal YAML config for a specific TP size.

        Args:
            report: The tuning report.
            tp_size: The TP size to generate config for.

        Returns:
            Dictionary suitable for YAML serialization.
        """
        # Start with default configs
        yaml_config = {}
        for key, val in DEFAULT_CONFIGS.items():
            yaml_config[key] = {
                "method": val["method"].value,
            }
            if val["method"] == OverlapMethod.RING_EXCHANGE:
                yaml_config[key]["aggregate"] = val.get("aggregate", 0)
            elif val["method"] == OverlapMethod.BULK:
                yaml_config[key]["num_sm"] = val.get("num_sm", 2)
                yaml_config[key]["set_sm_margin"] = val.get("set_sm_margin", 0)

        # Override with best configs from analysis
        for operator, tp_dict in report.operator_summaries.items():
            if tp_size not in tp_dict:
                continue

            summary = tp_dict[tp_size]

            # Update fprop config
            if (
                summary.best_fprop_config
                and summary.best_fprop_analysis
                and summary.best_fprop_analysis.forward_overlap_ratio
                >= self.overlap_threshold
            ):
                key = f"{operator}_fprop"
                yaml_config[key] = summary.best_fprop_config.to_yaml_dict()

            # Update dgrad config
            if (
                summary.best_dgrad_config
                and summary.best_dgrad_analysis
                and summary.best_dgrad_analysis.backward_overlap_ratio
                >= self.overlap_threshold
            ):
                key = f"{operator}_dgrad"
                yaml_config[key] = summary.best_dgrad_config.to_yaml_dict()

            # Update wgrad config
            if (
                summary.best_wgrad_config
                and summary.best_wgrad_analysis
                and summary.best_wgrad_analysis.backward_overlap_ratio
                >= self.overlap_threshold
            ):
                key = f"{operator}_wgrad"
                yaml_config[key] = summary.best_wgrad_config.to_yaml_dict()

        return yaml_config

    def save_report(self, report: TuningReport, output_dir: str) -> None:
        """Save the tuning report to files.

        Creates:
        - tuning_report.json: Full report in JSON format
        - summary.txt: Human-readable summary
        - optimal_tp_comm_overlap_cfg.yaml: Best config for each TP size
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON report
        json_path = os.path.join(output_dir, "tuning_report.json")
        self._save_json_report(report, json_path)

        # Save text summary
        summary_path = os.path.join(output_dir, "summary.txt")
        self._save_text_summary(report, summary_path)

        # Save optimal YAML for each TP size found in results
        tp_sizes = set()
        for op_dict in report.operator_summaries.values():
            tp_sizes.update(op_dict.keys())

        for tp_size in sorted(tp_sizes):
            yaml_config = self.generate_optimal_yaml(report, tp_size)
            yaml_path = os.path.join(
                output_dir, f"optimal_tp_comm_overlap_cfg_tp{tp_size}.yaml"
            )
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

        # Save the default optimal config using the optimal TP size from scaling analysis
        # This is based on: if TP_new = n × TP, Time(TP_new) should be 1/n × Time(TP)
        if tp_sizes:
            # Use the optimal TP size determined by scaling efficiency analysis
            optimal_tp = report.optimal_tp_size
            if optimal_tp not in tp_sizes:
                # Fallback to smallest if optimal TP was not tested
                optimal_tp = min(tp_sizes)

            yaml_config = self.generate_optimal_yaml(report, optimal_tp)
            yaml_path = os.path.join(output_dir, "optimal_tp_comm_overlap_cfg.yaml")
            with open(yaml_path, "w") as f:
                # Add header comment explaining why this TP size was chosen
                f.write(f"# Optimal TP size: {optimal_tp}\n")
                f.write("# Selected based on TP scaling efficiency analysis\n")
                f.write(
                    "# Rule: If TP_new = n × TP, Time(TP_new) should be ≈ 1/n × Time(TP)\n"
                )
                f.write("#\n")
                yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

    def _save_json_report(self, report: TuningReport, path: str) -> None:
        """Save the full report as JSON."""
        data = {
            "timestamp": report.timestamp,
            "tuner_config": {
                "model_name": report.tuner_config.model_name,
                "hidden_size": report.tuner_config.hidden_size,
                "ffn_hidden_size": report.tuner_config.ffn_hidden_size,
                "num_attention_heads": report.tuner_config.num_attention_heads,
                "num_kv_heads": report.tuner_config.num_kv_heads,
                "max_tp_size": report.tuner_config.max_tp_size,
                "max_token_len": report.tuner_config.max_token_len,
                "operators": report.tuner_config.operators,
            },
            "analyses": [a.to_dict() for a in report.all_analyses],
            "recommendations": report.recommendations,
            "tp_scaling": {
                "optimal_tp_size": report.optimal_tp_size,
                "tolerance": self.TP_SCALING_TOLERANCE,
                "results": {
                    op: {
                        "optimal_tp_size": r.optimal_tp_size,
                        "scaling_efficient": r.scaling_efficient,
                        "scaling_ratios": r.scaling_ratios,
                        "e2e_times": r.e2e_times,
                        "reason": r.reason,
                    }
                    for op, r in report.tp_scaling_results.items()
                },
            },
            "operator_summaries": {},
        }

        # Add operator summaries
        for operator, tp_dict in report.operator_summaries.items():
            data["operator_summaries"][operator] = {}
            for tp_size, summary in tp_dict.items():
                data["operator_summaries"][operator][str(tp_size)] = {
                    "has_effective_overlap": summary.has_effective_overlap,
                    "best_fprop": (
                        summary.best_fprop_config.get_test_id()
                        if summary.best_fprop_config
                        else None
                    ),
                    "best_dgrad": (
                        summary.best_dgrad_config.get_test_id()
                        if summary.best_dgrad_config
                        else None
                    ),
                    "best_wgrad": (
                        summary.best_wgrad_config.get_test_id()
                        if summary.best_wgrad_config
                        else None
                    ),
                }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_text_summary(self, report: TuningReport, path: str) -> None:
        """Save a human-readable summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("TP OVERLAP TUNING REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {report.timestamp}")
        lines.append(f"Model: {report.tuner_config.model_name}")
        lines.append(f"Hidden Size: {report.tuner_config.hidden_size}")
        lines.append(f"FFN Hidden Size: {report.tuner_config.ffn_hidden_size}")
        lines.append(f"Num Attention Heads: {report.tuner_config.num_attention_heads}")
        lines.append(f"Num KV Heads: {report.tuner_config.num_kv_heads}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("TP SCALING EFFICIENCY")
        lines.append("-" * 60)
        lines.append(f"Overall Optimal TP Size: {report.optimal_tp_size}")
        lines.append(
            "Rule: If TP_new = n × TP, Time(TP_new) should be ≈ 1/n × Time(TP)"
        )
        lines.append(f"Tolerance: {self.TP_SCALING_TOLERANCE:.0%}")
        lines.append("")

        for operator, result in report.tp_scaling_results.items():
            lines.append(f"  {operator}:")
            lines.append(f"    Optimal TP: {result.optimal_tp_size}")
            lines.append(f"    E2E Times: {result.e2e_times}")
            lines.append(f"    Scaling Ratios: {result.scaling_ratios}")
            lines.append(f"    Analysis: {result.reason}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 60)
        for rec in report.recommendations:
            lines.append(f"  * {rec}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("DETAILED RESULTS")
        lines.append("-" * 60)

        for operator, tp_dict in sorted(report.operator_summaries.items()):
            lines.append(f"\n{operator.upper()}")
            lines.append("-" * 40)

            for tp_size, summary in sorted(tp_dict.items()):
                lines.append(f"\n  TP Size: {tp_size}")

                if summary.best_fprop_analysis:
                    a = summary.best_fprop_analysis
                    lines.append(
                        f"    fprop: GEMM={a.forward_gemm_time:.1f}us, "
                        f"Comm={a.forward_comm_time:.1f}us, "
                        f"Overlap={a.forward_overlap_time:.1f}us "
                        f"({a.forward_overlap_ratio:.1%})"
                    )
                    lines.append(f"           E2E={a.forward_e2e_time:.1f}us")

                if summary.best_dgrad_analysis:
                    a = summary.best_dgrad_analysis
                    lines.append(
                        f"    dgrad: GEMM={a.backward_gemm_time:.1f}us, "
                        f"Comm={a.backward_comm_time:.1f}us, "
                        f"Overlap={a.backward_overlap_time:.1f}us "
                        f"({a.backward_overlap_ratio:.1%})"
                    )
                    lines.append(f"           E2E={a.backward_e2e_time:.1f}us")

                if summary.best_wgrad_analysis:
                    a = summary.best_wgrad_analysis
                    lines.append(
                        f"    wgrad: GEMM={a.backward_gemm_time:.1f}us, "
                        f"Comm={a.backward_comm_time:.1f}us, "
                        f"Overlap={a.backward_overlap_time:.1f}us "
                        f"({a.backward_overlap_ratio:.1%})"
                    )
                    lines.append(f"           E2E={a.backward_e2e_time:.1f}us")

                # Show total operator e2e time if we have any analysis
                best_analysis = (
                    summary.best_fprop_analysis
                    or summary.best_dgrad_analysis
                    or summary.best_wgrad_analysis
                )
                if best_analysis:
                    lines.append(
                        f"    Total Operator E2E: {best_analysis.operator_e2e_time:.1f}us"
                    )

        lines.append("")
        lines.append("=" * 60)

        with open(path, "w") as f:
            f.write("\n".join(lines))

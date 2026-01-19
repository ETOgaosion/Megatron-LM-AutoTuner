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
        if self.best_fprop_analysis and self.best_fprop_analysis.forward_overlap_ratio >= threshold:
            return True
        if self.best_dgrad_analysis and self.best_dgrad_analysis.backward_overlap_ratio >= threshold:
            return True
        if self.best_wgrad_analysis and self.best_wgrad_analysis.backward_overlap_ratio >= threshold:
            return True
        return False


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

    def _generate_recommendations(self, report: TuningReport) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []

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

        # Also save a default optimal config (using the smallest TP size)
        if tp_sizes:
            default_tp = min(tp_sizes)
            yaml_config = self.generate_optimal_yaml(report, default_tp)
            yaml_path = os.path.join(output_dir, "optimal_tp_comm_overlap_cfg.yaml")
            with open(yaml_path, "w") as f:
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

                if summary.best_dgrad_analysis:
                    a = summary.best_dgrad_analysis
                    lines.append(
                        f"    dgrad: GEMM={a.backward_gemm_time:.1f}us, "
                        f"Comm={a.backward_comm_time:.1f}us, "
                        f"Overlap={a.backward_overlap_time:.1f}us "
                        f"({a.backward_overlap_ratio:.1%})"
                    )

                if summary.best_wgrad_analysis:
                    a = summary.best_wgrad_analysis
                    lines.append(
                        f"    wgrad: GEMM={a.backward_gemm_time:.1f}us, "
                        f"Comm={a.backward_comm_time:.1f}us, "
                        f"Overlap={a.backward_overlap_time:.1f}us "
                        f"({a.backward_overlap_ratio:.1%})"
                    )

        lines.append("")
        lines.append("=" * 60)

        with open(path, "w") as f:
            f.write("\n".join(lines))

"""Algorithm entry points for higher-level tuning workflows."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DenseTuningConfig",
    "DenseTuningReport",
    "DenseTuningStep1Result",
    "DenseTuningStep2Result",
    "DenseTuningStep3Result",
    "DenseTuningStep4Result",
    "DenseTuningAlgorithm",
    "MoETuningConfig",
    "MoETuningReport",
    "MoETuningStep1Result",
    "MoETuningStep2Result",
    "MoETuningStep3Result",
    "MoETuningAlgorithm",
    "Step1TaskSummary",
    "run_auto_tuning",
    "run_dense_tuning",
    "run_moe_tuning",
]


def __getattr__(name: str) -> Any:
    if name in {
        "DenseTuningConfig",
        "DenseTuningReport",
        "DenseTuningStep1Result",
        "DenseTuningStep2Result",
        "DenseTuningStep3Result",
        "DenseTuningStep4Result",
        "DenseTuningAlgorithm",
        "MoETuningConfig",
        "MoETuningReport",
        "MoETuningStep1Result",
        "MoETuningStep2Result",
        "MoETuningStep3Result",
        "MoETuningAlgorithm",
        "Step1TaskSummary",
        "run_auto_tuning",
        "run_dense_tuning",
        "run_moe_tuning",
    }:
        from .main import main as auto_main
        from .dense_tuning import (
            DenseTuningAlgorithm,
            DenseTuningConfig,
            DenseTuningReport,
            DenseTuningStep1Result,
            DenseTuningStep2Result,
            DenseTuningStep3Result,
            DenseTuningStep4Result,
            Step1TaskSummary,
            main,
        )
        from .moe_tuning import (
            MoETuningAlgorithm,
            MoETuningConfig,
            MoETuningReport,
            MoETuningStep1Result,
            MoETuningStep2Result,
            MoETuningStep3Result,
            main as moe_main,
        )

        exports = {
            "DenseTuningConfig": DenseTuningConfig,
            "DenseTuningReport": DenseTuningReport,
            "DenseTuningStep1Result": DenseTuningStep1Result,
            "DenseTuningStep2Result": DenseTuningStep2Result,
            "DenseTuningStep3Result": DenseTuningStep3Result,
            "DenseTuningStep4Result": DenseTuningStep4Result,
            "DenseTuningAlgorithm": DenseTuningAlgorithm,
            "MoETuningConfig": MoETuningConfig,
            "MoETuningReport": MoETuningReport,
            "MoETuningStep1Result": MoETuningStep1Result,
            "MoETuningStep2Result": MoETuningStep2Result,
            "MoETuningStep3Result": MoETuningStep3Result,
            "MoETuningAlgorithm": MoETuningAlgorithm,
            "Step1TaskSummary": Step1TaskSummary,
            "run_auto_tuning": auto_main,
            "run_dense_tuning": main,
            "run_moe_tuning": moe_main,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

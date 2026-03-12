"""
TP overlap tuning package.

This subpackage contains TP-specific config generation, orchestration, report
generation, and TP-only trace helpers.
"""

from .config_generator import (
    LinearType,
    OverlapMethod,
    Phase,
    TPOverlapConfigGenerator,
    TPOverlapTestConfig,
    TPOverlapTunerConfig,
    generate_single_test_yaml,
    generate_yaml_config_file,
    load_yaml_config,
)
from .main import main as run_tuner
from .report_generator import (
    OperatorAnalysisSummary,
    ReportGenerator,
    TPScalingResult,
    TuningReport,
)
from .tuner import TPOverlapTuner

__all__ = [
    "TPOverlapTuner",
    "TPOverlapTunerConfig",
    "TPOverlapConfigGenerator",
    "TPOverlapTestConfig",
    "OverlapMethod",
    "LinearType",
    "Phase",
    "generate_single_test_yaml",
    "generate_yaml_config_file",
    "load_yaml_config",
    "ReportGenerator",
    "TuningReport",
    "OperatorAnalysisSummary",
    "TPScalingResult",
    "run_tuner",
]

"""
TP Overlap Tuner Package.

This package provides tools for auto-tuning TP (Tensor Parallel)
communication/computation overlap configurations for RLHF training.

Main components:
- TPOverlapTuner: Main orchestrator for the tuning workflow
- TPOverlapConfigGenerator: Generates test configurations
- TraceAnalyzer: Parses torch profiler JSON traces
- OverlapDetector: Detects compute/comm overlap from traces
- ReportGenerator: Generates tuning reports and optimal YAML configs

Example usage:
    from AutoTuner.Profiler.overlap import TPOverlapTuner, TPOverlapTunerConfig

    # Model parameters (hidden_size, ffn_hidden_size, etc.) are auto-fetched
    # from HuggingFace based on model_name
    config = TPOverlapTunerConfig(
        model_name="Qwen/Qwen3-0.6B",  # Model params auto-fetched
        max_tp_size=8,
        operators=["fc1", "fc2", "qkv", "proj"],
        output_dir="outputs/tp_overlap_tuner",
    )

    # Access auto-fetched model parameters
    print(f"Hidden Size: {config.hidden_size}")
    print(f"FFN Hidden Size: {config.ffn_hidden_size}")

    tuner = TPOverlapTuner(config)
    report = tuner.run()

    # Check TP scaling efficiency results
    print(f"Optimal TP Size: {report.optimal_tp_size}")
    for op, result in report.tp_scaling_results.items():
        print(f"{op}: {result.reason}")
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
from .overlap_detector import (
    OverlapAnalysis,
    OverlapDetector,
    TimeInterval,
    calculate_overlap_ratio,
    is_overlap_effective,
)
from .report_generator import (
    OperatorAnalysisSummary,
    ReportGenerator,
    TPScalingResult,
    TuningReport,
)
from .trace_analyzer import (
    EventCategory,
    EventType,
    TraceAnalyzer,
    TraceEvent,
    TraceMetadata,
    analyze_trace_file,
)
from .main import main as run_tuner
from .tuner import TPOverlapTuner

__all__ = [
    # Main classes
    "TPOverlapTuner",
    "TPOverlapTunerConfig",
    "TPOverlapConfigGenerator",
    "TPOverlapTestConfig",
    # Trace analysis
    "TraceAnalyzer",
    "TraceEvent",
    "TraceMetadata",
    "EventCategory",
    "EventType",
    "analyze_trace_file",
    # Overlap detection
    "OverlapDetector",
    "OverlapAnalysis",
    "TimeInterval",
    "calculate_overlap_ratio",
    "is_overlap_effective",
    # Report generation
    "ReportGenerator",
    "TuningReport",
    "OperatorAnalysisSummary",
    "TPScalingResult",
    # Config utilities
    "OverlapMethod",
    "LinearType",
    "Phase",
    "generate_single_test_yaml",
    "generate_yaml_config_file",
    "load_yaml_config",
    # Entry point
    "run_tuner",  # Main entry point (main.py)
]

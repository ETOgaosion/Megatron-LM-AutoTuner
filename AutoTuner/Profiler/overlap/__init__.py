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

    config = TPOverlapTunerConfig(
        model_name="Qwen/Qwen3-0.6B",
        hidden_size=1024,
        ffn_hidden_size=3072,
        num_attention_heads=16,
        num_kv_heads=8,
        max_tp_size=8,
        operators=["fc1", "fc2", "qkv", "proj"],
        output_dir="outputs/tp_overlap_tuner",
    )

    tuner = TPOverlapTuner(config)
    report = tuner.run()
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

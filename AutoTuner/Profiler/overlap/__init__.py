"""
Shared overlap analysis utilities.

This package keeps reusable trace parsing and overlap detection helpers that can
be shared by TP and future CP overlap tests.

TP overlap tuning logic now lives under ``AutoTuner.Profiler.overlap.tp``.
"""

from .overlap_detector import (
    OverlapAnalysis,
    OverlapDetector,
    TimeInterval,
    calculate_overlap_ratio,
    is_overlap_effective,
)
from .trace_analyzer import (
    EventCategory,
    EventType,
    TraceAnalyzer,
    TraceEvent,
    TraceMetadata,
    analyze_trace_file,
)

__all__ = [
    "TraceAnalyzer",
    "TraceEvent",
    "TraceMetadata",
    "EventCategory",
    "EventType",
    "analyze_trace_file",
    "OverlapDetector",
    "OverlapAnalysis",
    "TimeInterval",
    "calculate_overlap_ratio",
    "is_overlap_effective",
]

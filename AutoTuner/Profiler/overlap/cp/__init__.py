"""CP overlap profiling and trace analysis helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "analyze_trace",
    "CPOverlapInputCase",
    "CPOverlapRunner",
    "CPOverlapRunnerConfig",
    "ProfilingResult",
]


def __getattr__(name: str) -> Any:
    if name == "analyze_trace":
        from .cp_overlap_trace_analyzer import analyze_trace

        return analyze_trace
    if name in {
        "CPOverlapInputCase",
        "CPOverlapRunner",
        "CPOverlapRunnerConfig",
        "ProfilingResult",
    }:
        from .runner import (
            CPOverlapInputCase,
            CPOverlapRunner,
            CPOverlapRunnerConfig,
            ProfilingResult,
        )

        return {
            "CPOverlapInputCase": CPOverlapInputCase,
            "CPOverlapRunner": CPOverlapRunner,
            "CPOverlapRunnerConfig": CPOverlapRunnerConfig,
            "ProfilingResult": ProfilingResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

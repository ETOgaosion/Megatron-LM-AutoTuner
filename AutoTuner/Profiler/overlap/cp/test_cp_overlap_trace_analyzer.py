#!/usr/bin/env python3
"""Tests for the CP overlap trace analyzer."""

from __future__ import annotations

import json
from pathlib import Path

from AutoTuner.Profiler.overlap.cp.cp_overlap_trace_analyzer import analyze_trace


def _write_trace(path: Path) -> None:
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "python_function",
                "name": (
                    "TransformerEngine-Enhanced/transformer_engine/pytorch/attention/"
                    "dot_product_attention/context_parallel_nvshmem.py(479): forward"
                ),
                "pid": 1,
                "tid": 1,
                "ts": 100.0,
                "dur": 40.0,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "void flash::flash_fwd_kernel(fake0)",
                "pid": 2,
                "tid": 10,
                "ts": 110.0,
                "dur": 10.0,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "void flash::flash_fwd_kernel(fake1)",
                "pid": 2,
                "tid": 11,
                "ts": 115.0,
                "dur": 10.0,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "void flash::flash_fwd_kernel(ignored0)",
                "pid": 2,
                "tid": 12,
                "ts": 112.0,
                "dur": 2.0,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "void flash::flash_fwd_kernel(ignored1)",
                "pid": 2,
                "tid": 12,
                "ts": 118.0,
                "dur": 2.0,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "gpu_memcpy",
                "name": "Memcpy PtoP (Device -> Device)",
                "pid": 3,
                "tid": 20,
                "ts": 118.0,
                "dur": 4.0,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "gpu_memcpy",
                "name": "Memcpy PtoP (Device -> Device)",
                "pid": 3,
                "tid": 21,
                "ts": 119.0,
                "dur": 1.0,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "gpu_memcpy",
                "name": "Memcpy PtoP (Device -> Device)",
                "pid": 3,
                "tid": 21,
                "ts": 123.0,
                "dur": 1.0,
                "args": {},
            },
        ]
    }
    path.write_text(json.dumps(trace), encoding="utf-8")


def test_analyze_trace_selects_unique_flash_and_p2p_streams(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path)

    result = analyze_trace(str(trace_path), range_index=0)

    assert result["compute_stream_tids"] == [10, 11]
    assert result["comm_stream_tids"] == [20]
    assert result["compute_event_count"] == 2
    assert result["comm_event_count"] == 1
    assert result["compute_time_us"] == 15.0
    assert result["comm_time_us"] == 4.0
    assert result["overlapped_comm_time_us"] == 4.0
    assert abs(result["overlap_ratio"] - (4.0 / 15.0)) < 1e-9

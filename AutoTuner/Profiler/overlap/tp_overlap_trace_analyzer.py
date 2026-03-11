"""
TP overlap trace analyzer for a single micro-batch range.

This module parses a torch profiler JSON trace, selects one run_micro_batch
range, detects computation/communication streams, and computes the overlap
ratio: overlapped_comm_time / total_compute_time.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .trace_analyzer import TraceAnalyzer, TraceEvent

COMPUTE_EXTRA_PATTERNS = [
    r"relu",
    r"gelu",
    r"softmax",
    r"layernorm",
    r"bias",
    r"activation",
]

COMM_EXTRA_PATTERNS = [
    r"allreduce",
    r"allgather",
    r"all_gather",
    r"reduce_scatter",
    r"_reduce_scatter",
    r"rr_rs",
    r"pushsend",
    r"pullrecv",
    r"Memcpy\s+PtoP",
    r"p2p",
]

LINEAR_TYPE_COLUMN = "column"
LINEAR_TYPE_ROW = "row"


@dataclass
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class StreamStats:
    compute_duration: float = 0.0
    comm_duration: float = 0.0
    comm_focus_duration: float = 0.0
    total_duration: float = 0.0
    compute_events: int = 0
    comm_events: int = 0


def _matches_any(patterns: Iterable[str], name: str) -> bool:
    for pattern in patterns:
        if re.search(pattern, name, re.IGNORECASE):
            return True
    return False


def _normalize_linear_type(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"columnparallellinear", "column", "cpl"}:
        return LINEAR_TYPE_COLUMN
    if lowered in {"rowparallellinear", "row", "rpl"}:
        return LINEAR_TYPE_ROW
    raise ValueError("linear_type must be ColumnParallelLinear or RowParallelLinear")


def _comm_focus_patterns(linear_type: str) -> List[str]:
    if linear_type == LINEAR_TYPE_COLUMN:
        return [
            r"allgather",
            r"all_gather",
            r"userbuffers.*ag",
            r"pushsend",
            r"pullrecv",
        ]
    return [
        r"reduce_scatter",
        r"_reduce_scatter",
        r"userbuffers.*rr",
        r"userbuffers.*rs",
        r"rr_rs",
    ]


def _is_compute_event(event: TraceEvent) -> bool:
    patterns = TraceEvent.GEMM_PATTERNS + COMPUTE_EXTRA_PATTERNS
    return _matches_any(patterns, event.name)


def _is_comm_event(event: TraceEvent, linear_type: str) -> bool:
    patterns = TraceEvent.COMM_PATTERNS + COMM_EXTRA_PATTERNS
    if _matches_any(patterns, event.name):
        return True
    return _matches_any(_comm_focus_patterns(linear_type), event.name)


def _is_gpu_event(event: TraceEvent) -> bool:
    return event.category in {"kernel", "gpu_memcpy"}


def _find_run_micro_batch_ranges(events: List[TraceEvent]) -> List[TraceEvent]:
    ranges = []
    for event in events:
        if "run_micro_batch" in event.name:
            if event.duration > 0:
                ranges.append(event)
    return ranges


def _select_range(
    ranges: List[TraceEvent],
    seed: int,
    index: Optional[int],
) -> TraceEvent:
    if not ranges:
        raise ValueError("No run_micro_batch ranges found in trace.")
    if index is not None:
        if index < 0 or index >= len(ranges):
            raise ValueError(
                f"range_index out of bounds: {index} (0..{len(ranges) - 1})"
            )
        return ranges[index]
    rng = random.Random(seed)
    return rng.choice(ranges)


def _clip_interval(
    start: float, end: float, range_start: float, range_end: float
) -> Optional[Interval]:
    clipped_start = max(start, range_start)
    clipped_end = min(end, range_end)
    if clipped_end <= clipped_start:
        return None
    return Interval(start=clipped_start, end=clipped_end)


def _merge_intervals(intervals: List[Interval]) -> List[Interval]:
    if not intervals:
        return []
    intervals_sorted = sorted(intervals, key=lambda i: i.start)
    merged = [intervals_sorted[0]]
    for interval in intervals_sorted[1:]:
        last = merged[-1]
        if interval.start <= last.end:
            merged[-1] = Interval(start=last.start, end=max(last.end, interval.end))
        else:
            merged.append(interval)
    return merged


def _sum_intervals(intervals: List[Interval]) -> float:
    return sum(interval.duration for interval in intervals)


def _intersection_duration(a: List[Interval], b: List[Interval]) -> float:
    if not a or not b:
        return 0.0
    i = 0
    j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        start = max(a[i].start, b[j].start)
        end = min(a[i].end, b[j].end)
        if end > start:
            total += end - start
        if a[i].end < b[j].end:
            i += 1
        else:
            j += 1
    return total


def analyze_trace(
    trace_path: str,
    linear_type: str,
    seed: int = 0,
    range_index: Optional[int] = None,
) -> Dict[str, object]:
    analyzer = TraceAnalyzer(trace_path)
    events = analyzer.parse_trace()

    ranges = _find_run_micro_batch_ranges(events)
    range_event = _select_range(ranges, seed=seed, index=range_index)

    range_start = range_event.timestamp
    range_end = range_event.end_timestamp

    linear_type_norm = _normalize_linear_type(linear_type)

    stream_stats: Dict[int, StreamStats] = {}
    compute_intervals_by_stream: Dict[int, List[Interval]] = {}
    comm_intervals_by_stream: Dict[int, List[Interval]] = {}

    for event in events:
        if not _is_gpu_event(event):
            continue
        if event.end_timestamp <= range_start or event.timestamp >= range_end:
            continue
        clipped = _clip_interval(
            event.timestamp, event.end_timestamp, range_start, range_end
        )
        if clipped is None:
            continue
        tid = event.tid
        stats = stream_stats.setdefault(tid, StreamStats())
        stats.total_duration += clipped.duration

        if _is_compute_event(event):
            compute_intervals_by_stream.setdefault(tid, []).append(clipped)
            stats.compute_duration += clipped.duration
            stats.compute_events += 1
        if _is_comm_event(event, linear_type_norm):
            comm_intervals_by_stream.setdefault(tid, []).append(clipped)
            stats.comm_duration += clipped.duration
            if _matches_any(_comm_focus_patterns(linear_type_norm), event.name):
                stats.comm_focus_duration += clipped.duration
            stats.comm_events += 1

    if not stream_stats:
        raise ValueError("No GPU events found inside selected run_micro_batch range.")

    compute_stream = max(
        stream_stats.items(),
        key=lambda item: (
            item[1].compute_duration,
            item[1].total_duration,
        ),
    )[0]

    comm_stream_candidates = sorted(
        stream_stats.items(),
        key=lambda item: (
            item[1].comm_focus_duration,
            item[1].comm_duration,
            item[1].total_duration,
        ),
        reverse=True,
    )
    comm_stream = comm_stream_candidates[0][0]
    if comm_stream == compute_stream and len(comm_stream_candidates) > 1:
        comm_stream = comm_stream_candidates[1][0]

    compute_intervals = _merge_intervals(
        compute_intervals_by_stream.get(compute_stream, [])
    )
    comm_intervals = _merge_intervals(comm_intervals_by_stream.get(comm_stream, []))

    compute_time = _sum_intervals(compute_intervals)
    comm_time = _sum_intervals(comm_intervals)
    overlapped_comm_time = _intersection_duration(comm_intervals, compute_intervals)
    overlap_ratio = overlapped_comm_time / compute_time if compute_time > 0 else 0.0

    result = {
        "trace_path": trace_path,
        "linear_type": linear_type,
        "range_event_name": range_event.name,
        "range_start_us": range_start,
        "range_end_us": range_end,
        "range_duration_us": range_end - range_start,
        "range_count": len(ranges),
        "selected_range_index": ranges.index(range_event),
        "compute_stream_tid": compute_stream,
        "comm_stream_tid": comm_stream,
        "compute_time_us": compute_time,
        "comm_time_us": comm_time,
        "overlapped_comm_time_us": overlapped_comm_time,
        "overlap_ratio": overlap_ratio,
        "compute_event_count": stream_stats[compute_stream].compute_events,
        "comm_event_count": stream_stats[comm_stream].comm_events,
    }
    return result


def _print_summary(result: Dict[str, object]) -> None:
    print("TP Overlap Trace Analysis")
    print(f"Trace: {result['trace_path']}")
    print(f"Linear type: {result['linear_type']}")
    print(f"Range: {result['range_event_name']}")
    print(
        f"Range time: {result['range_duration_us']:.2f} us "
        f"(index {result['selected_range_index']}/{result['range_count']})"
    )
    print(f"Compute stream: {result['compute_stream_tid']}")
    print(f"Comm stream: {result['comm_stream_tid']}")
    print(f"Compute time: {result['compute_time_us']:.2f} us")
    print(f"Comm time: {result['comm_time_us']:.2f} us")
    print(f"Overlapped comm time: {result['overlapped_comm_time_us']:.2f} us")
    print(f"Overlap ratio: {result['overlap_ratio']:.4f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze TP overlap in a random run_micro_batch range."
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to a torch profiler JSON trace file.",
    )
    parser.add_argument(
        "--linear-type",
        required=True,
        help="ColumnParallelLinear or RowParallelLinear.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for range selection.",
    )
    parser.add_argument(
        "--range-index",
        type=int,
        default=None,
        help="Explicit run_micro_batch range index (overrides random).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON only (no human-readable summary).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write JSON output to this file.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = analyze_trace(
        trace_path=args.trace,
        linear_type=args.linear_type,
        seed=args.seed,
        range_index=args.range_index,
    )
    if not args.json:
        _print_summary(result)
        print("")
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(output)
            f.write("\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

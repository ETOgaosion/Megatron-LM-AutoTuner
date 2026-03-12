"""
CP overlap trace analyzer for a single context-parallel forward range.

This module parses a torch profiler JSON trace, selects one
`context_parallel_nvshmem.py(...): forward` Python range, finds the compute and
communication streams requested by the CP overlap requirement, and computes the
overlap ratio:

    overlap_ratio = overlapped_comm_time / total_compute_time
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from ..trace_analyzer import TraceAnalyzer, TraceEvent

DEFAULT_FORWARD_PATTERN = (
    "TransformerEngine-Enhanced/transformer_engine/pytorch/attention/"
    "dot_product_attention/context_parallel_nvshmem.py(479): forward"
)
COMPUTE_KERNEL_PATTERN = "flash::flash_fwd_kernel"
COMM_KERNEL_NAME = "Memcpy PtoP (Device -> Device)"


@dataclass
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _find_forward_ranges(
    events: Iterable[TraceEvent], forward_pattern: str
) -> List[TraceEvent]:
    return [
        event
        for event in events
        if event.category == "python_function"
        and forward_pattern in event.name
        and event.duration > 0
    ]


def _select_range(
    ranges: List[TraceEvent], seed: int, index: Optional[int]
) -> TraceEvent:
    if not ranges:
        raise ValueError("No CP forward ranges found in trace.")
    if index is not None:
        if index < 0 or index >= len(ranges):
            raise ValueError(f"range_index out of bounds: {index} (0..{len(ranges) - 1})")
        return ranges[index]
    rng = random.Random(seed)
    return rng.choice(ranges)


def _is_gpu_event(event: TraceEvent) -> bool:
    return event.category in {"kernel", "gpu_memcpy"}


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
    intervals = sorted(intervals, key=lambda item: item.start)
    merged = [intervals[0]]
    for interval in intervals[1:]:
        previous = merged[-1]
        if interval.start <= previous.end:
            merged[-1] = Interval(previous.start, max(previous.end, interval.end))
        else:
            merged.append(interval)
    return merged


def _sum_intervals(intervals: List[Interval]) -> float:
    return sum(interval.duration for interval in intervals)


def _intersection_duration(a: List[Interval], b: List[Interval]) -> float:
    if not a or not b:
        return 0.0
    total = 0.0
    a_idx = 0
    b_idx = 0
    while a_idx < len(a) and b_idx < len(b):
        start = max(a[a_idx].start, b[b_idx].start)
        end = min(a[a_idx].end, b[b_idx].end)
        if end > start:
            total += end - start
        if a[a_idx].end < b[b_idx].end:
            a_idx += 1
        else:
            b_idx += 1
    return total


def analyze_trace(
    trace_path: str,
    seed: int = 0,
    range_index: Optional[int] = None,
    forward_pattern: str = DEFAULT_FORWARD_PATTERN,
) -> Dict[str, object]:
    analyzer = TraceAnalyzer(trace_path)
    events = analyzer.parse_trace()

    ranges = _find_forward_ranges(events, forward_pattern=forward_pattern)
    range_event = _select_range(ranges, seed=seed, index=range_index)
    range_start = range_event.timestamp
    range_end = range_event.end_timestamp

    gpu_events: List[TraceEvent] = []
    compute_counts: Counter[int] = Counter()
    comm_counts: Counter[int] = Counter()

    for event in events:
        if not _is_gpu_event(event):
            continue
        if event.end_timestamp <= range_start or event.timestamp >= range_end:
            continue
        gpu_events.append(event)
        if COMPUTE_KERNEL_PATTERN in event.name:
            compute_counts[event.tid] += 1
        if event.name == COMM_KERNEL_NAME:
            comm_counts[event.tid] += 1

    compute_streams = sorted(
        tid for tid, count in compute_counts.items() if count == 1
    )
    comm_streams = sorted(tid for tid, count in comm_counts.items() if count == 1)

    if not compute_streams:
        raise ValueError("No compute streams found in the selected CP forward range.")
    if not comm_streams:
        raise ValueError("No communication streams found in the selected CP forward range.")

    compute_intervals: List[Interval] = []
    comm_intervals: List[Interval] = []
    compute_event_count = 0
    comm_event_count = 0

    for event in gpu_events:
        clipped = _clip_interval(
            event.timestamp, event.end_timestamp, range_start, range_end
        )
        if clipped is None:
            continue
        if (
            COMPUTE_KERNEL_PATTERN in event.name
            and event.tid in compute_streams
            and compute_counts[event.tid] == 1
        ):
            compute_intervals.append(clipped)
            compute_event_count += 1
        if (
            event.name == COMM_KERNEL_NAME
            and event.tid in comm_streams
            and comm_counts[event.tid] == 1
        ):
            comm_intervals.append(clipped)
            comm_event_count += 1

    merged_compute = _merge_intervals(compute_intervals)
    merged_comm = _merge_intervals(comm_intervals)

    compute_time = _sum_intervals(merged_compute)
    comm_time = _sum_intervals(merged_comm)
    overlapped_comm_time = _intersection_duration(merged_compute, merged_comm)
    overlap_ratio = overlapped_comm_time / compute_time if compute_time > 0 else 0.0

    return {
        "trace_path": trace_path,
        "forward_pattern": forward_pattern,
        "range_event_name": range_event.name,
        "range_start_us": range_start,
        "range_end_us": range_end,
        "range_duration_us": range_end - range_start,
        "range_count": len(ranges),
        "selected_range_index": ranges.index(range_event),
        "compute_stream_tids": compute_streams,
        "comm_stream_tids": comm_streams,
        "compute_time_us": compute_time,
        "comm_time_us": comm_time,
        "overlapped_comm_time_us": overlapped_comm_time,
        "overlap_ratio": overlap_ratio,
        "compute_event_count": compute_event_count,
        "comm_event_count": comm_event_count,
    }


def _print_summary(result: Dict[str, object]) -> None:
    print("CP Overlap Trace Analysis")
    print(f"Trace: {result['trace_path']}")
    print(f"Forward range: {result['range_event_name']}")
    print(
        f"Range time: {result['range_duration_us']:.2f} us "
        f"(index {result['selected_range_index']}/{result['range_count']})"
    )
    print(f"Compute streams: {result['compute_stream_tids']}")
    print(f"Comm streams: {result['comm_stream_tids']}")
    print(f"Compute time: {result['compute_time_us']:.2f} us")
    print(f"Comm time: {result['comm_time_us']:.2f} us")
    print(f"Overlapped comm time: {result['overlapped_comm_time_us']:.2f} us")
    print(f"Overlap ratio: {result['overlap_ratio']:.4f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze CP overlap in a random or selected forward range."
    )
    parser.add_argument("--trace", required=True, help="Path to a torch profiler trace.")
    parser.add_argument(
        "--forward-pattern",
        default=DEFAULT_FORWARD_PATTERN,
        help="Python-function range name substring used to select CP forward ranges.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for range selection."
    )
    parser.add_argument(
        "--range-index",
        type=int,
        default=None,
        help="Explicit CP forward range index (overrides random selection).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print JSON only without the summary."
    )
    parser.add_argument(
        "--output-json", default=None, help="Write JSON output to this file."
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = analyze_trace(
        trace_path=args.trace,
        seed=args.seed,
        range_index=args.range_index,
        forward_pattern=args.forward_pattern,
    )
    if not args.json:
        _print_summary(result)
        print("")
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        with open(args.output_json, "w") as handle:
            handle.write(output)
            handle.write("\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Overlap Detector for TP Communication/Computation.

This module detects and quantifies the overlap between compute (GEMM) operations
and communication operations from torch profiler traces.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config_generator import TPOverlapTestConfig
from .trace_analyzer import TraceAnalyzer, TraceEvent


@dataclass
class TimeInterval:
    """Represents a time interval."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        """Get the duration of this interval."""
        return max(0, self.end - self.start)

    def overlaps(self, other: "TimeInterval") -> bool:
        """Check if this interval overlaps with another."""
        return self.start < other.end and other.start < self.end

    def intersection(self, other: "TimeInterval") -> Optional["TimeInterval"]:
        """Get the intersection of two intervals."""
        if not self.overlaps(other):
            return None
        return TimeInterval(
            start=max(self.start, other.start), end=min(self.end, other.end)
        )

    def union_duration(self, other: "TimeInterval") -> float:
        """Get the total duration covered by the union of two intervals."""
        if not self.overlaps(other):
            return self.duration + other.duration
        return max(self.end, other.end) - min(self.start, other.start)


@dataclass
class OverlapAnalysis:
    """Results of overlap analysis for a configuration."""

    config: TPOverlapTestConfig
    # Forward pass metrics
    forward_gemm_time: float = 0.0
    forward_comm_time: float = 0.0
    forward_overlap_time: float = 0.0
    # Backward pass metrics
    backward_gemm_time: float = 0.0
    backward_comm_time: float = 0.0
    backward_overlap_time: float = 0.0
    # Total metrics
    total_time: float = 0.0
    total_gemm_time: float = 0.0
    total_comm_time: float = 0.0
    total_overlap_time: float = 0.0
    # Raw event counts
    num_gemm_events: int = 0
    num_comm_events: int = 0
    # End-to-end execution time for the Linear operator
    # This is the wall clock time from first event to last event
    forward_e2e_time: float = 0.0  # Forward pass end-to-end time
    backward_e2e_time: float = 0.0  # Backward pass end-to-end time
    operator_e2e_time: float = 0.0  # Total operator end-to-end time

    @property
    def forward_overlap_ratio(self) -> float:
        """Calculate forward overlap ratio: overlap_time / min(gemm, comm)."""
        min_time = min(self.forward_gemm_time, self.forward_comm_time)
        if min_time <= 0:
            return 0.0
        return self.forward_overlap_time / min_time

    @property
    def backward_overlap_ratio(self) -> float:
        """Calculate backward overlap ratio: overlap_time / min(gemm, comm)."""
        min_time = min(self.backward_gemm_time, self.backward_comm_time)
        if min_time <= 0:
            return 0.0
        return self.backward_overlap_time / min_time

    @property
    def total_overlap_ratio(self) -> float:
        """Calculate total overlap ratio: overlap_time / min(gemm, comm)."""
        min_time = min(self.total_gemm_time, self.total_comm_time)
        if min_time <= 0:
            return 0.0
        return self.total_overlap_time / min_time

    @property
    def overlap_efficiency(self) -> float:
        """Calculate overlap efficiency: overlap_time / (gemm + comm - overlap)."""
        denominator = (
            self.total_gemm_time + self.total_comm_time - self.total_overlap_time
        )
        if denominator <= 0:
            return 0.0
        return self.total_overlap_time / denominator

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_id": self.config.get_test_id(),
            "operator": self.config.operator,
            "phase": self.config.phase,
            "tp_size": self.config.tp_size,
            "overlap_method": self.config.overlap_method.value,
            # Forward pass metrics
            "forward_gemm_time_us": self.forward_gemm_time,
            "forward_comm_time_us": self.forward_comm_time,
            "forward_overlap_time_us": self.forward_overlap_time,
            "forward_overlap_ratio": self.forward_overlap_ratio,
            "forward_e2e_time_us": self.forward_e2e_time,
            # Backward pass metrics
            "backward_gemm_time_us": self.backward_gemm_time,
            "backward_comm_time_us": self.backward_comm_time,
            "backward_overlap_time_us": self.backward_overlap_time,
            "backward_overlap_ratio": self.backward_overlap_ratio,
            "backward_e2e_time_us": self.backward_e2e_time,
            # Total metrics
            "total_time_us": self.total_time,
            "total_gemm_time_us": self.total_gemm_time,
            "total_comm_time_us": self.total_comm_time,
            "total_overlap_time_us": self.total_overlap_time,
            "total_overlap_ratio": self.total_overlap_ratio,
            "overlap_efficiency": self.overlap_efficiency,
            # End-to-end execution time
            "operator_e2e_time_us": self.operator_e2e_time,
            # Event counts
            "num_gemm_events": self.num_gemm_events,
            "num_comm_events": self.num_comm_events,
        }


class OverlapDetector:
    """Detects and quantifies compute/communication overlap from traces."""

    def __init__(self):
        """Initialize the overlap detector."""
        pass

    def analyze_overlap(
        self, trace_path: str, config: TPOverlapTestConfig
    ) -> OverlapAnalysis:
        """Analyze a trace file and compute overlap metrics.

        Args:
            trace_path: Path to the torch profiler JSON trace file.
            config: The test configuration used to generate this trace.

        Returns:
            OverlapAnalysis with computed metrics.
        """
        analyzer = TraceAnalyzer(trace_path)
        analyzer.parse_trace()

        # Get GEMM and communication kernel events
        gemm_kernels = analyzer.extract_gemm_kernels()
        comm_kernels = analyzer.extract_comm_kernels()

        # Calculate overlap
        analysis = OverlapAnalysis(config=config)
        analysis.num_gemm_events = len(gemm_kernels)
        analysis.num_comm_events = len(comm_kernels)

        # Calculate total times
        analysis.total_gemm_time = sum(e.duration for e in gemm_kernels)
        analysis.total_comm_time = sum(e.duration for e in comm_kernels)

        # Calculate overlap time
        analysis.total_overlap_time = self._calculate_overlap_time(
            gemm_kernels, comm_kernels
        )

        # Try to split forward/backward (using NVTX markers if available)
        forward_gemm, backward_gemm = self._split_forward_backward(
            gemm_kernels, analyzer.events
        )
        forward_comm, backward_comm = self._split_forward_backward(
            comm_kernels, analyzer.events
        )

        # Calculate forward metrics
        analysis.forward_gemm_time = sum(e.duration for e in forward_gemm)
        analysis.forward_comm_time = sum(e.duration for e in forward_comm)
        analysis.forward_overlap_time = self._calculate_overlap_time(
            forward_gemm, forward_comm
        )

        # Calculate backward metrics
        analysis.backward_gemm_time = sum(e.duration for e in backward_gemm)
        analysis.backward_comm_time = sum(e.duration for e in backward_comm)
        analysis.backward_overlap_time = self._calculate_overlap_time(
            backward_gemm, backward_comm
        )

        # Calculate total time (wall clock)
        time_range = analyzer.get_time_range()
        analysis.total_time = time_range[1] - time_range[0]

        # Calculate end-to-end execution time for the Linear operator
        # E2E time = wall clock from first event to last event (GEMM + comm combined)
        all_operator_events = gemm_kernels + comm_kernels
        analysis.operator_e2e_time = self._calculate_e2e_time(all_operator_events)

        # Forward e2e time
        forward_events = forward_gemm + forward_comm
        analysis.forward_e2e_time = self._calculate_e2e_time(forward_events)

        # Backward e2e time
        backward_events = backward_gemm + backward_comm
        analysis.backward_e2e_time = self._calculate_e2e_time(backward_events)

        return analysis

    def _calculate_e2e_time(self, events: List[TraceEvent]) -> float:
        """Calculate end-to-end execution time from a list of events.

        This is the wall clock time from the start of the first event
        to the end of the last event.
        """
        if not events:
            return 0.0

        min_start = min(e.timestamp for e in events)
        max_end = max(e.end_timestamp for e in events)
        return max_end - min_start

    def _calculate_overlap_time(
        self, events1: List[TraceEvent], events2: List[TraceEvent]
    ) -> float:
        """Calculate the total overlap time between two sets of events.

        Uses interval intersection to find overlapping time periods.
        """
        if not events1 or not events2:
            return 0.0

        # Convert events to intervals
        intervals1 = [
            TimeInterval(start=e.timestamp, end=e.end_timestamp) for e in events1
        ]
        intervals2 = [
            TimeInterval(start=e.timestamp, end=e.end_timestamp) for e in events2
        ]

        # Merge overlapping intervals within each set
        merged1 = self._merge_intervals(intervals1)
        merged2 = self._merge_intervals(intervals2)

        # Calculate intersection of merged intervals
        total_overlap = 0.0
        for int1 in merged1:
            for int2 in merged2:
                intersection = int1.intersection(int2)
                if intersection:
                    total_overlap += intersection.duration

        return total_overlap

    def _merge_intervals(self, intervals: List[TimeInterval]) -> List[TimeInterval]:
        """Merge overlapping intervals into non-overlapping intervals."""
        if not intervals:
            return []

        # Sort by start time
        sorted_intervals = sorted(intervals, key=lambda x: x.start)

        merged = [sorted_intervals[0]]
        for interval in sorted_intervals[1:]:
            if interval.start <= merged[-1].end:
                # Overlapping, extend the last interval
                merged[-1] = TimeInterval(
                    start=merged[-1].start, end=max(merged[-1].end, interval.end)
                )
            else:
                # Non-overlapping, add new interval
                merged.append(interval)

        return merged

    def _split_forward_backward(
        self, events: List[TraceEvent], all_events: List[TraceEvent]
    ) -> Tuple[List[TraceEvent], List[TraceEvent]]:
        """Split events into forward and backward pass based on NVTX markers.

        Looks for "forward" and "backward" NVTX range markers to determine
        the boundary between forward and backward pass.
        """
        # Find NVTX markers for forward/backward
        forward_marker = None
        backward_marker = None

        for event in all_events:
            name_lower = event.name.lower()
            if "forward" in name_lower and event.category in [
                "user_annotation",
                "cpu_op",
            ]:
                if forward_marker is None or event.timestamp < forward_marker.timestamp:
                    forward_marker = event
            if "backward" in name_lower and event.category in [
                "user_annotation",
                "cpu_op",
            ]:
                if (
                    backward_marker is None
                    or event.timestamp < backward_marker.timestamp
                ):
                    backward_marker = event

        if backward_marker is None:
            # No backward marker found, use heuristic: first half is forward
            if not events:
                return [], []
            sorted_events = sorted(events, key=lambda e: e.timestamp)
            mid_time = (
                sorted_events[0].timestamp + sorted_events[-1].end_timestamp
            ) / 2
            forward = [e for e in events if e.end_timestamp <= mid_time]
            backward = [e for e in events if e.timestamp >= mid_time]
            return forward, backward

        # Split based on backward marker timestamp
        backward_start = backward_marker.timestamp
        forward = [e for e in events if e.end_timestamp <= backward_start]
        backward = [e for e in events if e.timestamp >= backward_start]

        return forward, backward

    def analyze_multiple_traces(
        self, trace_configs: List[Tuple[str, TPOverlapTestConfig]]
    ) -> List[OverlapAnalysis]:
        """Analyze multiple trace files.

        Args:
            trace_configs: List of (trace_path, config) tuples.

        Returns:
            List of OverlapAnalysis results.
        """
        results = []
        for trace_path, config in trace_configs:
            try:
                analysis = self.analyze_overlap(trace_path, config)
                results.append(analysis)
            except Exception as e:
                print(f"Error analyzing trace {trace_path}: {e}")
                # Create empty analysis for failed traces
                analysis = OverlapAnalysis(config=config)
                results.append(analysis)
        return results


def calculate_overlap_ratio(
    gemm_time: float, comm_time: float, overlap_time: float
) -> float:
    """Calculate overlap ratio: overlap / min(gemm, comm)."""
    min_time = min(gemm_time, comm_time)
    if min_time <= 0:
        return 0.0
    return overlap_time / min_time


def is_overlap_effective(analysis: OverlapAnalysis, threshold: float = 0.5) -> bool:
    """Check if overlap is effective (ratio >= threshold)."""
    return analysis.total_overlap_ratio >= threshold

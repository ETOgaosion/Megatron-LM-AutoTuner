"""
Trace Analyzer for Torch Profiler JSON Traces.

This module parses torch profiler JSON trace files and extracts
relevant events for GEMM operations and communication operations.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class EventCategory(Enum):
    """Categories of trace events."""

    KERNEL = "kernel"
    CPU_OP = "cpu_op"
    USER_ANNOTATION = "user_annotation"
    PYTHON_FUNCTION = "python_function"
    GPU_MEMCPY = "gpu_memcpy"
    OTHER = "other"


class EventType(Enum):
    """Types of events we care about."""

    GEMM = "gemm"
    COMMUNICATION = "communication"
    OTHER = "other"


@dataclass
class TraceEvent:
    """Represents a single trace event from torch profiler."""

    name: str
    category: str
    timestamp: float  # in microseconds
    duration: float  # in microseconds
    pid: int
    tid: int
    args: Dict[str, Any] = field(default_factory=dict)

    # GEMM detection patterns
    GEMM_PATTERNS = [
        r"cutlass",
        r"gemm",
        r"matmul",
        r"tgemm",
        r"cublas",
        r"sm\d+_xmma",  # CUDA tensor core operations
        r"generic_gemm",
    ]

    # Communication detection patterns
    COMM_PATTERNS = [
        r"userbuffers",
        r"allgather",
        r"all_gather",
        r"reduce_scatter",
        r"_reduce_scatter",
        r"ncclDevKernel",
        r"nccl:",
        r"Memcpy\s+PtoP",
        r"c10d::",
        r"kuserbuffers",
    ]

    def is_gemm(self) -> bool:
        """Check if this event is a GEMM operation."""
        name_lower = self.name.lower()
        for pattern in self.GEMM_PATTERNS:
            if re.search(pattern, name_lower, re.IGNORECASE):
                return True
        return False

    def is_communication(self) -> bool:
        """Check if this event is a communication operation."""
        name_lower = self.name.lower()
        for pattern in self.COMM_PATTERNS:
            if re.search(pattern, name_lower, re.IGNORECASE):
                return True
        # Also check category for NCCL annotations
        if self.category == "user_annotation" and "nccl" in name_lower:
            return True
        return False

    def is_kernel(self) -> bool:
        """Check if this event is a GPU kernel."""
        return self.category == "kernel"

    def get_event_type(self) -> EventType:
        """Determine the type of this event."""
        if self.is_gemm():
            return EventType.GEMM
        if self.is_communication():
            return EventType.COMMUNICATION
        return EventType.OTHER

    @property
    def end_timestamp(self) -> float:
        """Get the end timestamp of this event."""
        return self.timestamp + self.duration


@dataclass
class TraceMetadata:
    """Metadata from the trace file."""

    schema_version: int = 1
    device_properties: List[Dict] = field(default_factory=list)
    distributed_info: Dict = field(default_factory=dict)
    cuda_runtime_version: int = 0
    base_time_ns: int = 0


class TraceAnalyzer:
    """Analyzer for torch profiler JSON trace files."""

    def __init__(self, trace_path: str):
        """Initialize the analyzer with a trace file path."""
        self.trace_path = trace_path
        self.events: List[TraceEvent] = []
        self.metadata: Optional[TraceMetadata] = None
        self._raw_data: Optional[Dict] = None

    def parse_trace(self) -> List[TraceEvent]:
        """Parse the trace file and return all events."""
        with open(self.trace_path, "r") as f:
            self._raw_data = json.load(f)

        # Parse metadata
        self.metadata = self._parse_metadata()

        # Parse events
        trace_events = self._raw_data.get("traceEvents", [])
        self.events = []

        for event in trace_events:
            parsed_event = self._parse_event(event)
            if parsed_event is not None:
                self.events.append(parsed_event)

        return self.events

    def _parse_metadata(self) -> TraceMetadata:
        """Parse trace metadata."""
        metadata = TraceMetadata()
        if self._raw_data is None:
            return metadata

        metadata.schema_version = self._raw_data.get("schemaVersion", 1)
        metadata.device_properties = self._raw_data.get("deviceProperties", [])
        metadata.distributed_info = self._raw_data.get("distributedInfo", {})
        metadata.cuda_runtime_version = self._raw_data.get("cuda_runtime_version", 0)
        metadata.base_time_ns = self._raw_data.get("baseTimeNanoseconds", 0)

        return metadata

    def _parse_event(self, event: Dict) -> Optional[TraceEvent]:
        """Parse a single event from the trace."""
        # Only process complete events (ph == "X")
        if event.get("ph") != "X":
            return None

        name = event.get("name", "")
        category = event.get("cat", "")
        timestamp = event.get("ts", 0)
        duration = event.get("dur", 0)
        pid = event.get("pid", 0)
        tid = event.get("tid", 0)
        args = event.get("args", {})

        return TraceEvent(
            name=name,
            category=category,
            timestamp=timestamp,
            duration=duration,
            pid=pid,
            tid=tid,
            args=args,
        )

    def extract_gemm_events(self) -> List[TraceEvent]:
        """Extract all GEMM-related events."""
        if not self.events:
            self.parse_trace()
        return [e for e in self.events if e.is_gemm()]

    def extract_comm_events(self) -> List[TraceEvent]:
        """Extract all communication-related events."""
        if not self.events:
            self.parse_trace()
        return [e for e in self.events if e.is_communication()]

    def extract_kernel_events(self) -> List[TraceEvent]:
        """Extract all GPU kernel events."""
        if not self.events:
            self.parse_trace()
        return [e for e in self.events if e.is_kernel()]

    def extract_gemm_kernels(self) -> List[TraceEvent]:
        """Extract GEMM kernels (GPU kernel events that are GEMM operations)."""
        if not self.events:
            self.parse_trace()
        return [e for e in self.events if e.is_kernel() and e.is_gemm()]

    def extract_comm_kernels(self) -> List[TraceEvent]:
        """Extract communication kernels (GPU kernel events that are comm operations)."""
        if not self.events:
            self.parse_trace()
        return [e for e in self.events if e.is_kernel() and e.is_communication()]

    def get_events_by_stream(self) -> Dict[int, List[TraceEvent]]:
        """Group kernel events by their stream (tid for GPU events)."""
        if not self.events:
            self.parse_trace()

        stream_events: Dict[int, List[TraceEvent]] = {}
        for event in self.events:
            if event.is_kernel():
                tid = event.tid
                if tid not in stream_events:
                    stream_events[tid] = []
                stream_events[tid].append(event)

        # Sort events by timestamp within each stream
        for tid in stream_events:
            stream_events[tid].sort(key=lambda e: e.timestamp)

        return stream_events

    def get_time_range(self) -> Tuple[float, float]:
        """Get the time range of all events."""
        if not self.events:
            self.parse_trace()

        if not self.events:
            return (0, 0)

        min_ts = min(e.timestamp for e in self.events)
        max_ts = max(e.end_timestamp for e in self.events)
        return (min_ts, max_ts)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the trace."""
        if not self.events:
            self.parse_trace()

        gemm_events = self.extract_gemm_events()
        comm_events = self.extract_comm_events()
        kernel_events = self.extract_kernel_events()
        gemm_kernels = self.extract_gemm_kernels()
        comm_kernels = self.extract_comm_kernels()

        time_range = self.get_time_range()

        return {
            "total_events": len(self.events),
            "kernel_events": len(kernel_events),
            "gemm_events": len(gemm_events),
            "gemm_kernels": len(gemm_kernels),
            "comm_events": len(comm_events),
            "comm_kernels": len(comm_kernels),
            "time_range_us": time_range,
            "duration_us": time_range[1] - time_range[0],
            "num_streams": len(self.get_events_by_stream()),
            "device_properties": self.metadata.device_properties if self.metadata else [],
            "distributed_info": self.metadata.distributed_info if self.metadata else {},
        }


def analyze_trace_file(trace_path: str) -> Dict[str, Any]:
    """Convenience function to analyze a trace file and return summary."""
    analyzer = TraceAnalyzer(trace_path)
    analyzer.parse_trace()
    return analyzer.get_summary()

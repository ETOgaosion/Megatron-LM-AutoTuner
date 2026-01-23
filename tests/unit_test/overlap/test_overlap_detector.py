"""
Unit tests for AutoTuner.Profiler.overlap.overlap_detector module.

These tests use mock trace JSON files to test overlap detection logic.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from AutoTuner.Profiler.overlap.config_generator import (
    OverlapMethod,
    TPOverlapTestConfig,
)
from AutoTuner.Profiler.overlap.overlap_detector import (
    OverlapAnalysis,
    OverlapDetector,
    TimeInterval,
    calculate_overlap_ratio,
    is_overlap_effective,
)


def create_mock_trace_json(events: list, output_path: str) -> None:
    """Create a mock torch profiler trace JSON file.

    Args:
        events: List of event dicts with name, cat, ts, dur, pid, tid.
        output_path: Path to write the JSON file.
    """
    trace_data = {
        "schemaVersion": 1,
        "deviceProperties": [
            {
                "id": 0,
                "name": "Mock GPU",
                "totalGlobalMem": 16000000000,
                "computeMajor": 8,
                "computeMinor": 0,
                "numSms": 108,
            }
        ],
        "traceEvents": [
            {
                "ph": "X",
                "cat": e.get("cat", "kernel"),
                "name": e["name"],
                "pid": e.get("pid", 0),
                "tid": e.get("tid", 0),
                "ts": e["ts"],
                "dur": e["dur"],
                "args": e.get("args", {}),
            }
            for e in events
        ],
    }
    with open(output_path, "w") as f:
        json.dump(trace_data, f)


class TestTimeInterval(unittest.TestCase):
    """Tests for TimeInterval dataclass."""

    def test_duration(self):
        """Test duration calculation."""
        interval = TimeInterval(start=100, end=200)
        self.assertEqual(interval.duration, 100)

    def test_duration_zero(self):
        """Test zero duration when start equals end."""
        interval = TimeInterval(start=100, end=100)
        self.assertEqual(interval.duration, 0)

    def test_duration_negative_clamped(self):
        """Test that negative duration is clamped to 0."""
        interval = TimeInterval(start=200, end=100)
        self.assertEqual(interval.duration, 0)

    def test_overlaps_true(self):
        """Test overlapping intervals."""
        interval1 = TimeInterval(start=100, end=200)
        interval2 = TimeInterval(start=150, end=250)
        self.assertTrue(interval1.overlaps(interval2))
        self.assertTrue(interval2.overlaps(interval1))

    def test_overlaps_false_no_overlap(self):
        """Test non-overlapping intervals."""
        interval1 = TimeInterval(start=100, end=200)
        interval2 = TimeInterval(start=300, end=400)
        self.assertFalse(interval1.overlaps(interval2))
        self.assertFalse(interval2.overlaps(interval1))

    def test_overlaps_false_touching(self):
        """Test touching intervals (not overlapping)."""
        interval1 = TimeInterval(start=100, end=200)
        interval2 = TimeInterval(start=200, end=300)
        self.assertFalse(interval1.overlaps(interval2))

    def test_intersection(self):
        """Test intersection of overlapping intervals."""
        interval1 = TimeInterval(start=100, end=200)
        interval2 = TimeInterval(start=150, end=250)
        intersection = interval1.intersection(interval2)

        self.assertIsNotNone(intersection)
        self.assertEqual(intersection.start, 150)
        self.assertEqual(intersection.end, 200)
        self.assertEqual(intersection.duration, 50)

    def test_intersection_none(self):
        """Test intersection of non-overlapping intervals."""
        interval1 = TimeInterval(start=100, end=200)
        interval2 = TimeInterval(start=300, end=400)
        intersection = interval1.intersection(interval2)

        self.assertIsNone(intersection)

    def test_union_duration_overlapping(self):
        """Test union duration for overlapping intervals."""
        interval1 = TimeInterval(start=100, end=200)
        interval2 = TimeInterval(start=150, end=250)
        union_dur = interval1.union_duration(interval2)

        self.assertEqual(union_dur, 150)  # 250 - 100

    def test_union_duration_non_overlapping(self):
        """Test union duration for non-overlapping intervals."""
        interval1 = TimeInterval(start=100, end=200)
        interval2 = TimeInterval(start=300, end=400)
        union_dur = interval1.union_duration(interval2)

        self.assertEqual(union_dur, 200)  # 100 + 100


class TestOverlapAnalysis(unittest.TestCase):
    """Tests for OverlapAnalysis dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = TPOverlapTestConfig(
            tp_size=2,
            operator="fc1",
            phase="fprop",
            overlap_method=OverlapMethod.RING_EXCHANGE,
        )

    def test_forward_overlap_ratio(self):
        """Test forward overlap ratio calculation."""
        analysis = OverlapAnalysis(
            config=self.mock_config,
            forward_gemm_time=1000,
            forward_comm_time=500,
            forward_overlap_time=400,
        )
        # overlap_ratio = overlap_time / min(gemm, comm) = 400 / 500 = 0.8
        self.assertAlmostEqual(analysis.forward_overlap_ratio, 0.8)

    def test_forward_overlap_ratio_zero_time(self):
        """Test forward overlap ratio with zero time."""
        analysis = OverlapAnalysis(
            config=self.mock_config,
            forward_gemm_time=0,
            forward_comm_time=0,
            forward_overlap_time=0,
        )
        self.assertEqual(analysis.forward_overlap_ratio, 0.0)

    def test_backward_overlap_ratio(self):
        """Test backward overlap ratio calculation."""
        analysis = OverlapAnalysis(
            config=self.mock_config,
            backward_gemm_time=2000,
            backward_comm_time=800,
            backward_overlap_time=600,
        )
        # overlap_ratio = overlap_time / min(gemm, comm) = 600 / 800 = 0.75
        self.assertAlmostEqual(analysis.backward_overlap_ratio, 0.75)

    def test_total_overlap_ratio(self):
        """Test total overlap ratio calculation."""
        analysis = OverlapAnalysis(
            config=self.mock_config,
            total_gemm_time=3000,
            total_comm_time=1000,
            total_overlap_time=800,
        )
        # overlap_ratio = overlap_time / min(gemm, comm) = 800 / 1000 = 0.8
        self.assertAlmostEqual(analysis.total_overlap_ratio, 0.8)

    def test_overlap_efficiency(self):
        """Test overlap efficiency calculation."""
        analysis = OverlapAnalysis(
            config=self.mock_config,
            total_gemm_time=1000,
            total_comm_time=500,
            total_overlap_time=400,
        )
        # efficiency = overlap / (gemm + comm - overlap) = 400 / (1000 + 500 - 400) = 400/1100
        self.assertAlmostEqual(analysis.overlap_efficiency, 400 / 1100)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        analysis = OverlapAnalysis(
            config=self.mock_config,
            forward_gemm_time=1000,
            forward_comm_time=500,
            forward_overlap_time=400,
            num_gemm_events=4,
            num_comm_events=2,
        )
        result = analysis.to_dict()

        self.assertEqual(result["config_id"], "tp2_fc1_fprop_ring_agg0")
        self.assertEqual(result["operator"], "fc1")
        self.assertEqual(result["phase"], "fprop")
        self.assertEqual(result["tp_size"], 2)
        self.assertEqual(result["forward_gemm_time_us"], 1000)
        self.assertEqual(result["num_gemm_events"], 4)
        self.assertEqual(result["num_comm_events"], 2)


class TestOverlapDetector(unittest.TestCase):
    """Tests for OverlapDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = OverlapDetector()
        self.mock_config = TPOverlapTestConfig(
            tp_size=2,
            operator="fc1",
            phase="fprop",
            overlap_method=OverlapMethod.RING_EXCHANGE,
        )
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_analyze_overlap_with_full_overlap(self):
        """Test overlap detection when GEMM and comm fully overlap."""
        # Create mock trace with overlapping GEMM and comm events
        # GEMM: 0-1000us, Comm: 200-800us (fully inside GEMM)
        events = [
            {"name": "cutlass_gemm_kernel", "cat": "kernel", "ts": 0, "dur": 1000},
            {"name": "userbuffers_allgather", "cat": "kernel", "ts": 200, "dur": 600},
        ]
        trace_path = os.path.join(self.temp_dir, "full_overlap.json")
        create_mock_trace_json(events, trace_path)

        analysis = self.detector.analyze_overlap(trace_path, self.mock_config)

        self.assertEqual(analysis.num_gemm_events, 1)
        self.assertEqual(analysis.num_comm_events, 1)
        self.assertEqual(analysis.total_gemm_time, 1000)
        self.assertEqual(analysis.total_comm_time, 600)
        # Overlap should be 600us (entire comm duration)
        self.assertEqual(analysis.total_overlap_time, 600)
        # Overlap ratio = 600 / min(1000, 600) = 600 / 600 = 1.0
        self.assertAlmostEqual(analysis.total_overlap_ratio, 1.0)

    def test_analyze_overlap_with_partial_overlap(self):
        """Test overlap detection when GEMM and comm partially overlap."""
        # GEMM: 0-1000us, Comm: 800-1500us (200us overlap)
        events = [
            {"name": "cutlass_gemm_kernel", "cat": "kernel", "ts": 0, "dur": 1000},
            {
                "name": "userbuffers_reduce_scatter",
                "cat": "kernel",
                "ts": 800,
                "dur": 700,
            },
        ]
        trace_path = os.path.join(self.temp_dir, "partial_overlap.json")
        create_mock_trace_json(events, trace_path)

        analysis = self.detector.analyze_overlap(trace_path, self.mock_config)

        self.assertEqual(analysis.total_gemm_time, 1000)
        self.assertEqual(analysis.total_comm_time, 700)
        # Overlap should be 200us (1000 - 800)
        self.assertEqual(analysis.total_overlap_time, 200)

    def test_analyze_overlap_with_no_overlap(self):
        """Test overlap detection when GEMM and comm don't overlap."""
        # GEMM: 0-1000us, Comm: 2000-3000us (no overlap)
        events = [
            {"name": "cutlass_gemm_kernel", "cat": "kernel", "ts": 0, "dur": 1000},
            {"name": "nccl:all_gather", "cat": "kernel", "ts": 2000, "dur": 1000},
        ]
        trace_path = os.path.join(self.temp_dir, "no_overlap.json")
        create_mock_trace_json(events, trace_path)

        analysis = self.detector.analyze_overlap(trace_path, self.mock_config)

        self.assertEqual(analysis.total_overlap_time, 0)
        self.assertEqual(analysis.total_overlap_ratio, 0.0)

    def test_analyze_overlap_multiple_gemm_events(self):
        """Test overlap with multiple GEMM events."""
        # Multiple GEMM kernels overlapping with comm
        events = [
            {"name": "cutlass_gemm_kernel_1", "cat": "kernel", "ts": 0, "dur": 500},
            {"name": "cutlass_gemm_kernel_2", "cat": "kernel", "ts": 500, "dur": 500},
            {"name": "userbuffers_allgather", "cat": "kernel", "ts": 200, "dur": 600},
        ]
        trace_path = os.path.join(self.temp_dir, "multi_gemm.json")
        create_mock_trace_json(events, trace_path)

        analysis = self.detector.analyze_overlap(trace_path, self.mock_config)

        self.assertEqual(analysis.num_gemm_events, 2)
        self.assertEqual(analysis.total_gemm_time, 1000)  # 500 + 500

    def test_analyze_overlap_e2e_time(self):
        """Test end-to-end time calculation."""
        # Events spanning 0-1500us
        events = [
            {"name": "cutlass_gemm_kernel", "cat": "kernel", "ts": 0, "dur": 1000},
            {"name": "userbuffers_allgather", "cat": "kernel", "ts": 500, "dur": 1000},
        ]
        trace_path = os.path.join(self.temp_dir, "e2e_time.json")
        create_mock_trace_json(events, trace_path)

        analysis = self.detector.analyze_overlap(trace_path, self.mock_config)

        # E2E time should be from 0 to 1500
        self.assertEqual(analysis.operator_e2e_time, 1500)

    def test_analyze_overlap_gemm_patterns(self):
        """Test various GEMM kernel name patterns are detected."""
        gemm_names = [
            "void cutlass_80_tensorop_bf16_s16816gemm",
            "cublas_gemm_wrapper",
            "sm80_xmma_gemm_bf16",
            "matmul_kernel",
        ]
        for gemm_name in gemm_names:
            events = [
                {"name": gemm_name, "cat": "kernel", "ts": 0, "dur": 100},
            ]
            trace_path = os.path.join(self.temp_dir, f"gemm_{hash(gemm_name)}.json")
            create_mock_trace_json(events, trace_path)

            analysis = self.detector.analyze_overlap(trace_path, self.mock_config)
            self.assertEqual(
                analysis.num_gemm_events, 1, f"Failed to detect GEMM: {gemm_name}"
            )

    def test_analyze_overlap_comm_patterns(self):
        """Test various communication kernel name patterns are detected."""
        comm_names = [
            "userbuffers_fp16_sum_inplace_gpu_rw_ag",
            "kuserbuffers_pushsend",
            "Memcpy PtoP (Device -> Device)",
            "ncclDevKernel_AllGather",
            "nccl:reduce_scatter",
        ]
        for comm_name in comm_names:
            events = [
                {"name": comm_name, "cat": "kernel", "ts": 0, "dur": 100},
            ]
            trace_path = os.path.join(self.temp_dir, f"comm_{hash(comm_name)}.json")
            create_mock_trace_json(events, trace_path)

            analysis = self.detector.analyze_overlap(trace_path, self.mock_config)
            self.assertEqual(
                analysis.num_comm_events, 1, f"Failed to detect comm: {comm_name}"
            )

    def test_analyze_multiple_traces(self):
        """Test analyzing multiple trace files."""
        # Create two mock traces
        events1 = [
            {"name": "cutlass_gemm", "cat": "kernel", "ts": 0, "dur": 1000},
            {"name": "userbuffers_ag", "cat": "kernel", "ts": 0, "dur": 800},
        ]
        events2 = [
            {"name": "cutlass_gemm", "cat": "kernel", "ts": 0, "dur": 500},
            {"name": "userbuffers_rs", "cat": "kernel", "ts": 0, "dur": 400},
        ]

        trace_path1 = os.path.join(self.temp_dir, "trace1.json")
        trace_path2 = os.path.join(self.temp_dir, "trace2.json")
        create_mock_trace_json(events1, trace_path1)
        create_mock_trace_json(events2, trace_path2)

        config1 = TPOverlapTestConfig(
            tp_size=2,
            operator="fc1",
            phase="fprop",
            overlap_method=OverlapMethod.RING_EXCHANGE,
        )
        config2 = TPOverlapTestConfig(
            tp_size=2,
            operator="fc1",
            phase="dgrad",
            overlap_method=OverlapMethod.BULK,
            num_sm=4,
        )

        results = self.detector.analyze_multiple_traces(
            [
                (trace_path1, config1),
                (trace_path2, config2),
            ]
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].config.phase, "fprop")
        self.assertEqual(results[1].config.phase, "dgrad")


class TestUtilityFunctions(unittest.TestCase):
    """Tests for utility functions."""

    def test_calculate_overlap_ratio(self):
        """Test overlap ratio calculation."""
        self.assertAlmostEqual(calculate_overlap_ratio(1000, 500, 400), 0.8)  # 400/500
        self.assertAlmostEqual(calculate_overlap_ratio(500, 1000, 400), 0.8)  # 400/500

    def test_calculate_overlap_ratio_zero(self):
        """Test overlap ratio with zero values."""
        self.assertEqual(calculate_overlap_ratio(0, 0, 0), 0.0)
        self.assertEqual(calculate_overlap_ratio(1000, 0, 0), 0.0)
        self.assertEqual(calculate_overlap_ratio(0, 1000, 0), 0.0)

    def test_is_overlap_effective_true(self):
        """Test overlap effectiveness check (effective)."""
        mock_config = TPOverlapTestConfig(
            tp_size=2,
            operator="fc1",
            phase="fprop",
            overlap_method=OverlapMethod.RING_EXCHANGE,
        )
        analysis = OverlapAnalysis(
            config=mock_config,
            total_gemm_time=1000,
            total_comm_time=500,
            total_overlap_time=400,  # ratio = 0.8
        )
        self.assertTrue(is_overlap_effective(analysis, threshold=0.5))

    def test_is_overlap_effective_false(self):
        """Test overlap effectiveness check (not effective)."""
        mock_config = TPOverlapTestConfig(
            tp_size=2,
            operator="fc1",
            phase="fprop",
            overlap_method=OverlapMethod.RING_EXCHANGE,
        )
        analysis = OverlapAnalysis(
            config=mock_config,
            total_gemm_time=1000,
            total_comm_time=500,
            total_overlap_time=100,  # ratio = 0.2
        )
        self.assertFalse(is_overlap_effective(analysis, threshold=0.5))


class TestMergeIntervals(unittest.TestCase):
    """Tests for interval merging logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = OverlapDetector()

    def test_merge_overlapping_intervals(self):
        """Test merging overlapping intervals."""
        intervals = [
            TimeInterval(0, 100),
            TimeInterval(50, 150),
            TimeInterval(140, 200),
        ]
        merged = self.detector._merge_intervals(intervals)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].start, 0)
        self.assertEqual(merged[0].end, 200)

    def test_merge_non_overlapping_intervals(self):
        """Test merging non-overlapping intervals."""
        intervals = [
            TimeInterval(0, 100),
            TimeInterval(200, 300),
            TimeInterval(400, 500),
        ]
        merged = self.detector._merge_intervals(intervals)

        self.assertEqual(len(merged), 3)

    def test_merge_empty_intervals(self):
        """Test merging empty list."""
        merged = self.detector._merge_intervals([])
        self.assertEqual(len(merged), 0)

    def test_merge_single_interval(self):
        """Test merging single interval."""
        intervals = [TimeInterval(0, 100)]
        merged = self.detector._merge_intervals(intervals)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].start, 0)
        self.assertEqual(merged[0].end, 100)


if __name__ == "__main__":
    unittest.main()

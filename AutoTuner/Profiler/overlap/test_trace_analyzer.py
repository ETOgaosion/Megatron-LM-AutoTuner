#!/usr/bin/env python3
"""
Test script for TraceAnalyzer.

Run with:
    python -m AutoTuner.Profiler.overlap.test_trace_analyzer
"""

import os
import sys


def test_trace_analyzer():
    """Test the TraceAnalyzer with the sample trace file."""
    from AutoTuner.Profiler.overlap.trace_analyzer import TraceAnalyzer, analyze_trace_file

    # Sample trace path
    sample_trace = "outputs/sample/jss-Rack-Server_951.1768541469734146129.pt.trace.json"

    if not os.path.exists(sample_trace):
        print(f"Error: Sample trace file not found: {sample_trace}")
        print("Please ensure the sample trace exists.")
        return False

    print("=" * 60)
    print("Testing TraceAnalyzer")
    print("=" * 60)
    print(f"Trace file: {sample_trace}")
    print("")

    # Test using analyze_trace_file helper
    print("1. Testing analyze_trace_file()...")
    summary = analyze_trace_file(sample_trace)
    print(f"   Total events: {summary['total_events']}")
    print(f"   Kernel events: {summary['kernel_events']}")
    print(f"   GEMM events: {summary['gemm_events']}")
    print(f"   GEMM kernels: {summary['gemm_kernels']}")
    print(f"   Comm events: {summary['comm_events']}")
    print(f"   Comm kernels: {summary['comm_kernels']}")
    print(f"   Num streams: {summary['num_streams']}")
    print(f"   Duration: {summary['duration_us']:.2f} us")
    print("")

    # Test TraceAnalyzer class
    print("2. Testing TraceAnalyzer class...")
    analyzer = TraceAnalyzer(sample_trace)
    events = analyzer.parse_trace()
    print(f"   Parsed {len(events)} events")

    gemm_kernels = analyzer.extract_gemm_kernels()
    comm_kernels = analyzer.extract_comm_kernels()
    print(f"   GEMM kernels: {len(gemm_kernels)}")
    print(f"   Comm kernels: {len(comm_kernels)}")

    # Show sample GEMM kernel names
    if gemm_kernels:
        print("")
        print("   Sample GEMM kernels:")
        for k in gemm_kernels[:3]:
            print(f"     - {k.name[:80]}...")

    # Show sample comm kernel names
    if comm_kernels:
        print("")
        print("   Sample comm kernels:")
        for k in comm_kernels[:3]:
            print(f"     - {k.name[:80]}...")

    # Test stream grouping
    stream_events = analyzer.get_events_by_stream()
    print("")
    print(f"3. Events grouped by stream: {len(stream_events)} streams")
    for tid, events in list(stream_events.items())[:5]:
        print(f"   Stream {tid}: {len(events)} events")

    print("")
    print("=" * 60)
    print("TraceAnalyzer tests passed!")
    print("=" * 60)
    return True


def test_config_generator():
    """Test the TPOverlapConfigGenerator."""
    from AutoTuner.Profiler.overlap.config_generator import (
        TPOverlapConfigGenerator,
        TPOverlapTunerConfig,
    )

    print("")
    print("=" * 60)
    print("Testing TPOverlapConfigGenerator")
    print("=" * 60)

    config = TPOverlapTunerConfig(
        model_name="Qwen/Qwen3-0.6B",
        hidden_size=1024,
        ffn_hidden_size=3072,
        num_attention_heads=16,
        num_kv_heads=8,
        max_tp_size=4,
        operators=["fc1", "qkv"],
    )

    generator = TPOverlapConfigGenerator(config)
    all_configs = generator.generate_all_configs()

    print(f"Generated {len(all_configs)} test configurations")
    print("")
    print("Sample configurations:")
    for cfg in all_configs[:5]:
        print(f"  - {cfg.get_test_id()}")

    # Test YAML generation
    default_yaml = generator.generate_default_yaml_config()
    print("")
    print("Default YAML config keys:")
    for key in list(default_yaml.keys())[:5]:
        print(f"  - {key}: {default_yaml[key]}")

    print("")
    print("=" * 60)
    print("ConfigGenerator tests passed!")
    print("=" * 60)
    return True


def test_overlap_detector():
    """Test the OverlapDetector with sample trace."""
    from AutoTuner.Profiler.overlap.config_generator import (
        OverlapMethod,
        TPOverlapTestConfig,
    )
    from AutoTuner.Profiler.overlap.overlap_detector import OverlapDetector

    sample_trace = "outputs/sample/jss-Rack-Server_951.1768541469734146129.pt.trace.json"

    if not os.path.exists(sample_trace):
        print("Skipping OverlapDetector test (sample trace not found)")
        return True

    print("")
    print("=" * 60)
    print("Testing OverlapDetector")
    print("=" * 60)

    # Create a dummy config for testing
    config = TPOverlapTestConfig(
        tp_size=2,
        operator="fc1",
        phase="fprop",
        overlap_method=OverlapMethod.RING_EXCHANGE,
        aggregate=0,
    )

    detector = OverlapDetector()
    analysis = detector.analyze_overlap(sample_trace, config)

    print(f"Analysis results for {config.get_test_id()}:")
    print(f"  Total GEMM time: {analysis.total_gemm_time:.2f} us")
    print(f"  Total comm time: {analysis.total_comm_time:.2f} us")
    print(f"  Total overlap time: {analysis.total_overlap_time:.2f} us")
    print(f"  Total overlap ratio: {analysis.total_overlap_ratio:.2%}")
    print(f"  Forward overlap ratio: {analysis.forward_overlap_ratio:.2%}")
    print(f"  Backward overlap ratio: {analysis.backward_overlap_ratio:.2%}")
    print(f"  Num GEMM events: {analysis.num_gemm_events}")
    print(f"  Num comm events: {analysis.num_comm_events}")

    print("")
    print("=" * 60)
    print("OverlapDetector tests passed!")
    print("=" * 60)
    return True


def main():
    """Run all tests."""
    print("Running TP Overlap Tuner Tests")
    print("")

    success = True

    try:
        if not test_trace_analyzer():
            success = False
    except Exception as e:
        print(f"TraceAnalyzer test failed: {e}")
        success = False

    try:
        if not test_config_generator():
            success = False
    except Exception as e:
        print(f"ConfigGenerator test failed: {e}")
        success = False

    try:
        if not test_overlap_detector():
            success = False
    except Exception as e:
        print(f"OverlapDetector test failed: {e}")
        success = False

    if success:
        print("")
        print("All tests passed!")
        return 0
    else:
        print("")
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

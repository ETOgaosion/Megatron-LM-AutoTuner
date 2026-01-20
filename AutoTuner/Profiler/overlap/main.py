#!/usr/bin/env python3
"""
TP Overlap Tuner - Main Entry Point.

This module implements the complete TP overlap tuning workflow:
    0. [INPUT] User input a model config
    1. Generate test cases using binary search strategy
    2. Run the configs with torch profiler
    3. Analyze the results from torch profiler JSON
    4. Generate report for each operator with optimal configurations

Usage:
    python -m AutoTuner.Profiler.overlap.main --model-name Qwen/Qwen3-0.6B
    python -m AutoTuner.Profiler.overlap.main --model-name meta-llama/Llama-2-7b --max-tp-size 4
"""

import argparse
import sys
from datetime import datetime
from typing import List, Optional

from .config_generator import (
    TPOverlapConfigGenerator,
    TPOverlapTunerConfig,
)
from .tuner import TPOverlapTuner


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="tp-overlap-tuner",
        description="TP Overlap Tuner - Auto-tune TP communication/computation overlap configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  0. [INPUT] User provides model config via --model-name
  1. Generate test cases using binary search for num_sm parameter
  2. Run configs with torch profiler (TP=2,4,8 up to max-tp-size)
  3. Analyze torch profiler JSON traces for GEMM/comm overlap
  4. Generate report with optimal configurations for each operator

Examples:
  # Run complete tuning workflow
  python -m AutoTuner.Profiler.overlap.main --model-name Qwen/Qwen3-0.6B

  # Tune with specific TP size limit
  python -m AutoTuner.Profiler.overlap.main --model-name meta-llama/Llama-2-7b --max-tp-size 4

  # Tune specific operators only
  python -m AutoTuner.Profiler.overlap.main --model-name Qwen/Qwen3-0.6B --operators qkv proj
""",
    )

    # Step 0: Model config input
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name from HuggingFace. Model parameters (hidden_size, ffn_hidden_size, "
        "num_attention_heads, num_kv_heads) are automatically fetched.",
    )

    # Step 1: Test case generation parameters
    parser.add_argument(
        "--max-tp-size",
        type=int,
        default=8,
        help="Maximum TP size to test. Tests TP=2,4,8 up to this value. (default: 8)",
    )
    parser.add_argument(
        "--max-token-len",
        type=int,
        default=8192,
        help="Maximum token length for testing. Use high value for peak computational intensity. (default: 8192)",
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        default=["fc1", "fc2", "qkv", "proj"],
        choices=["fc1", "fc2", "qkv", "proj"],
        help="Operators to tune: fc1, fc2 (MLP), qkv, proj (attention). (default: all)",
    )
    parser.add_argument(
        "--min-num-sm",
        type=int,
        default=1,
        help="Minimum num_sm for bulk method binary search. (default: 1)",
    )
    parser.add_argument(
        "--max-num-sm",
        type=int,
        default=16,
        help="Maximum num_sm for bulk method binary search. (default: 16)",
    )

    # Step 3: Analysis parameters
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.5,
        help="Minimum overlap ratio to consider effective. (default: 0.5)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. (default: auto-generated with timestamp)",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point - runs the complete TP overlap tuning workflow.

    Workflow:
        0. [INPUT] Parse user model config
        1. Generate test cases using binary search
        2. Run configs with torch profiler
        3. Analyze torch profiler results
        4. Generate report for each operator
    """
    # ========================================
    # Step 0: [INPUT] User input a model config
    # ========================================
    args = parse_args(argv)

    # Set default output directory with timestamp
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/tp_overlap_tuner/{timestamp}"

    print("=" * 70)
    print("TP OVERLAP TUNER")
    print("=" * 70)
    print("")
    print("Step 0: [INPUT] Loading model configuration...")
    print(f"  Model: {args.model_name}")

    # Create tuner config (model params auto-fetched from model_name)
    tuner_config = TPOverlapTunerConfig(
        model_name=args.model_name,
        max_tp_size=args.max_tp_size,
        max_token_len=args.max_token_len,
        operators=args.operators,
        output_dir=output_dir,
        min_num_sm=args.min_num_sm,
        max_num_sm=args.max_num_sm,
    )

    # Display auto-fetched model parameters
    print(f"  Hidden Size: {tuner_config.hidden_size}")
    print(f"  FFN Hidden Size: {tuner_config.ffn_hidden_size}")
    print(f"  Num Attention Heads: {tuner_config.num_attention_heads}")
    print(f"  Num KV Heads: {tuner_config.num_kv_heads}")
    print("")
    print(f"  Operators to tune: {', '.join(tuner_config.operators)}")
    print(f"  Max TP Size: {tuner_config.max_tp_size}")
    print(f"  Max Token Length: {tuner_config.max_token_len}")
    print(f"  Output Directory: {tuner_config.output_dir}")
    print("")

    # ========================================
    # Step 1: Generate test cases (binary search)
    # ========================================
    print("-" * 70)
    print("Step 1: Generating test cases using binary search strategy...")

    generator = TPOverlapConfigGenerator(tuner_config)
    all_configs = generator.generate_all_configs()

    print(f"  TP sizes to test: {generator.tp_sizes}")
    print(f"    - TP=1: Baseline (no tensor parallelism, pure computation)")
    print(f"    - TP>=2: Test with overlap configurations")
    print(f"  Operators: {tuner_config.operators}")
    print(f"  Phases: fprop, dgrad, wgrad")
    print(f"  Methods: baseline, ring_exchange (agg 0/1), bulk (num_sm 1,2,4,8,16)")
    print(f"  Total configurations: {len(all_configs)}")
    print("")

    # ========================================
    # Steps 2-4: Run profiling, analyze, generate report
    # ========================================
    # The TPOverlapTuner.run() method handles:
    #   Step 2: Run configs with torch profiler
    #   Step 3: Analyze torch profiler JSON traces
    #   Step 4: Generate report for each operator

    tuner = TPOverlapTuner(
        tuner_config=tuner_config,
        overlap_threshold=args.overlap_threshold,
    )

    # Run the complete workflow (profile + analyze + report)
    report = tuner.run(skip_profiling=False)

    # ========================================
    # Summary
    # ========================================
    print("")
    print("=" * 70)
    print("TUNING COMPLETED")
    print("=" * 70)
    print("")
    print("Generated files:")
    print(f"  - {output_dir}/tuning_report.json    (Full analysis)")
    print(f"  - {output_dir}/summary.txt           (Human-readable summary)")
    print(f"  - {output_dir}/optimal_tp_comm_overlap_cfg.yaml")
    print("")
    print("To use the optimal config, copy it to:")
    print("  AutoTuner/testbench/profile/configs/local/tp_comm_overlap_cfg.yaml")
    print("")

    # Print top recommendations
    if report.recommendations:
        print("Top recommendations:")
        for rec in report.recommendations[:10]:
            print(f"  * {rec}")
        if len(report.recommendations) > 10:
            print(f"  ... and {len(report.recommendations) - 10} more (see summary.txt)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

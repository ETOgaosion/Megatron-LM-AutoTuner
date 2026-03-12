#!/usr/bin/env python3
"""CLI entry point for CP overlap profiling and analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Optional

from .cp_overlap_trace_analyzer import DEFAULT_FORWARD_PATTERN
from .runner import CPOverlapInputCase, CPOverlapRunner, CPOverlapRunnerConfig


def _parse_case(value: str) -> CPOverlapInputCase:
    try:
        seqlen_raw, max_token_len_raw = value.split(":", maxsplit=1)
        return CPOverlapInputCase(
            seqlen=int(seqlen_raw),
            max_token_len=int(max_token_len_raw),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--case must use the form <seqlen>:<max_token_len>"
        ) from exc


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cp-overlap-profiler",
        description="Profile TEAttenWithCPEnhanced with CP=2 and analyze CP overlap.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Model name from HuggingFace. Optional with --skip-profiling if it can "
            "be inferred from output-dir/test_cases/cases_manifest.json."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        default=None,
        help="Input case in the form <seqlen>:<max_token_len>. Repeat to add more cases.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for generated cases."
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=2,
        help="Micro batch size for generated cases.",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="thd",
        choices=["thd", "bshd"],
        help="Input layout for generated cases.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="megatron",
        help="System field written into the generated test cases.",
    )
    parser.add_argument(
        "--cp-size",
        type=int,
        default=2,
        help="Context parallel size. Requirement default is 2.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to outputs/cp_overlap_tuner/<timestamp>.",
    )
    parser.add_argument(
        "--skip-profiling",
        action="store_true",
        help="Skip profiling and analyze existing traces under --output-dir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for CP forward-range selection.",
    )
    parser.add_argument(
        "--range-index",
        type=int,
        default=None,
        help="Explicit CP forward-range index to analyze.",
    )
    parser.add_argument(
        "--forward-pattern",
        type=str,
        default=DEFAULT_FORWARD_PATTERN,
        help="Python-function range name substring used to select CP forward ranges.",
    )
    return parser.parse_args(argv)


def _resolve_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return args.model_name
    if not args.skip_profiling or not args.output_dir:
        raise ValueError("--model-name is required unless --skip-profiling can infer it.")

    manifest_path = os.path.join(args.output_dir, "test_cases", "cases_manifest.json")
    if not os.path.exists(manifest_path):
        raise ValueError(
            "--model-name was not provided and no prior "
            f"{manifest_path} was found."
        )

    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)
    model_name = manifest.get("model")
    if not model_name:
        raise ValueError(
            f"Could not infer model name from {manifest_path}; please pass --model-name."
        )
    return model_name


def _resolve_cases(args: argparse.Namespace, output_dir: str) -> List[CPOverlapInputCase]:
    if args.case:
        return [
            CPOverlapInputCase(
                seqlen=case.seqlen,
                max_token_len=case.max_token_len,
                batch_size=args.batch_size,
                micro_batch_size=args.micro_batch_size,
                shape=args.shape,
                system=args.system,
            )
            for case in args.case
        ]

    if not args.skip_profiling:
        return [
            CPOverlapInputCase(
                seqlen=40960,
                max_token_len=40960,
                batch_size=args.batch_size,
                micro_batch_size=args.micro_batch_size,
                shape=args.shape,
                system=args.system,
            )
        ]

    manifest_path = os.path.join(output_dir, "test_cases", "cases_manifest.json")
    if not os.path.exists(manifest_path):
        raise ValueError(
            "No --case values were provided and no existing cases manifest was found "
            f"under {manifest_path}."
        )

    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)
    return [CPOverlapInputCase(**case_data) for case_data in manifest.get("cases", [])]


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("outputs", "cp_overlap_tuner", timestamp)

    model_name = _resolve_model_name(args)
    cases = _resolve_cases(args, output_dir)

    print("=" * 70)
    print("CP OVERLAP PROFILER")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"CP Size: {args.cp_size}")
    print(f"Output Directory: {output_dir}")
    print("Cases:")
    for case in cases:
        print(
            "  - "
            f"seqlen={case.seqlen}, max_token_len={case.max_token_len}, "
            f"batch_size={case.batch_size}, micro_batch_size={case.micro_batch_size}, "
            f"shape={case.shape}"
        )
    print("")

    runner = CPOverlapRunner(
        CPOverlapRunnerConfig(
            model_name=model_name,
            cases=cases,
            output_dir=output_dir,
            cp_size=args.cp_size,
            seed=args.seed,
            range_index=args.range_index,
            forward_pattern=args.forward_pattern,
        )
    )
    results = runner.run(skip_profiling=args.skip_profiling)

    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    for result in results:
        if not result.success:
            print(f"{result.case.get_case_id()}: ERROR: {result.error_message}")
            continue
        if not result.analysis:
            print(f"{result.case.get_case_id()}: ERROR: analysis missing")
            continue
        print(
            f"{result.case.get_case_id()}: "
            f"overlap_ratio={result.analysis['overlap_ratio']:.4f}, "
            f"compute_time_us={result.analysis['compute_time_us']:.2f}, "
            f"comm_time_us={result.analysis['comm_time_us']:.2f}, "
            f"overlapped_comm_time_us={result.analysis['overlapped_comm_time_us']:.2f}"
        )

    print("")
    print(f"Full report: {os.path.join(output_dir, 'cp_overlap_report.json')}")
    print(f"Summary: {os.path.join(output_dir, 'summary.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

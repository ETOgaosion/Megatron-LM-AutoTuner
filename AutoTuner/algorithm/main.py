#!/usr/bin/env python3
"""Dispatch dense or MoE tuning based on the input model config."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    from transformers import PretrainedConfig
else:
    PretrainedConfig = Any

_MOE_EXPERT_COUNT_FIELDS = (
    "num_moe_experts",
    "num_local_experts",
    "num_experts",
    "moe_num_experts",
    "n_routed_experts",
)
_MOE_ROUTING_FIELDS = (
    "moe_router_topk",
    "num_experts_per_tok",
    "num_experts_per_token",
)
_MOE_HINT_KEYWORDS = ("moe", "mixtral", "deepseekv3", "deepseek_v3")


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="tuning-algorithm",
        description="Dispatch dense or MoE tuning according to the input model config.",
        add_help=False,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file. Values are expanded into CLI args before dispatch.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("auto", "dense", "moe"),
        default="auto",
        help="Override automatic model-type detection.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name or local HuggingFace config path used for automatic dispatch.",
    )
    parser.add_argument("-h", "--help", action="store_true", help="Show help.")
    args, remaining = parser.parse_known_args(argv)
    return parser, args, remaining


def _strip_config_args(argv: Sequence[str]) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--config":
            skip_next = True
            continue
        if arg.startswith("--config="):
            continue
        stripped.append(arg)
    return stripped


def _stringify_argv_items(values: Sequence[Any], field_name: str) -> list[str]:
    argv: list[str] = []
    for value in values:
        if value is None:
            raise ValueError(f"'{field_name}' entries cannot be null.")
        argv.append(str(value))
    return argv


def load_cli_args_from_config(config_path: str) -> list[str]:
    config_file = Path(config_path).expanduser().resolve()
    with config_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return _stringify_argv_items(payload, "config")
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must be either an object or an array of CLI args.")

    argv: list[str] = []
    for key, value in payload.items():
        if key in {"argv", "extra_args"}:
            continue
        if value is None:
            continue

        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, list):
            if value:
                argv.append(flag)
                argv.extend(_stringify_argv_items(value, key))
            continue
        if isinstance(value, dict):
            raise ValueError(
                f"'{key}' must be a scalar, boolean, array, or be moved into 'argv'."
            )
        argv.extend([flag, str(value)])

    raw_argv = payload.get("argv")
    if raw_argv is not None:
        if not isinstance(raw_argv, list):
            raise ValueError("'argv' must be an array of CLI arguments.")
        argv.extend(_stringify_argv_items(raw_argv, "argv"))

    extra_args = payload.get("extra_args")
    if extra_args is not None:
        if not isinstance(extra_args, list):
            raise ValueError("'extra_args' must be an array of CLI arguments.")
        argv.extend(_stringify_argv_items(extra_args, "extra_args"))

    return argv


def _resolve_model_reference(model_name: str) -> str:
    model_root = os.environ.get("HUGGINGFACE_MODEL_DIR")
    if model_root:
        return os.path.join(model_root, model_name)
    return model_name


def load_model_config(model_name: str) -> PretrainedConfig:
    resolved_reference = _resolve_model_reference(model_name)
    if os.path.isfile(resolved_reference) and resolved_reference.endswith(".json"):
        with open(resolved_reference, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Model config JSON at {resolved_reference} must contain an object."
            )
        return SimpleNamespace(**payload)

    from transformers import AutoConfig

    return AutoConfig.from_pretrained(
        resolved_reference,
        trust_remote_code=True,
    )


def is_moe_model_config(model_config: PretrainedConfig) -> bool:
    model_type = str(getattr(model_config, "model_type", "")).lower()
    architectures = [
        str(architecture).lower()
        for architecture in getattr(model_config, "architectures", []) or []
    ]
    if any(keyword in model_type for keyword in _MOE_HINT_KEYWORDS):
        return True
    if any(
        keyword in architecture
        for architecture in architectures
        for keyword in _MOE_HINT_KEYWORDS
    ):
        return True

    expert_count = 0
    for field_name in _MOE_EXPERT_COUNT_FIELDS:
        value = getattr(model_config, field_name, None)
        if isinstance(value, int):
            expert_count = max(expert_count, value)
    if expert_count <= 1:
        return False

    if any(hasattr(model_config, field_name) for field_name in _MOE_ROUTING_FIELDS):
        return True
    return True


def select_algorithm(
    algorithm: str,
    model_config: Optional[PretrainedConfig] = None,
) -> str:
    if algorithm != "auto":
        return algorithm
    if model_config is None:
        raise ValueError("model_config is required when algorithm='auto'.")
    return "moe" if is_moe_model_config(model_config) else "dense"


def _resolve_entrypoint(selected_algorithm: str):
    if selected_algorithm == "dense":
        from .dense_tuning import main as dense_main

        return dense_main
    if selected_algorithm == "moe":
        from .moe_tuning import main as moe_main

        return moe_main
    raise ValueError(f"Unsupported algorithm: {selected_algorithm}")


def _run_help(selected_algorithm: Optional[str], argv: Sequence[str]) -> int:
    if selected_algorithm is None:
        print(
            "Usage: python -m AutoTuner.algorithm.main "
            "[--config path.json] [--algorithm auto|dense|moe] ..."
        )
        print("")
        print(
            "Pass --model-name in auto mode to detect dense vs MoE from a "
            "HuggingFace repo, local model directory, or local config.json."
        )
        print("JSON config fields are expanded into CLI flags before dispatch.")
        print("All other arguments are forwarded to the selected algorithm module.")
        print("")
        print("Examples:")
        print(
            "  python -m AutoTuner.algorithm.main --model-name Qwen/Qwen3-0.6B "
            "--prompt-len 4096 --response-len 1024 --nprocs-per-node 8"
        )
        print(
            "  python -m AutoTuner.algorithm.main --config "
            "tests/functional_test/algorithm/algorithm_config_auto.json"
        )
        print(
            "  python -m AutoTuner.algorithm.main --algorithm moe "
            "--model-name Qwen/Qwen3-30B-A3B --prompt-len 4096 "
            "--response-len 1024 --nprocs-per-node 16 --help"
        )
        return 0

    entrypoint = _resolve_entrypoint(selected_algorithm)
    try:
        return int(entrypoint(argv))
    except SystemExit as exc:
        return int(exc.code or 0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args(raw_argv)

    config_argv: list[str] = []
    if config_args.config:
        config_argv = load_cli_args_from_config(config_args.config)

    argv = config_argv + _strip_config_args(raw_argv)
    parser, args, remaining = parse_args(argv)

    selected_algorithm: Optional[str] = None
    if args.algorithm == "auto" and args.model_name:
        selected_algorithm = select_algorithm(
            algorithm="auto",
            model_config=load_model_config(args.model_name),
        )
    elif args.algorithm != "auto":
        selected_algorithm = args.algorithm

    if args.help:
        return _run_help(selected_algorithm, argv=remaining + ["--help"])

    if args.algorithm == "auto":
        if not args.model_name:
            parser.error("--model-name is required when --algorithm=auto.")
        assert selected_algorithm is not None
        print(
            f"Selected tuning algorithm: {selected_algorithm} "
            f"(detected from model config for {args.model_name})"
        )
    else:
        selected_algorithm = args.algorithm
        print(f"Selected tuning algorithm: {selected_algorithm} (--algorithm override)")

    entrypoint = _resolve_entrypoint(selected_algorithm)
    return int(entrypoint(remaining))


if __name__ == "__main__":
    raise SystemExit(main())

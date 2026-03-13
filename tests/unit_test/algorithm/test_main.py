from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import AutoTuner.algorithm.main as algorithm_main


def test_is_moe_model_config_detects_moe_fields() -> None:
    config = SimpleNamespace(num_local_experts=8, moe_router_topk=2, model_type="qwen2")

    assert algorithm_main.is_moe_model_config(config) is True


def test_is_moe_model_config_detects_dense_model() -> None:
    config = SimpleNamespace(
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        num_attention_heads=16,
    )

    assert algorithm_main.is_moe_model_config(config) is False


def test_select_algorithm_returns_dense_for_dense_config() -> None:
    config = SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])

    assert algorithm_main.select_algorithm("auto", config) == "dense"


def test_load_model_config_supports_local_config_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"model_type": "qwen2_moe", "num_moe_experts": 8}),
        encoding="utf-8",
    )

    model_config = algorithm_main.load_model_config(str(config_path))

    assert model_config.model_type == "qwen2_moe"
    assert model_config.num_moe_experts == 8


def test_main_dispatches_to_moe(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        algorithm_main,
        "load_model_config",
        lambda model_name: SimpleNamespace(num_moe_experts=8, moe_router_topk=2),
    )
    monkeypatch.setattr(
        algorithm_main,
        "_resolve_entrypoint",
        lambda algorithm: (
            lambda forwarded_argv: calls.append((algorithm, list(forwarded_argv))) or 0
        ),
    )

    exit_code = algorithm_main.main(
        [
            "--model-name",
            "Qwen/Qwen3-30B-A3B",
            "--prompt-len",
            "4096",
            "--response-len",
            "1024",
            "--nprocs-per-node",
            "16",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            "moe",
            [
                "--prompt-len",
                "4096",
                "--response-len",
                "1024",
                "--nprocs-per-node",
                "16",
            ],
        )
    ]


def test_main_dispatches_to_dense_with_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        algorithm_main,
        "_resolve_entrypoint",
        lambda algorithm: (
            lambda forwarded_argv: calls.append((algorithm, list(forwarded_argv))) or 0
        ),
    )

    exit_code = algorithm_main.main(
        [
            "--algorithm",
            "dense",
            "--model-name",
            "Qwen/Qwen3-0.6B",
            "--prompt-len",
            "4096",
            "--response-len",
            "1024",
            "--nprocs-per-node",
            "8",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            "dense",
            [
                "--prompt-len",
                "4096",
                "--response-len",
                "1024",
                "--nprocs-per-node",
                "8",
            ],
        )
    ]


def test_main_reads_args_from_config_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, list[str]]] = []
    config_path = tmp_path / "algorithm_config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": "Qwen/Qwen3-0.6B",
                "prompt_len": 2048,
                "response_len": 512,
                "nprocs_per_node": 8,
                "skip_tp_profiling": True,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        algorithm_main,
        "load_model_config",
        lambda model_name: SimpleNamespace(model_type="qwen2"),
    )
    monkeypatch.setattr(
        algorithm_main,
        "_resolve_entrypoint",
        lambda algorithm: (
            lambda forwarded_argv: calls.append((algorithm, list(forwarded_argv))) or 0
        ),
    )

    exit_code = algorithm_main.main(["--config", str(config_path)])

    assert exit_code == 0
    assert calls == [
        (
            "dense",
            [
                "--prompt-len",
                "2048",
                "--response-len",
                "512",
                "--nprocs-per-node",
                "8",
                "--skip-tp-profiling",
            ],
        )
    ]


def test_main_reads_moe_selection_from_config_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, list[str]]] = []
    config_path = tmp_path / "algorithm_config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": "Qwen/Qwen3-30B-A3B",
                "prompt_len": 2048,
                "response_len": 512,
                "nprocs_per_node": 16,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        algorithm_main,
        "load_model_config",
        lambda model_name: SimpleNamespace(num_experts=8, moe_router_topk=2),
    )
    monkeypatch.setattr(
        algorithm_main,
        "_resolve_entrypoint",
        lambda algorithm: (
            lambda forwarded_argv: calls.append((algorithm, list(forwarded_argv))) or 0
        ),
    )

    exit_code = algorithm_main.main(["--config", str(config_path)])

    assert exit_code == 0
    assert calls == [
        (
            "moe",
            [
                "--prompt-len",
                "2048",
                "--response-len",
                "512",
                "--nprocs-per-node",
                "16",
            ],
        )
    ]


def test_main_allows_cli_to_override_config_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, list[str]]] = []
    config_path = tmp_path / "algorithm_config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": "Qwen/Qwen3-0.6B",
                "prompt_len": 2048,
                "response_len": 512,
                "nprocs_per_node": 8,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        algorithm_main,
        "load_model_config",
        lambda model_name: SimpleNamespace(model_type="qwen2"),
    )
    monkeypatch.setattr(
        algorithm_main,
        "_resolve_entrypoint",
        lambda algorithm: (
            lambda forwarded_argv: calls.append((algorithm, list(forwarded_argv))) or 0
        ),
    )

    exit_code = algorithm_main.main(
        ["--config", str(config_path), "--response-len", "1024"]
    )

    assert exit_code == 0
    assert calls == [
        (
            "dense",
            [
                "--prompt-len",
                "2048",
                "--response-len",
                "512",
                "--nprocs-per-node",
                "8",
                "--response-len",
                "1024",
            ],
        )
    ]


def test_main_requires_model_name_for_auto_mode() -> None:
    with pytest.raises(SystemExit) as excinfo:
        algorithm_main.main(
            [
                "--prompt-len",
                "4096",
                "--response-len",
                "1024",
                "--nprocs-per-node",
                "8",
            ]
        )

    assert excinfo.value.code == 2

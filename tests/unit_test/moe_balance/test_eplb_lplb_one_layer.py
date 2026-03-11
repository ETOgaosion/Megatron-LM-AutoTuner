from __future__ import annotations

import importlib.util
import json
import math
import zlib
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
EPLB_PATH = REPO_ROOT / "EPLB" / "eplb.py"
LPLB_EPLB_PATH = REPO_ROOT / "LPLB" / "lplb" / "eplb.py"

MODEL_CONFIG_PATHS = {
    "Qwen/Qwen1.5-MoE-A2.7B": Path(
        "/data/common/models/Qwen/Qwen1.5-MoE-A2.7B/config.json"
    ),
    "Qwen/Qwen3-30B-A3B-Base": Path(
        "/data/common/models/Qwen/Qwen3-30B-A3B-Base/config.json"
    ),
    "Qwen/Qwen3-235B-A22B": Path(
        "/data/common/models/Qwen/Qwen3-235B-A22B/config.json"
    ),
    "deepseek-ai/DeepSeek-V3-Base": Path(
        "/data/common/models/deepseek-ai/DeepSeek-V3-Base/config.json"
    ),
}

FALLBACK_NUM_EXPERTS = {
    "Qwen/Qwen1.5-MoE-A2.7B": 60,
    "Qwen/Qwen3-30B-A3B-Base": 128,
    "Qwen/Qwen3-235B-A22B": 128,
    "deepseek-ai/DeepSeek-V3-Base": 256,
}


def _load_module(module_name: str, module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(f"Module path does not exist: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EPLB = _load_module("deepseek_eplb_module", EPLB_PATH)
LPLB_EPLB = _load_module("deepseek_lplb_eplb_module", LPLB_EPLB_PATH)


def _read_num_experts(model_name: str) -> int:
    cfg_path = MODEL_CONFIG_PATHS[model_name]
    if not cfg_path.exists():
        return FALLBACK_NUM_EXPERTS[model_name]

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    num_experts = cfg.get("num_experts", cfg.get("n_routed_experts"))
    if num_experts is None:
        return FALLBACK_NUM_EXPERTS[model_name]
    return int(num_experts)


def _choose_num_groups(num_experts: int) -> int:
    for candidate in (16, 12, 10, 8, 6, 5, 4, 3, 2, 1):
        if num_experts % candidate == 0:
            return candidate
    return 1


def _mock_one_layer_weight(model_name: str, num_experts: int) -> torch.Tensor:
    seed = zlib.crc32(model_name.encode("utf-8")) & 0xFFFFFFFF
    generator = torch.Generator().manual_seed(seed)
    base = torch.randint(
        50, 5000, (num_experts,), generator=generator, dtype=torch.int64
    )
    slope = torch.linspace(1.0, 2.5, steps=num_experts)
    weight = (base.float() * slope).round().to(torch.float32)
    return weight.view(1, -1)


def _pick_topology(num_experts: int) -> tuple[int, int, int, int]:
    num_gpus = 8 if num_experts >= 64 else 4
    num_groups = _choose_num_groups(num_experts)
    num_nodes = 1
    redundant = max(1, math.ceil(num_experts * 0.25))
    num_replicas = int(math.ceil((num_experts + redundant) / num_gpus) * num_gpus)
    return num_replicas, num_groups, num_nodes, num_gpus


def _assert_rebalance_shapes_and_maps(
    phy2log: torch.Tensor,
    log2phy: torch.Tensor,
    logcnt: torch.Tensor,
    num_experts: int,
    num_replicas: int,
) -> None:
    assert phy2log.shape == (1, num_replicas)
    assert logcnt.shape == (1, num_experts)
    assert log2phy.shape[0] == 1
    assert log2phy.shape[1] == num_experts
    assert int(logcnt.sum().item()) == num_replicas
    assert int(logcnt.min().item()) >= 1
    assert int(phy2log.min().item()) >= 0
    assert int(phy2log.max().item()) < num_experts

    counted = torch.bincount(phy2log[0], minlength=num_experts).to(dtype=logcnt.dtype)
    assert torch.equal(counted, logcnt[0])

    for log_idx in range(num_experts):
        from_phy = torch.where(phy2log[0] == log_idx)[0].to(dtype=torch.int64)
        from_log2phy = log2phy[0, log_idx]
        from_log2phy = from_log2phy[from_log2phy >= 0].to(dtype=torch.int64)
        assert torch.equal(from_phy.sort().values, from_log2phy.sort().values)


@pytest.mark.parametrize("model_name", list(MODEL_CONFIG_PATHS.keys()))
def test_eplb_single_layer_mocked_weight(model_name: str) -> None:
    num_experts = _read_num_experts(model_name)
    weight = _mock_one_layer_weight(model_name, num_experts)
    num_replicas, num_groups, num_nodes, num_gpus = _pick_topology(num_experts)

    phy2log, log2phy, logcnt = EPLB.rebalance_experts(
        weight=weight,
        num_replicas=num_replicas,
        num_groups=num_groups,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
    )
    _assert_rebalance_shapes_and_maps(
        phy2log, log2phy, logcnt, num_experts, num_replicas
    )


@pytest.mark.parametrize("model_name", list(MODEL_CONFIG_PATHS.keys()))
def test_lplb_embedded_eplb_single_layer_mocked_weight(model_name: str) -> None:
    num_experts = _read_num_experts(model_name)
    weight = _mock_one_layer_weight(model_name, num_experts)
    num_replicas, num_groups, num_nodes, num_gpus = _pick_topology(num_experts)

    phy2log, log2phy, logcnt = LPLB_EPLB.rebalance_experts(
        weight=weight,
        num_replicas=num_replicas,
        num_groups=num_groups,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
    )
    _assert_rebalance_shapes_and_maps(
        phy2log, log2phy, logcnt, num_experts, num_replicas
    )


@pytest.mark.parametrize("model_name", list(MODEL_CONFIG_PATHS.keys()))
def test_eplb_matches_lplb_embedded_eplb_for_one_layer(model_name: str) -> None:
    num_experts = _read_num_experts(model_name)
    weight = _mock_one_layer_weight(model_name, num_experts)
    num_replicas, num_groups, num_nodes, num_gpus = _pick_topology(num_experts)

    eplb_phy2log, eplb_log2phy, eplb_logcnt = EPLB.rebalance_experts(
        weight=weight,
        num_replicas=num_replicas,
        num_groups=num_groups,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
    )
    lplb_phy2log, lplb_log2phy, lplb_logcnt = LPLB_EPLB.rebalance_experts(
        weight=weight,
        num_replicas=num_replicas,
        num_groups=num_groups,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
    )

    assert torch.equal(eplb_phy2log, lplb_phy2log)
    assert torch.equal(eplb_log2phy, lplb_log2phy)
    assert torch.equal(eplb_logcnt, lplb_logcnt)

#!/usr/bin/env python3

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_TEST_CASES_DIR = "tests/functional_test/runtime/generated_cases/qwen_longctx"
DEFAULT_OUTPUT_DIR = "outputs"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run runtime baseline for multiple models from config JSON."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(script_dir / "runtime_baseline_config.json"),
        help="Path to config JSON.",
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        default=os.getenv("MODEL_FILTER", ""),
        help="Only run models that contain this substring.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands but do not launch torchrun.",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default="localhost",
        help="torchrun master address.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=6010,
        help="torchrun master port.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="torchrun --nnodes.",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="torchrun --node_rank.",
    )
    return parser.parse_args()


def get_first(mapping: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def ensure_dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"'{field_name}' must be an object")
    return value


def parse_bool(value: Any, field_name: str, *, allow_none: bool = False) -> bool | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"'{field_name}' cannot be null")
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "y"):
            return True
        if normalized in ("false", "0", "no", "n"):
            return False
        if allow_none and normalized in ("none", "null", ""):
            return None
    raise ValueError(f"'{field_name}' must be true/false")


def parse_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"'{field_name}' must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must be an integer") from exc
    return parsed


def parse_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", "null", ""):
        return None
    return parse_int(value, field_name)


def normalize_model_name(model_name: str) -> str:
    return re.sub(r"[\/\.\-]+", "_", model_name.strip().lower())


def resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def build_cases(case_cfg: dict[str, Any], merged_cfg: dict[str, Any]) -> dict[str, Any]:
    seqlen = parse_int(get_first(case_cfg, ["seqlen"], merged_cfg.get("seqlen", 20480)), "seqlen")
    max_token_len = parse_int(
        get_first(case_cfg, ["max_token_len"], merged_cfg.get("max_token_len", 40960)),
        "max_token_len",
    )
    batch_size = parse_int(
        get_first(case_cfg, ["batch_size"], merged_cfg.get("batch_size", 1)),
        "batch_size",
    )
    micro_batch_size = parse_int(
        get_first(
            case_cfg,
            ["micro_batch_size"],
            merged_cfg.get("micro_batch_size", 1),
        ),
        "micro_batch_size",
    )
    shapes = get_first(case_cfg, ["shapes"], merged_cfg.get("shapes", ["bshd", "thd"]))
    if not isinstance(shapes, list) or not shapes:
        raise ValueError("'shapes' must be a non-empty array")
    for shape in shapes:
        if not isinstance(shape, str) or not shape.strip():
            raise ValueError("each item in 'shapes' must be a non-empty string")

    system = get_first(case_cfg, ["system"], merged_cfg.get("system", "megatron"))
    if not isinstance(system, str) or not system.strip():
        raise ValueError("'system' must be a non-empty string")

    cases = []
    for shape in shapes:
        cases.append(
            {
                "batch_size": batch_size,
                "micro_batch_size": micro_batch_size,
                "seqlen": seqlen,
                "max_token_len": max_token_len,
                "shape": shape,
                "system": system,
            }
        )

    return {
        "seqlen": seqlen,
        "max_token_len": max_token_len,
        "batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "cases": cases,
    }


def build_parallel(parallel_cfg: dict[str, Any], merged_cfg: dict[str, Any]) -> dict[str, Any]:
    tp_size = parse_int(
        get_first(parallel_cfg, ["tp_size", "tp"], get_first(merged_cfg, ["tp_size", "tp"], 1)),
        "tp_size",
    )
    cp_size = parse_int(
        get_first(parallel_cfg, ["cp_size", "cp"], get_first(merged_cfg, ["cp_size", "cp"], 4)),
        "cp_size",
    )
    ep_size = parse_int(
        get_first(parallel_cfg, ["ep_size", "ep"], get_first(merged_cfg, ["ep_size", "ep"], 1)),
        "ep_size",
    )
    etp_size = parse_int(
        get_first(
            parallel_cfg,
            ["etp_size", "etp"],
            get_first(merged_cfg, ["etp_size", "etp"], 1),
        ),
        "etp_size",
    )
    pp_size = parse_int(
        get_first(parallel_cfg, ["pp_size", "pp"], get_first(merged_cfg, ["pp_size", "pp"], 2)),
        "pp_size",
    )
    vpp_size = parse_optional_int(
        get_first(
            parallel_cfg,
            ["vpp_size", "vpp"],
            get_first(merged_cfg, ["vpp_size", "vpp"], 2),
        ),
        "vpp_size",
    )

    for name, value in (
        ("tp_size", tp_size),
        ("cp_size", cp_size),
        ("ep_size", ep_size),
        ("etp_size", etp_size),
        ("pp_size", pp_size),
    ):
        if value < 1:
            raise ValueError(f"'{name}' must be >= 1")

    return {
        "tp_size": tp_size,
        "cp_size": cp_size,
        "ep_size": ep_size,
        "etp_size": etp_size,
        "pp_size": pp_size,
        "vpp_size": vpp_size,
    }


def build_runtime(runtime_cfg: dict[str, Any], merged_cfg: dict[str, Any]) -> dict[str, Any]:
    num_test_cases = parse_int(
        get_first(
            runtime_cfg,
            ["num_test_cases"],
            merged_cfg.get("num_test_cases", 1),
        ),
        "num_test_cases",
    )
    max_iterations = parse_int(
        get_first(
            runtime_cfg,
            ["max_iterations"],
            merged_cfg.get("max_iterations", 10),
        ),
        "max_iterations",
    )
    warmup_iterations = parse_int(
        get_first(
            runtime_cfg,
            ["warmup_iterations"],
            merged_cfg.get("warmup_iterations", 3),
        ),
        "warmup_iterations",
    )
    share_emb = parse_bool(
        get_first(
            runtime_cfg,
            ["share_emb", "share_embeddings_and_output_weights"],
            get_first(merged_cfg, ["share_emb", "share_embeddings_and_output_weights"], None),
        ),
        "share_emb",
        allow_none=True,
    )
    run_one_data = parse_bool(
        get_first(runtime_cfg, ["run_one_data"], merged_cfg.get("run_one_data", False)),
        "run_one_data",
    )
    no_ddp = parse_bool(
        get_first(runtime_cfg, ["no_ddp"], merged_cfg.get("no_ddp", False)),
        "no_ddp",
    )
    config_dir = get_first(runtime_cfg, ["config_dir"], merged_cfg.get("config_dir"))
    override_model_config_file = get_first(
        runtime_cfg,
        ["override_model_config_file"],
        merged_cfg.get("override_model_config_file"),
    )
    override_tf_config_file = get_first(
        runtime_cfg,
        ["override_tf_config_file"],
        merged_cfg.get("override_tf_config_file"),
    )
    tp_comm_overlap_cfg = get_first(
        runtime_cfg,
        ["tp_comm_overlap_cfg"],
        merged_cfg.get("tp_comm_overlap_cfg"),
    )

    return {
        "num_test_cases": num_test_cases,
        "max_iterations": max_iterations,
        "warmup_iterations": warmup_iterations,
        "share_emb": share_emb,
        "run_one_data": run_one_data,
        "no_ddp": no_ddp,
        "config_dir": config_dir,
        "override_model_config_file": override_model_config_file,
        "override_tf_config_file": override_tf_config_file,
        "tp_comm_overlap_cfg": tp_comm_overlap_cfg,
    }


def build_distributed(args: argparse.Namespace) -> dict[str, Any]:
    master_addr = args.master_addr
    if not isinstance(master_addr, str) or not master_addr.strip():
        raise ValueError("'master_addr' must be a non-empty string")
    master_port = parse_int(args.master_port, "master_port")
    num_nodes = parse_int(args.num_nodes, "num_nodes")
    node_rank = parse_int(args.node_rank, "node_rank")
    return {
        "master_addr": master_addr,
        "master_port": master_port,
        "num_nodes": num_nodes,
        "node_rank": node_rank,
    }


def build_env(env_cfg: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NVTE_FLASH_ATTN": "1",
        "NVTE_FUSED_ATTN": "0",
        "UB_SKIPMC": "1",
    }
    run_env: dict[str, str] = {}
    merged_env = defaults.copy()
    for key, value in env_cfg.items():
        if value is None:
            continue
        merged_env[key] = str(value)

    for key, value in merged_env.items():
        run_env[key] = str(os.environ.get(key, value))
    return run_env


def build_run_spec(
    repo_root: Path, merged_cfg: dict[str, Any], distributed_info: dict[str, Any]
) -> dict[str, Any]:
    model_name = get_first(merged_cfg, ["model_name", "model"])
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("each model entry must include non-empty 'model_name'")

    case_cfg = ensure_dict(merged_cfg.get("case"), "case")
    parallel_cfg = ensure_dict(merged_cfg.get("parallel"), "parallel")
    runtime_cfg = ensure_dict(merged_cfg.get("runtime"), "runtime")
    paths_cfg = ensure_dict(merged_cfg.get("paths"), "paths")
    env_cfg = ensure_dict(merged_cfg.get("env"), "env")

    case_info = build_cases(case_cfg, merged_cfg)
    parallel_info = build_parallel(parallel_cfg, merged_cfg)
    runtime_info = build_runtime(runtime_cfg, merged_cfg)
    env_info = build_env(env_cfg)

    test_cases_dir = resolve_path(
        repo_root,
        str(get_first(paths_cfg, ["test_cases_dir"], merged_cfg.get("test_cases_dir", DEFAULT_TEST_CASES_DIR))),
    )
    output_dir = resolve_path(
        repo_root,
        str(get_first(paths_cfg, ["output_dir"], merged_cfg.get("output_dir", DEFAULT_OUTPUT_DIR))),
    )
    model_norm = normalize_model_name(model_name)
    test_cases_file = (
        f"{model_norm}_s{case_info['seqlen']}_mtl{case_info['max_token_len']}.json"
    )
    test_cases_path = test_cases_dir / test_cases_file

    runtime_config_dir = runtime_info["config_dir"]
    if runtime_config_dir is not None:
        runtime_info["config_dir"] = str(resolve_path(repo_root, str(runtime_config_dir)))

    gpus_per_node = (
        parallel_info["tp_size"]
        * parallel_info["cp_size"]
        * parallel_info["ep_size"]
        * parallel_info["etp_size"]
        * parallel_info["pp_size"]
    )

    return {
        "model_name": model_name,
        "test_cases_dir": test_cases_dir,
        "test_cases_file": test_cases_file,
        "test_cases_path": test_cases_path,
        "output_dir": output_dir,
        "gpus_per_node": gpus_per_node,
        "case": case_info,
        "parallel": parallel_info,
        "runtime": runtime_info,
        "distributed": distributed_info,
        "env": env_info,
    }


def build_command(spec: dict[str, Any]) -> list[str]:
    runtime = spec["runtime"]
    parallel = spec["parallel"]
    dist = spec["distributed"]
    python_bin = os.environ.get("PYTHON_BIN", sys.executable)

    cmd = [
        python_bin,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(spec["gpus_per_node"]),
        "--nnodes",
        str(dist["num_nodes"]),
        "--node_rank",
        str(dist["node_rank"]),
        "--master_addr",
        str(dist["master_addr"]),
        "--master_port",
        str(dist["master_port"]),
        "-m",
        "AutoTuner.runtime.baseline.main",
        "--model-name",
        spec["model_name"],
        "--test-cases-dir",
        str(spec["test_cases_dir"]),
        "--test-cases-file",
        spec["test_cases_file"],
        "--num-test-cases",
        str(runtime["num_test_cases"]),
        "--max-iterations",
        str(runtime["max_iterations"]),
        "--warmup-iterations",
        str(runtime["warmup_iterations"]),
        "--output-dir",
        str(spec["output_dir"]),
        "--tensor-model-parallel-size",
        str(parallel["tp_size"]),
        "--pipeline-model-parallel-size",
        str(parallel["pp_size"]),
        "--context-parallel-size",
        str(parallel["cp_size"]),
        "--expert-parallel-size",
        str(parallel["ep_size"]),
        "--expert-tensor-parallel-size",
        str(parallel["etp_size"]),
    ]

    if parallel["vpp_size"] is not None:
        cmd.extend(
            [
                "--virtual-pipeline-model-parallel-size",
                str(parallel["vpp_size"]),
            ]
        )
    if runtime["share_emb"] is not None:
        cmd.extend(
            [
                "--share-embeddings-and-output-weights",
                "true" if runtime["share_emb"] else "false",
            ]
        )
    if runtime["run_one_data"]:
        cmd.append("--run-one-data")
    if runtime["no_ddp"]:
        cmd.append("--no-ddp")
    if runtime["config_dir"]:
        cmd.extend(["--config-dir", str(runtime["config_dir"])])
    if runtime["override_model_config_file"]:
        cmd.extend(
            [
                "--override-model-config-file",
                str(runtime["override_model_config_file"]),
            ]
        )
    if runtime["override_tf_config_file"]:
        cmd.extend(
            [
                "--override-tf-config-file",
                str(runtime["override_tf_config_file"]),
            ]
        )
    if runtime["tp_comm_overlap_cfg"]:
        cmd.extend(["--tp-comm-overlap-cfg", str(runtime["tp_comm_overlap_cfg"])])
    return cmd


def write_case_file(spec: dict[str, Any]) -> None:
    spec["test_cases_dir"].mkdir(parents=True, exist_ok=True)
    content = {"model": spec["model_name"], "cases": spec["case"]["cases"]}
    with open(spec["test_cases_path"], "w", encoding="utf-8") as fp:
        json.dump(content, fp, indent=2)
        fp.write("\n")


def load_config(path: Path) -> list[Any]:
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("config JSON root must be an array")
    return data


def select_model_entries(config: list[Any], model_filter: str) -> list[dict[str, Any]]:
    if not config:
        raise ValueError("config JSON array must be non-empty")

    entries: list[dict[str, Any]] = []
    for idx, item in enumerate(config):
        if not isinstance(item, dict):
            raise ValueError(f"config[{idx}] must be an object")
        if item.get("enabled", True) is False:
            continue
        model_name = item.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"config[{idx}] missing non-empty 'model_name'")
        if model_filter and model_filter not in model_name:
            continue
        configs = item.get("configs")
        if not isinstance(configs, dict):
            raise ValueError(f"config[{idx}] missing object 'configs'")
        merged_cfg = deepcopy(configs)
        merged_cfg["model_name"] = model_name
        entries.append(merged_cfg)
    return entries


def run_one(spec: dict[str, Any], dry_run: bool, repo_root: Path) -> None:
    write_case_file(spec)
    cmd = build_command(spec)
    parallel = spec["parallel"]
    case = spec["case"]

    print("===================================================================")
    print(f"Running runtime baseline for {spec['model_name']}")
    print(f"case={spec['test_cases_path']}")
    print(
        "tp={tp} cp={cp} ep={ep} etp={etp} pp={pp} vpp={vpp}".format(
            tp=parallel["tp_size"],
            cp=parallel["cp_size"],
            ep=parallel["ep_size"],
            etp=parallel["etp_size"],
            pp=parallel["pp_size"],
            vpp=parallel["vpp_size"],
        )
    )
    print(
        "seqlen={seqlen} max_token_len={max_token_len} "
        "batch_size={batch} micro_batch_size={micro}".format(
            seqlen=case["seqlen"],
            max_token_len=case["max_token_len"],
            batch=case["batch_size"],
            micro=case["micro_batch_size"],
        )
    )
    print(f"cmd={shlex.join(cmd)}")
    print("===================================================================")

    if dry_run:
        return

    run_env = os.environ.copy()
    run_env.update(spec["env"])
    run_env["PYTHON_BIN"] = run_env.get("PYTHON_BIN", sys.executable)
    subprocess.run(cmd, check=True, cwd=repo_root, env=run_env)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    config = load_config(config_path)
    model_entries = select_model_entries(config, args.model_filter)
    if not model_entries:
        print("No models selected. Nothing to run.")
        return 0

    distributed_info = build_distributed(args)

    for merged_cfg in model_entries:
        spec = build_run_spec(repo_root, merged_cfg, distributed_info)
        run_one(spec, args.dry_run, repo_root)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[runtime_baseline_run_from_config] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

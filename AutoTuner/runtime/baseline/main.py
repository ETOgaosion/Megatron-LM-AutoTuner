import argparse
import json
import logging
import os
from datetime import datetime
from typing import List, Optional

import torch

from AutoTuner.runtime.baseline.launcher import RuntimeLauncher
from AutoTuner.utils.distributed import destroy_distributed, init_distributed_multi_nodes
from AutoTuner.utils.logging import log_rank0, log_with_rank, set_logging_level
from AutoTuner.utils.structs import InputTestCase


def str_to_bool(value: str) -> bool:
    val_processed = value.strip().lower()
    if val_processed == "true":
        return True
    if val_processed == "false":
        return False
    raise argparse.ArgumentTypeError(
        f"Value must be 'true' or 'false', but got '{value}'."
    )


def validate_args(args):
    args.real_test_cases_file = os.path.join(args.test_cases_dir, args.test_cases_file)
    assert os.path.exists(
        args.real_test_cases_file
    ), f"{args.real_test_cases_file} not found"

    args.real_override_model_config_file = os.path.join(
        args.config_dir, args.override_model_config_file
    )
    assert os.path.exists(
        args.real_override_model_config_file
    ), (
        f"{args.real_override_model_config_file} not found, "
        f"please place your override model config file in {args.config_dir}"
    )

    args.real_override_tf_config_file = os.path.join(
        args.config_dir, args.override_tf_config_file
    )
    assert os.path.exists(
        args.real_override_tf_config_file
    ), f"{args.real_override_tf_config_file} not found"

    if args.tp_comm_overlap_cfg is not None:
        candidate = os.path.join(args.config_dir, args.tp_comm_overlap_cfg)
        if os.path.exists(candidate):
            args.real_tp_comm_overlap_cfg = candidate
        else:
            args.real_tp_comm_overlap_cfg = None
    else:
        args.real_tp_comm_overlap_cfg = None

    return args


def parse_distributed_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="tp size of megatron",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="pp size of megatron",
    )
    parser.add_argument(
        "--virtual-pipeline-model-parallel-size",
        type=int,
        required=False,
        default=None,
        help="vpp size of megatron",
    )
    parser.add_argument(
        "--context-parallel-size",
        type=int,
        required=False,
        default=1,
        help="cp size of megatron",
    )
    parser.add_argument(
        "--expert-parallel-size",
        type=int,
        required=False,
        default=1,
        help="ep size of megatron",
    )
    parser.add_argument(
        "--expert-tensor-parallel-size",
        type=int,
        required=False,
        default=1,
        help="etp size of megatron",
    )
    return parser


def parse_args():
    parser = argparse.ArgumentParser(description="Runtime launcher")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        default="Qwen/Qwen3-0.6B",
        help="model name to test",
    )
    parser.add_argument(
        "--test-cases-dir",
        type=str,
        required=False,
        default="AutoTuner/testbench/profile/cases/local/",
        help="Base dir holds the test cases files",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        required=False,
        default="AutoTuner/testbench/profile/configs/local/",
        help="Base dir holds the config files",
    )
    parser.add_argument(
        "--test-cases-file",
        type=str,
        required=True,
        help="Test cases JSON file name in test-cases-dir",
    )
    parser.add_argument(
        "--override-model-config-file",
        type=str,
        required=False,
        default="override_model_config.json",
        help="huggingface model configs to override",
    )
    parser.add_argument(
        "--override-tf-config-file",
        type=str,
        required=False,
        default="override_tf_config.json",
        help="TransformerConfig to override",
    )
    parser.add_argument(
        "--tp-comm-overlap-cfg",
        type=str,
        required=False,
        default="tp_comm_overlap_cfg.yaml",
        help="TP overlap config file name in config-dir (optional)",
    )
    parser.add_argument(
        "--num-test-cases",
        type=int,
        default=None,
        help="Optional number of test cases to run from the start of the list",
    )
    parser.add_argument(
        "--run-one-data",
        action="store_true",
        help="Only run one microbatch for each test case",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum iterations to run per test case",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=3,
        help="Warmup iterations to exclude from MFU/throughput averages",
    )
    parser.add_argument(
        "--share-embeddings-and-output-weights",
        type=str_to_bool,
        default=None,
        metavar="[true|false]",
        help="Tie input embeddings and output weights",
    )
    parser.add_argument(
        "--no-ddp",
        action="store_true",
        help="Disable wrapping model with DDP",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("AUTOTUNER_LOG_LEVEL", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="outputs",
        help="Base output directory. Results are saved to outputs/<timestamp>/<model>/runtime_baseline",
    )

    parser = parse_distributed_args(parser)
    args = parser.parse_args()
    return validate_args(args)


def load_test_cases(args) -> List[InputTestCase]:
    with open(args.real_test_cases_file, "r") as fp:
        json_test_cases = json.load(fp)

    test_cases = []
    for json_test_case in json_test_cases["cases"]:
        test_case = InputTestCase(**json_test_case)
        test_case.tensor_model_parallel_size = args.tensor_model_parallel_size
        test_case.pipeline_model_parallel_size = args.pipeline_model_parallel_size
        test_case.virtual_pipeline_model_parallel_size = (
            args.virtual_pipeline_model_parallel_size
        )
        test_case.context_parallel_size = args.context_parallel_size
        test_case.expert_parallel_size = args.expert_parallel_size
        test_case.expert_tensor_parallel_size = args.expert_tensor_parallel_size
        test_cases.append(test_case)
    return test_cases


def load_override_config(path: str) -> dict:
    with open(path, "r") as fp:
        return json.load(fp)


def build_runtime_output_dir(base_output_dir: str, model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if torch.distributed.is_initialized():
        timestamp_holder = [timestamp if torch.distributed.get_rank() == 0 else None]
        torch.distributed.broadcast_object_list(timestamp_holder, src=0)
        timestamp = timestamp_holder[0]

    output_dir = os.path.join(base_output_dir, timestamp, model_name, "runtime_baseline")
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return output_dir


def main():
    args = parse_args()
    level_name = args.log_level.strip().upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(
            f"Invalid log level: {args.log_level}. Use DEBUG, INFO, WARNING, or ERROR."
        )
    set_logging_level(level)
    log_rank0(
        "runtime baseline start: "
        f"model={args.model_name} "
        f"test_cases={args.real_test_cases_file} "
        f"override_model_cfg={args.real_override_model_config_file} "
        f"override_tf_cfg={args.real_override_tf_config_file} "
        f"tp_comm_overlap_cfg={args.real_tp_comm_overlap_cfg} "
        f"tp={args.tensor_model_parallel_size} "
        f"cp={args.context_parallel_size} "
        f"ep={args.expert_parallel_size} "
        f"etp={args.expert_tensor_parallel_size} "
        f"pp={args.pipeline_model_parallel_size} "
        f"vpp={args.virtual_pipeline_model_parallel_size} "
        f"num_test_cases={args.num_test_cases} "
        f"max_iterations={args.max_iterations} "
        f"warmup_iterations={args.warmup_iterations} "
        f"share_emb={args.share_embeddings_and_output_weights} "
        f"no_ddp={args.no_ddp}"
    )

    init_distributed_multi_nodes(
        tp=args.tensor_model_parallel_size,
        cp=args.context_parallel_size,
        ep=args.expert_parallel_size,
        etp=args.expert_tensor_parallel_size,
        pp=args.pipeline_model_parallel_size,
        vpp=args.virtual_pipeline_model_parallel_size,
    )
    if torch.distributed.is_initialized():
        log_with_rank(
            "distributed initialized: "
            f"rank={torch.distributed.get_rank()} "
            f"world_size={torch.distributed.get_world_size()} "
            f"local_rank={os.getenv('LOCAL_RANK', 'n/a')} "
            f"cuda_device={torch.cuda.current_device()} "
            f"cuda_name={torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
    runtime_output_dir = build_runtime_output_dir(args.output_dir, args.model_name)
    log_rank0(f"runtime baseline output_dir={runtime_output_dir}")

    try:
        test_cases = load_test_cases(args)
        override_model_config = load_override_config(
            args.real_override_model_config_file
        )
        override_tf_config = load_override_config(args.real_override_tf_config_file)

        launcher = RuntimeLauncher(
            model_name=args.model_name,
            test_cases=test_cases,
            override_model_kwargs=override_model_config,
            override_tf_config_kwargs=override_tf_config,
            tp_comm_overlap_cfg=args.real_tp_comm_overlap_cfg,
            share_embeddings_and_output_weights=args.share_embeddings_and_output_weights,
            wrap_with_ddp=not args.no_ddp,
            use_distributed_optimizer=False,
            fix_compute_amount=True,
        )
        metrics_by_test_case = launcher.run_pipeline(
            num_test_cases=args.num_test_cases,
            run_one_data=args.run_one_data,
            max_iterations=args.max_iterations,
            warmup_iterations=args.warmup_iterations,
            output_dir=runtime_output_dir,
        )
        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            with open(os.path.join(runtime_output_dir, "runtime_summary.json"), "w") as fp:
                json.dump(metrics_by_test_case, fp, indent=2)
            with open(os.path.join(runtime_output_dir, "args.json"), "w") as fp:
                json.dump(vars(args), fp, indent=2)
            log_rank0(
                "runtime baseline summary saved to "
                f"{os.path.join(runtime_output_dir, 'runtime_summary.json')}"
            )
    finally:
        if torch.distributed.is_initialized():
            destroy_distributed()


if __name__ == "__main__":
    main()

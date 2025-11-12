import argparse
import json
import os
import sys
import time
from typing import List

import torch

from AutoTuner.testbench.profile.configs.config_struct import (
    PROFILE_MODEL_MAP,
    ProfileConfig,
    ProfileMode,
    TorchProfilerConfig,
)
from AutoTuner.testbench.profile.launcher.get_data_launch import (
    LaunchDataCollectionForOps,
)
from AutoTuner.testbench.profile.launcher.nsys_profile_launch import (
    LaunchNsysProfileForOps,
)
from AutoTuner.testbench.profile.launcher.torch_profile_launch import (
    LaunchTorchProfileForOps,
)
from AutoTuner.utils.distributed import (
    destroy_distributed,
    init_distributed_multi_nodes,
)
from AutoTuner.utils.nvtx import enable_nvtx_profiling
from AutoTuner.utils.structs import InputTestCase


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
    ), f"{args.real_override_model_config_file} not found, please place your override model config file in {args.config_dir}"
    args.real_override_tf_config_file = os.path.join(
        args.config_dir, args.override_tf_config_file
    )
    assert os.path.exists(
        args.real_override_tf_config_file
    ), f"{args.real_override_tf_config_file} not found"

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"output to {args.output_dir}")

    # Validate distributed
    assert os.environ.get("WORLD_SIZE") is not None, "WORLD_SIZE is not set"
    assert os.environ.get("RANK") is not None, "RANK is not set"
    assert os.environ.get("LOCAL_RANK") is not None, "LOCAL_RANK is not set"

    return args


def str_to_bool(value):
    val_processed = value.strip().lower()
    if val_processed == "true":
        return True
    elif val_processed == "false":
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Value must be 'true' or 'false', but got '{value}'."
        )


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
    parser = argparse.ArgumentParser(description="Profile Operators")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        default="Qwen/Qwen3-0.6B",
        help="model name to test",
    )
    # File and directories
    parser.add_argument(
        "--test-cases-dir",
        type=str,
        required=False,
        default="AutoTuner/testbench/profile/cases/",
        help="Base dir holds the test cases files, shall not modify this",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        required=False,
        default="AutoTuner/testbench/profile/configs/local/",
        help="Base dir holds the config files, shall not modify this",
    )
    parser.add_argument(
        "--test-cases-file",
        type=str,
        required=True,
        default="qwen3_0_6b.json",
        help="file in cases folder contains",
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

    # Profile configs
    parser.add_argument(
        "--profile-mode",
        type=int,
        help="0: collect data, 1: nsys profile, 2: torch profiler",
        default=0,
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=2,
        required=False,
        help="warmup op iterations",
    )

    # test choices for flexibility
    parser.add_argument(
        "--test-ops-list",
        type=str,
        nargs="+",
        default=None,
        help="List of operator names (strings) to test. Previously accepted integer indices; now expects operator names. If migrating from older scripts, replace indices with operator names.",
    )
    parser.add_argument("--test-case-idxs", type=int, nargs="+", default=None)

    # output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="outputs",
        help="output directory to save the database results and nsys profile results, actual output_dir is args.output_dir/model_name/profile_mode",
    )

    parser.add_argument(
        "--theoretical-flops",
        type=str_to_bool,
        default=False,
        metavar="[true|false]",
        help="Enable/disable theoretical FLOPS calculation. Default: false",
    )
    parser.add_argument(
        "--theoretical-activations",
        type=str_to_bool,
        default=True,
        metavar="[true|false]",
        help="Enable/disable theoretical activations calculation. Default: true",
    )
    # distributed
    parser = parse_distributed_args(parser)
    args = parser.parse_args()
    args = validate_args(args)
    return args


def handle_test_cases(args) -> List[InputTestCase]:
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


def handle_profile_configs(args) -> ProfileConfig:
    return ProfileConfig(
        args.profile_mode,
        args.warmup_iters,
        args.theoretical_flops,
        args.theoretical_activations,
    )


def handle_model_config(args) -> dict:
    with open(args.real_override_model_config_file, "r") as fp:
        override_model_config = json.load(fp)
    return override_model_config


def handle_tf_config(args) -> dict:
    with open(args.real_override_tf_config_file, "r") as fp:
        override_tf_config = json.load(fp)
    return override_tf_config


def call_launcher(
    args,
    test_cases: List[InputTestCase],
    profile_config: ProfileConfig,
    override_model_config: dict,
    override_tf_config: dict,
):
    launcher_cls = None
    if profile_config.profile_mode == ProfileMode.collect_data:
        launcher_cls = LaunchDataCollectionForOps
    elif profile_config.profile_mode == ProfileMode.nsys_profile:
        launcher_cls = LaunchNsysProfileForOps
    elif profile_config.profile_mode == ProfileMode.torch_profiler:
        launcher_cls = LaunchTorchProfileForOps
    else:
        raise ValueError(f"Unsupported profile mode: {profile_config.profile_mode}")

    launcher_kwargs = {
        "profile_config": profile_config,
        "test_cases": test_cases,
        "model_name": args.model_name,
        "override_model_kwargs": override_model_config,
        "override_tf_config_kwargs": override_tf_config,
    }
    launcher_kwargs["tp_comm_overlap_cfg"] = (
        "AutoTuner/testbench/profile/configs/tp_comm_overlap_cfg.yaml"
    )

    torch_profiler_config_kwargs = {}
    if profile_config.profile_mode == ProfileMode.torch_profiler:
        torch_profiler_config_kwargs = {
            "torch_profiler_config": TorchProfilerConfig(
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=profile_config.warmup_iters,
                    active=1,
                    repeat=1,
                ),
                output_dir=os.path.join(
                    args.output_dir,
                    args.model_name,
                    "torch_profiler",
                    # f"rank_{torch.distributed.get_rank()}",
                ),
            )
        }
        launcher_kwargs.update(torch_profiler_config_kwargs)
    launcher = launcher_cls(**launcher_kwargs)

    if (
        profile_config.profile_mode == ProfileMode.nsys_profile
        or profile_config.profile_mode == ProfileMode.torch_profiler
    ):
        enable_nvtx_profiling()

    if args.test_case_idxs is None and args.test_ops_list is None:
        launcher.run_all_supported_ops(test_case_idxs=None)
    else:
        launcher.run_op_list(args.test_ops_list, args.test_case_idxs)

    if profile_config.profile_mode == ProfileMode.collect_data:
        total_timing_db, total_memory_db = launcher.return_results()
        output_dir = os.path.join(
            args.output_dir,
            args.model_name,
            PROFILE_MODEL_MAP[profile_config.profile_mode],
            f"rank_{torch.distributed.get_rank()}",
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "timing.json"), "a+") as fp:
            json.dump(total_timing_db, fp, indent=4)
        with open(os.path.join(output_dir, "memory_activation.json"), "a+") as fp:
            json.dump(total_memory_db["activations"], fp, indent=4)
        with open(os.path.join(output_dir, "memory_weights.json"), "a+") as fp:
            json.dump(total_memory_db["weights"], fp, indent=4)
        print(f"results dumped to {output_dir}")
    else:
        print("Profiling finished.")
    if torch.distributed.get_rank() == 0:
        with open(
            os.path.join(
                args.output_dir,
                args.model_name,
                PROFILE_MODEL_MAP[profile_config.profile_mode],
                "args.txt",
            ),
            "w",
        ) as fp:
            fp.write(str(args))


if __name__ == "__main__":
    print(f"Parsing args ...")
    args = parse_args()
    print("Initializing distributed ...")
    init_distributed_multi_nodes(
        tp=args.tensor_model_parallel_size,
        pp=args.pipeline_model_parallel_size,
        vpp=args.virtual_pipeline_model_parallel_size,
        cp=args.context_parallel_size,
        ep=args.expert_parallel_size,
        etp=args.expert_tensor_parallel_size,
    )
    print("Distributed initialized.")
    if torch.distributed.get_rank() == 0:
        print(f"Args: {args}")
    test_cases = handle_test_cases(args)
    profile_config = handle_profile_configs(args)
    override_model_config = handle_model_config(args)
    override_tf_config = handle_tf_config(args)
    print("Calling launcher ...")
    try:
        call_launcher(
            args, test_cases, profile_config, override_model_config, override_tf_config
        )
    finally:
        destroy_distributed()

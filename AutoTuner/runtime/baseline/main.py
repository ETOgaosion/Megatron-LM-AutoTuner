import argparse
import json
import os
from typing import List, Optional

import torch

from AutoTuner.runtime.baseline.launcher import RuntimeLauncher
from AutoTuner.utils.distributed import destroy_distributed, init_distributed_multi_nodes
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


def main():
    args = parse_args()

    init_distributed_multi_nodes(
        tp=args.tensor_model_parallel_size,
        cp=args.context_parallel_size,
        ep=args.expert_parallel_size,
        etp=args.expert_tensor_parallel_size,
        pp=args.pipeline_model_parallel_size,
        vpp=args.virtual_pipeline_model_parallel_size,
    )

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
            share_embeddings_and_output_weights=args.share_embeddings_and_output_weights,
            wrap_with_ddp=not args.no_ddp,
            use_distributed_optimizer=False,
            fix_compute_amount=True,
        )
        launcher.run_pipeline(
            num_test_cases=args.num_test_cases,
            run_one_data=args.run_one_data,
            max_iterations=args.max_iterations,
            warmup_iterations=args.warmup_iterations,
        )
    finally:
        if torch.distributed.is_initialized():
            destroy_distributed()


if __name__ == "__main__":
    main()

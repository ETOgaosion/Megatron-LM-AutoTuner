import os

import torch
from torch.cuda import OutOfMemoryError

from AutoTuner.utils.memory_snapshots import (
    MemorySnapshotSampler,
    aggressive_empty_cache,
    enable_memory_visualize,
)
from AutoTuner.utils.structs import InputTestCase

from ..configs.config_struct import MemorySnapshotConfig, ProfileConfig, ProfileMode
from ..op_mapping import OP_TEST_MAPPING
from .launcher import Launcher


class LaunchTorchMemorySnapshotForOps(Launcher):
    def __init__(
        self,
        profile_config: ProfileConfig,
        test_cases: list[InputTestCase],
        model_name: str,
        override_model_kwargs: dict,
        override_tf_config_kwargs: dict,
        memory_snapshot_config: MemorySnapshotConfig,
        fix_compute_amount: bool = True,
        tp_comm_overlap_cfg: str = None,
        tp_comm_buffer_name: str = None,
    ):
        assert (
            profile_config.profile_mode == ProfileMode.torch_memory_snapshot
        ), "Nsys profile should enable torch memory snapshot mode."
        super().__init__(
            profile_config=profile_config,
            test_cases=test_cases,
            model_name=model_name,
            override_model_kwargs=override_model_kwargs,
            override_tf_config_kwargs=override_tf_config_kwargs,
            fix_compute_amount=fix_compute_amount,
            tp_comm_overlap_cfg=tp_comm_overlap_cfg,
            tp_comm_buffer_name=tp_comm_buffer_name,
        )
        self.memory_snapshot_config = memory_snapshot_config
        os.makedirs(self.memory_snapshot_config.output_dir, exist_ok=True)
        self.snapshot_sampler = MemorySnapshotSampler(
            out_dir=self.memory_snapshot_config.output_dir
        )
        enable_memory_visualize(
            trace_alloc_max_entries=100000, context="all", stacks="all"
        )

    def _run_op(self, op_name: str, test_case_idxs: list[int]):
        op_test_class = OP_TEST_MAPPING.get(op_name)
        if op_test_class is None:
            raise ValueError(f"Operator '{op_name}' is not supported.")
        op_class_instance = op_test_class(
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            tp_group=self.tp_group,
            profile_mode=self.profile_config.profile_mode,
            warmup_iters=self.profile_config.warmup_iters,
        )
        print(f"Running operator: {op_name}")
        if test_case_idxs is None:
            test_case_idxs = list(range(len(self.test_cases)))
        test_cases = [self.test_cases[i] for i in test_case_idxs]
        for test_case in test_cases:
            batch_data_generator = self.datasets.get_batch_generator(test_case)
            op_class_instance.run_test(
                test_case,
                batch_data_generator,
                run_one_data=self.profile_config.run_one_data,
            )
        aggressive_empty_cache(force_sync=True)
        return op_class_instance

    def run_op(self, op_name: str, test_case_idxs: list[int]):
        try:
            op_test_class = OP_TEST_MAPPING.get(op_name)
            if op_test_class is None:
                raise ValueError(f"Operator '{op_name}' is not supported.")

            kwargs = {
                "tf_config": self.tf_config,
                "hf_config": self.hf_config,
                "tp_group": self.tp_group,
                "profile_mode": self.profile_config.profile_mode,
                "warmup_iters": self.profile_config.warmup_iters,
            }
            if op_name == "GPTModel":
                from megatron.core.models.gpt.gpt_layer_specs import (
                    get_gpt_layer_with_transformer_engine_spec,
                )

                kwargs["transformer_layer_spec"] = (
                    get_gpt_layer_with_transformer_engine_spec(
                        num_experts=self.tf_config.num_moe_experts,
                        multi_latent_attention=self.tf_config.multi_latent_attention,
                        qk_layernorm=self.tf_config.qk_layernorm,
                        moe_grouped_gemm=self.tf_config.moe_grouped_gemm,
                    )
                )
            op_class_instance = op_test_class(**kwargs)
            print(f"Running operator: {op_name}")
            if test_case_idxs is None:
                test_case_idxs = list(range(len(self.test_cases)))
            test_cases = [self.test_cases[i] for i in test_case_idxs]
            for i in range(self.profile_config.warmup_iters + 1):
                for test_case in test_cases:
                    batch_data_generator = self.datasets.get_batch_generator(test_case)
                    op_class_instance.run_test(
                        test_case,
                        batch_data_generator,
                        run_one_data=self.profile_config.run_one_data,
                    )
            print(
                f"Torch profiling completed, trace saved to {self.memory_snapshot_config.output_dir}."
            )
        except OutOfMemoryError:
            combined_tag = op_name
            self.snapshot_sampler.dump_memory_snapshot(
                tag=combined_tag,
                out_dir=self.memory_snapshot_config.output_dir,
                sub_dir="memory_snapshot",
            )
            print(
                f"Torch memory snapshot completed due to OOM, trace saved to {self.memory_snapshot_config.output_dir}."
            )
        return op_class_instance

    def run_op_list(
        self,
        op_name_list: list[str],
        test_case_idxs: list[int],
    ):
        if op_name_list is None:
            op_name_list = self.all_supported_ops
        aggressive_empty_cache(force_sync=True)
        for i in range(self.profile_config.warmup_iters + 1):
            for op_name in op_name_list:
                print(f"Running operator: {op_name}")
                self.run_op(op_name, test_case_idxs)
        combined_tag = "_".join(op_name_list)
        self.snapshot_sampler.dump_memory_snapshot(
            tag=combined_tag,
            out_dir=self.memory_snapshot_config.output_dir,
            sub_dir="memory_snapshot",
        )
        print(
            f"Torch memory snapshot completed, trace saved to {self.memory_snapshot_config.output_dir}."
        )

    def run_all_supported_ops(self, test_case_idxs: list[int]):
        self.run_op_list(self.all_supported_ops, test_case_idxs)
        print(
            f"Torch memory snapshot completed, trace saved to {self.memory_snapshot_config.output_dir}."
        )

import subprocess
from collections import defaultdict

import torch
from megatron.core import parallel_state as mpu

from AutoTuner.utils.config import (
    get_hf_model_config,
    get_mcore_model_config_from_hf_config,
)
from AutoTuner.utils.model_inputs import DataSets
from AutoTuner.utils.nested_dict import NestedDict
from AutoTuner.utils.structs import InputTestCase
from AutoTuner.utils.tp_overlap import initialize_tp_communicators

from ..configs.config_struct import ProfileConfig
from ..op_mapping import OP_TEST_MAPPING


class Launcher:
    def __init__(
        self,
        profile_config: ProfileConfig,
        test_cases: list[InputTestCase],
        model_name: str,
        override_model_kwargs: dict,
        override_tf_config_kwargs: dict,
        tp_comm_overlap_cfg: str = None,
    ):
        self.model_name = model_name
        self.profile_config = profile_config
        self.hf_config = get_hf_model_config(model_name, **override_model_kwargs)
        self.tf_config = get_mcore_model_config_from_hf_config(
            self.hf_config, **override_tf_config_kwargs
        )
        assert (
            torch.distributed.is_initialized()
        ), f"torch distributed shall be initialized"
        self.tp_group = mpu.get_tensor_model_parallel_group()
        self.test_cases = test_cases
        self.datasets = DataSets(
            self.hf_config,
            self.test_cases,
            use_dynamic_bsz_balance=True,
            vpp_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
        )

        self.all_supported_ops = list(OP_TEST_MAPPING.keys())
        self.tp_comm_overlap_cfg = tp_comm_overlap_cfg

    def run_op(self, op_name: str, test_case_idxs: list[int]):
        op_test_class = OP_TEST_MAPPING.get(op_name)
        if op_test_class is None:
            raise ValueError(f"Operator '{op_name}' is not supported.")
        op_class_instance = op_test_class(
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            tp_group=self.tp_group,
            profile_mode=self.profile_config.profile_mode,
            warmup_iters=self.profile_config.warmup_iters,
            theoretical_flops=self.profile_config.theoretical_flops,
            theoretical_activations=self.profile_config.theoretical_activations,
        )
        if test_case_idxs is None:
            test_case_idxs = list(range(len(self.test_cases)))
        test_cases = [self.test_cases[i] for i in test_case_idxs]
        max_seqlen = max(tc.seqlen for tc in test_cases)
        max_batch_size = max(tc.micro_batch_size for tc in test_cases)
        if (
            mpu.get_tensor_model_parallel_world_size() > 1
            and self.tf_config.tp_comm_overlap
        ):
            initialize_tp_communicators(
                tp_comm_overlap_cfg=self.tp_comm_overlap_cfg,
                seq_length=max_seqlen,
                micro_batch_size=max_batch_size,
                hidden_size=self.hf_config.hidden_size,
            )
        for test_case in test_cases:
            print(f"Running operator: {op_name}, test case: {test_case}")
            batch_data_generator = self.datasets.get_batch_generator(test_case)
            op_class_instance.run_test(test_case, batch_data_generator)
        return op_class_instance

    def run_op_list(self, op_name_list: list[str], test_case_idxs: list[int]):
        if op_name_list is None:
            op_name_list = self.all_supported_ops
        for op_name in op_name_list:
            print(f"Running operator: {op_name}")
            self.run_op(op_name, test_case_idxs)

    def run_all_supported_ops(self, test_case_idxs: list[int]):
        self.run_op_list(self.all_supported_ops, test_case_idxs)

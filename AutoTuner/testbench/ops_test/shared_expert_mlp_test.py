from typing import Dict, Optional

import torch
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.testbench.ops.shared_expert_mlp import SharedExpertMLPForTest
from AutoTuner.testbench.ops_test.moe_layer_test import TestMoELayer
from AutoTuner.testbench.profile.configs.config_struct import ProfileMode
from AutoTuner.utils.memory import MemoryTrackerContext, get_memory_str
from AutoTuner.utils.structs import InputTestCase

try:
    import transformer_engine as te  # pylint: disable=unused-import
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from tensordict import TensorDict


class TestSharedExpertMLP(TestMoELayer):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        profile_mode: int = 0,
        warmup_iters: int = 2,
        theoretical_flops: bool = False,
        theoretical_activations: bool = False,
        tp_comm_overlap_cfg: str = None,
    ):
        super().__init__(
            tf_config=tf_config,
            hf_config=hf_config,
            tp_group=tp_group,
            profile_mode=profile_mode,
            warmup_iters=warmup_iters,
            theoretical_flops=theoretical_flops,
            theoretical_activations=theoretical_activations,
            tp_comm_overlap_cfg=tp_comm_overlap_cfg,
            overral_init=False,
        )
        self.module_name = "SharedExpertMLP"

        # we always use te
        use_te = tf_config.transformer_impl

        if use_te:
            backend: BackendSpecProvider = TESpecProvider()
        else:
            backend = LocalSpecProvider()

        linear_fc1 = backend.column_parallel_linear()
        linear_fc2 = backend.row_parallel_linear()
        activation_func = backend.activation_func()

        mlp = MLPSubmodules(
            linear_fc1=linear_fc1,
            linear_fc2=linear_fc2,
            activation_func=activation_func,
        )
        if profile_mode == ProfileMode.collect_data:
            with MemoryTrackerContext(self.module_name) as memory_tracker_ctx:
                self.op = SharedExpertMLPForTest(
                    tf_config,
                    mlp,
                    gate=False,
                    hook_activation=False,
                )

            detailed_mem_report = memory_tracker_ctx.get_result()

            # 未来实现计算时完成
            estimated_weight_mem_bytes = 0

            estimated_weight_mem_str = get_memory_str(
                estimated_weight_mem_bytes, human_readable=True
            )
            detailed_mem_report["estimated_peak_mem_diff"] = estimated_weight_mem_str
            self.memory_db["weights"][self.module_name] = detailed_mem_report

        else:
            self.op = SharedExpertMLPForTest(
                tf_config,
                mlp,
                gate=False,
                hook_activation=False,
            )

    def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
        return super().prepare_input(test_case, micro_batch)

    @override
    def calc_theoretical_flops(self, test_case: InputTestCase) -> Dict[str, float]:
        """
        TODO: theoretical FLOPS
        """
        return {"forward": 0, "backward": 0}

    @override
    def calc_theoretical_memory(self, test_case: InputTestCase) -> Dict[str, int]:
        """
        TODO: theoretical activation memory
        """
        return {"activations": {"activations": 0}}

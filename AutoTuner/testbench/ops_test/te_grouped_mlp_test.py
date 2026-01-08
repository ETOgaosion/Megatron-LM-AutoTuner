from typing import Dict, Optional

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tensordict import TensorDict
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.utils.structs import InputTestCase

from ..ops.te_grouped_mlp import TEGroupedMLPForTest
from ..profile.configs.config_struct import ProfileMode
from .moe_layer_test import TestMoELayer


class TestTEGroupedMLP(TestMoELayer):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        profile_mode: int = 0,
        warmup_iters: int = 2,
        profile_iters: int = 2,
        layer_number: int = 1,
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
            profile_iters=profile_iters,
            layer_number=layer_number,
            theoretical_flops=theoretical_flops,
            theoretical_activations=theoretical_activations,
            tp_comm_overlap_cfg=tp_comm_overlap_cfg,
        )
        self.moe_layer = self.op
        assert self.moe_layer is not None
        self.module_name = "TEGroupedMLP"

        # if you trigger this assertion, please set num moe_experts
        assert tf_config.num_moe_experts is not None
        num_local_experts = (
            tf_config.num_moe_experts
            // parallel_state.get_expert_model_parallel_world_size()
        )

        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=tf_config.num_moe_experts, moe_grouped_gemm=True
        )

        if profile_mode == ProfileMode.collect_data:
            with MemoryTrackerContext(self.module_name) as memory_tracker_ctx:
                self.op = TEGroupedMLPForTest(
                    num_local_experts=num_local_experts,
                    config=tf_config,
                    submodules=transformer_layer_spec.submodules.mlp.submodules.experts.submodules,
                    pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
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
            self.op = TEGroupedMLPForTest(
                num_local_experts=num_local_experts,
                config=tf_config,
                submodules=transformer_layer_spec.submodules.mlp.submodules.experts.submodules,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
                hook_activation=False,
            )

    def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
        (hidden_states,) = super().prepare_input(test_case, micro_batch)
        if (
            self.moe_layer.training
            and self.moe_layer.attn_tp_group.size() > 1
            and not self.moe_layer.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism "
                "are enabled without also enabling sequence parallelism."
            )
        hidden_states, probs, _ = self.moe_layer.router_and_preprocess(hidden_states)
        dispatched_input, probs = self.moe_layer.dispatch(hidden_states, probs)
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.moe_layer.token_dispatcher.dispatch_postprocess(
                dispatched_input, probs
            )
        )
        return (dispatched_input, tokens_per_expert, permuted_probs)

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

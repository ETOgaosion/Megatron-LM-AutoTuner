from typing import Dict, Optional

import torch
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tensordict import TensorDict
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.utils.memory import MemoryTracker, MemoryTrackerContext, get_memory_str
from AutoTuner.utils.structs import InputTestCase

from ..ops.moe_layer import MoELayerForTest
from ..profile.configs.config_struct import ProfileMode
from .common import TestCommon
from .test_with_hiddens import TestWithHiddenInputs


class TestMoELayer(TestWithHiddenInputs):
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
        overral_init: bool = True,
    ):
        super().__init__(
            hf_config=hf_config,
            tf_config=tf_config,
            profile_mode=profile_mode,
            warmup_iters=warmup_iters,
            profile_iters=profile_iters,
            tp_group=tp_group,
            theoretical_flops=theoretical_flops,
            theoretical_activations=theoretical_activations,
            tp_comm_overlap_cfg=tp_comm_overlap_cfg,
        )
        self.module_name = "MoELayer"

        self.self_attention = SelfAttention(
            tf_config,
            get_gpt_layer_with_transformer_engine_spec(
                multi_latent_attention=tf_config.multi_latent_attention,
                qk_layernorm=tf_config.qk_layernorm,
            ).submodules.self_attention.submodules,
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
        )

        if overral_init:
            if profile_mode == ProfileMode.collect_data:
                with MemoryTrackerContext(self.module_name) as memory_tracker_ctx:
                    self.op = MoELayerForTest(
                        tf_config,
                        model_comm_pgs=ModelCommProcessGroups.use_mpu_process_groups(),
                        hook_activation=False,
                    )

                detailed_mem_report = memory_tracker_ctx.get_result()

                # 未来实现计算时完成
                estimated_weight_mem_bytes = 0

                estimated_weight_mem_str = get_memory_str(
                    estimated_weight_mem_bytes, human_readable=True
                )
                detailed_mem_report["estimated_peak_mem_diff"] = (
                    estimated_weight_mem_str
                )
                self.memory_db["weights"][self.module_name] = detailed_mem_report

            else:
                self.op = MoELayerForTest(
                    tf_config,
                    model_comm_pgs=ModelCommProcessGroups.use_mpu_process_groups(),
                    hook_activation=False,
                )

    @override
    def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
        (
            decoder_input,
            attention_mask,
            rotary_pos_emb,
            packed_seq_params,
            sequence_len_offset,
        ) = super().prepare_input(test_case, micro_batch)

        output_attn, _ = self.self_attention(decoder_input, attention_mask)
        hidden_states = output_attn + decoder_input
        return (hidden_states,)

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

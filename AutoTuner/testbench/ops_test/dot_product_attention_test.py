from typing import Dict, Optional

import torch
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)

# from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tensordict import TensorDict
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.utils.memory import MemoryTrackerContext, get_memory_str
from AutoTuner.utils.structs import InputTestCase

from ..ops.dot_product_attention import TEDotProductAttentionForTest
from ..profile.configs.config_struct import ProfileMode
from .test_with_hiddens import TestWithHiddenInputs


class TestTEDotProductAttention(TestWithHiddenInputs):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        layer_number: int = 1,  # 先写死，日后查证这个的影响
        profile_mode: int = 0,
        warmup_iters: int = 2,
        theoretical_flops: bool = False,
        theoretical_activations: bool = False,
    ):
        super().__init__(
            hf_config=hf_config,
            tf_config=tf_config,
            profile_mode=profile_mode,
            warmup_iters=warmup_iters,
            tp_group=tp_group,
            theoretical_flops=theoretical_flops,
            theoretical_activations=theoretical_activations,
        )
        self.module_name = "TEDotProductAttention"

        self.self_attention = SelfAttention(
            tf_config,
            get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
        )

        if profile_mode == ProfileMode.collect_data:
            with MemoryTrackerContext(self.module_name) as memory_tracker_ctx:
                self.op = TEDotProductAttentionForTest(
                    tf_config, layer_number=layer_number, hook_activation=False
                )  # TODO: 写完理论计算之后将这里的false改为True

            detailed_mem_report = memory_tracker_ctx.get_result()

            # 显然，点积的权重估计为0
            estimated_weight_mem_bytes = 0

            estimated_weight_mem_str = get_memory_str(
                estimated_weight_mem_bytes, human_readable=True
            )
            detailed_mem_report["estimated_peak_mem_diff"] = estimated_weight_mem_str
            self.memory_db["weights"][self.module_name] = detailed_mem_report

        else:
            self.op = TEDotProductAttentionForTest(
                tf_config, layer_number=layer_number, hook_activation=False
            )

    @override
    def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
        micro_batch = micro_batch.to(torch.cuda.current_device())
        micro_batch = micro_batch.contiguous()
        (
            hidden_states,
            attention_mask,
            rotary_pos_emb,
            packed_seq_params,
            sequence_len_offset,
        ) = super().prepare_input(test_case, micro_batch)
        q, k, v = self.self_attention.get_query_key_value_tensors(hidden_states)
        # got sbhd, turn to bshd
        q = q.permute(1, 0, 2, 3).contiguous()
        k = k.permute(1, 0, 2, 3).contiguous()
        v = v.permute(1, 0, 2, 3).contiguous()
        if getattr(packed_seq_params, "qkv_format", "bshd") == "thd":
            q = q.reshape(q.shape[0] * q.shape[1], q.shape[2], q.shape[3])
            k = k.reshape(k.shape[0] * k.shape[1], k.shape[2], k.shape[3])
            v = v.reshape(v.shape[0] * v.shape[1], v.shape[2], v.shape[3])
        return (q, k, v, attention_mask, packed_seq_params)

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

import os
from typing import Dict, Optional

import torch
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.testbench.ops.decoder import DecoderForTest
from AutoTuner.testbench.ops_test.test_with_hiddens import TestWithHiddenInputs
from AutoTuner.testbench.profile.configs.config_struct import ProfileMode
from AutoTuner.utils.memory import MemoryTrackerContext, get_memory_str
from AutoTuner.utils.structs import InputTestCase

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestDecoderWithHiddenInputs(TestWithHiddenInputs):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        profile_mode: int = 0,
        warmup_iters: int = 2,
        profile_iters: int = 2,
        theoretical_flops: bool = False,
        theoretical_activations: bool = False,
        tp_comm_overlap_cfg: str = None,
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

        self.module_name = "Decoder"

        if profile_mode == ProfileMode.collect_data:
            with MemoryTrackerContext("Decoder init") as memory_tracker_ctx:
                self.op = DecoderForTest(
                    tf_config,
                    spec=get_gpt_layer_with_transformer_engine_spec(
                        num_experts=tf_config.num_moe_experts,
                        multi_latent_attention=tf_config.multi_latent_attention,
                        qk_layernorm=tf_config.qk_layernorm,
                        moe_grouped_gemm=tf_config.moe_grouped_gemm,
                    ),
                    hook_activation=(profile_mode == ProfileMode.collect_data),
                )

            detailed_mem_report = memory_tracker_ctx.get_result()

            # TODO: theoretical weight memory
            estimated_weight_mem_bytes = 0
            estimated_weight_mem_str = get_memory_str(
                estimated_weight_mem_bytes, human_readable=True
            )
            detailed_mem_report["estimated_peak_mem_diff"] = estimated_weight_mem_str
            self.memory_db["weights"][self.module_name] = detailed_mem_report

        else:
            self.op = DecoderForTest(
                tf_config,
                spec=get_gpt_layer_with_transformer_engine_spec(
                    num_experts=tf_config.num_moe_experts,
                    multi_latent_attention=tf_config.multi_latent_attention,
                    qk_layernorm=tf_config.qk_layernorm,
                    moe_grouped_gemm=tf_config.moe_grouped_gemm,
                ),
                hook_activation=(profile_mode == ProfileMode.collect_data),
            )

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

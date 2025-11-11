import os
from typing import Dict, Optional
from typing_extensions import override

import torch
from AutoTuner.utils.structs import InputTestCase
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import PretrainedConfig
from AutoTuner.utils.memory import get_memory_str
from AutoTuner.testbench.ops.decoder import DecoderForTest
from AutoTuner.testbench.ops_test.test_with_hiddens import TestWithHiddenInputs
from AutoTuner.testbench.profile.configs.config_struct import ProfileMode
from AutoTuner.utils.memory import MemoryTrackerContext

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestDecoderWithHiddenInputs(TestWithHiddenInputs):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
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

        self.module_name = "Decoder"

        if profile_mode == ProfileMode.collect_data:
            with MemoryTrackerContext("Decoder init") as memory_tracker_ctx:
                self.op = DecoderForTest(tf_config)
            
            detailed_mem_report = memory_tracker_ctx.get_result()

            # TODO: theoretical weight memory
            estimated_weight_mem_bytes = 0
            estimated_weight_mem_str = get_memory_str(estimated_weight_mem_bytes, human_readable=True)
            detailed_mem_report['estimated_peak_mem_diff'] = estimated_weight_mem_str
            self.memory_db["weights"][self.module_name] = detailed_mem_report

        else:
            self.op = DecoderForTest(tf_config)

    @override
    def calc_theoretical_flops(self, test_case: InputTestCase) -> Dict[str, float]:
        """
        TODO: theoretical FLOPS
        """
        return {
            "forward": 0,
            "backward": 0
        }

    @override
    def calc_theoretical_memory(self, test_case: InputTestCase) -> Dict[str, int]:
        """
        TODO: theoretical activation memory
        """
        return {
            "activations": {
                "activations": 0
            }
        }
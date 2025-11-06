import os
from typing import Optional

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import PretrainedConfig

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
    ):
        super().__init__(
            hf_config=hf_config,
            tf_config=tf_config,
            profile_mode=profile_mode,
            warmup_iters=warmup_iters,
            tp_group=tp_group,
        )

        self.module_name = "Decoder"

        with MemoryTrackerContext("Decoder init") as memory_tracker_ctx:
            self.op = DecoderForTest(tf_config)

        if profile_mode == ProfileMode.collect_data:
            self.memory_db["weights"][
                self.module_name
            ] = memory_tracker_ctx.get_result()

import os
from typing import Optional

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from tensordict import TensorDict
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.testbench.ops.decoder import DecoderForTest
from AutoTuner.testbench.ops_test.hiddenstatus_gen import HiddenStatusGenerator
from AutoTuner.utils.structs import InputTestCase

from .common import TestCommon

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestDecoderUsingHSG(TestCommon):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        profile_mode: int = 0,
        warmup_iters: int = 2,
    ):
        super().__init__(
            hf_config=hf_config, profile_mode=profile_mode, warmup_iters=warmup_iters
        )
        # Initialize HiddenStatusGenerator with your own configurations
        self.hiddenstatus_generator = HiddenStatusGenerator(
            tf_config,
            hf_config,
            tp_group=tp_group,
        )

        self.op = DecoderForTest(tf_config)

    # We get inputs for decoder after preprocess
    @override
    def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
        return self.hiddenstatus_generator.prepare_input(test_case, micro_batch)

import os
from typing import Optional

import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tensordict import TensorDict
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.utils.hidden_status_gen import HiddenStatusGenerator
from AutoTuner.utils.structs import InputTestCase

from .common import TestCommon

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestWithHiddenInputs(TestCommon):
    def __init__(
        self,
        # These args are given by launcher
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        profile_mode: int = 0,
        warmup_iters: int = 2,
        # These args are given by user and have default values
        scatter_to_sequence_parallel: bool = True,
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        seq_len_interpolation_factor: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        theoretical_flops: bool = False,
        theoretical_activations: bool = False,
    ):
        super().__init__(
            hf_config=hf_config,
            profile_mode=profile_mode,
            warmup_iters=warmup_iters,
            theoretical_flops=theoretical_flops,
            theoretical_activations=theoretical_activations,
        )
        # Initialize HiddenStatusGenerator with your own configurations
        self.hiddenstatus_generator = HiddenStatusGenerator(
            tf_config,
            hf_config,
            tp_group=tp_group,
            scatter_to_sequence_parallel=scatter_to_sequence_parallel,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            pg_collection=pg_collection,
        )

    # We get inputs for decoder after preprocess
    # Notice that the first element in output is hidden states, and others are for attention mask and rotary embedding
    @override
    def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
        return self.hiddenstatus_generator.prepare_input(test_case, micro_batch)

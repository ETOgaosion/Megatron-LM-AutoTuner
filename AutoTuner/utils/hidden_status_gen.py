import os
from typing import Optional

import torch
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tensordict import TensorDict
from transformers import PretrainedConfig

# from AutoTuner.testbench.ops_test.common import TestCommon
from AutoTuner.testbench.ops_test.preprocess_test import PreprocessForTest
from AutoTuner.utils.model_inputs import get_thd_model_input_from_bshd
from AutoTuner.utils.structs import InputTestCase

os.environ["NVTE_NVTX_ENABLED"] = "1"

# Using PreprocessForTest as GenHidden for HiddenStatusGenerator
GenHidden = PreprocessForTest


class HiddenStatusGenerator:
    def __init__(
        self,
        # These args are given by launcher
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        # These args are given by user and have default values
        scatter_to_sequence_parallel: bool = True,
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        seq_len_interpolation_factor: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        self.genhidden = GenHidden(
            LanguageModelEmbedding(
                config=tf_config,
                vocab_size=hf_config.vocab_size,
                max_sequence_length=hf_config.max_position_embeddings,
                position_embedding_type="rope",
                num_tokentypes=0,
                scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                tp_group=tp_group,
            ),
            RotaryEmbedding(
                kv_channels=tf_config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=tf_config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                rope_scaling=rope_scaling,
                rope_scaling_factor=rope_scaling_factor,
                use_cpu_initialization=tf_config.use_cpu_initialization,
                # cp_group=pg_collection.cp,
                cp_group=None,
            ),
            tf_config,
        )

    # We get inputs for decoder after preprocess
    def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
        micro_batch = micro_batch.to(torch.cuda.current_device())
        micro_batch = micro_batch.contiguous()
        input_ids_rmpad, attention_mask, position_ids_rmpad, packed_seq_params = (
            get_thd_model_input_from_bshd(micro_batch)
        )
        (
            decoder_input,
            rotary_pos_emb,
            sequence_len_offset,
            attention_mask,
            packed_seq_params,
        ) = self.genhidden(
            input_ids_rmpad, position_ids_rmpad, attention_mask, packed_seq_params
        )

        # decoder input is what we need as hidden states, and others are for attention mask and rotary embedding
        return (
            decoder_input,
            attention_mask,
            rotary_pos_emb,
            packed_seq_params,
            sequence_len_offset,
        )

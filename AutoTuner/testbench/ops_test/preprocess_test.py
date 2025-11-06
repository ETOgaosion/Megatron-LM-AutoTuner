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
from typing_extensions import override

from AutoTuner.utils.memory import MemoryTrackerContext
from AutoTuner.utils.model_inputs import get_thd_model_input_from_bshd
from AutoTuner.utils.structs import InputTestCase

from ..ops.preprocess import PreprocessForTest
from ..profile.configs.config_struct import ProfileMode
from .common import TestCommon

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestPreprocess(TestCommon):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        scatter_to_sequence_parallel: bool = True,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        profile_mode: int = 0,
        warmup_iters: int = 2,
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        seq_len_interpolation_factor: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(
            hf_config=hf_config, profile_mode=profile_mode, warmup_iters=warmup_iters
        )
        with MemoryTrackerContext("Preprocess init") as memory_tracker_ctx:
            self.op = PreprocessForTest(
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
        self.module_name = "Preprocess"

        if profile_mode == ProfileMode.collect_data:
            self.memory_db["weights"][
                self.module_name
            ] = memory_tracker_ctx.get_result()

    @override
    def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
        micro_batch = micro_batch.to(torch.cuda.current_device())
        micro_batch = micro_batch.contiguous()
        input_ids_rmpad, attention_mask, position_ids_rmpad, packed_seq_params = (
            get_thd_model_input_from_bshd(micro_batch)
        )
        return input_ids_rmpad, position_ids_rmpad, attention_mask, packed_seq_params

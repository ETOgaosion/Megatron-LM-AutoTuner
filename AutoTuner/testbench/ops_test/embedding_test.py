import os
from typing import Any, Iterator, List, Optional, Tuple

import torch
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.utils.memory import MemoryTracker, MemoryTrackerContext
from AutoTuner.utils.model_inputs import get_thd_model_input_from_bshd
from AutoTuner.utils.nested_dict import NestedDict
from AutoTuner.utils.structs import InputTestCase
from AutoTuner.utils.timing import Timer, TimerContext

from ..ops.embedding import LanguageModelEmbeddingForTest
from ..profile.configs.config_struct import ProfileMode
from .common import TestCommon

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestLanguageModelEmbedding(TestCommon):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        scatter_to_sequence_parallel: bool = True,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        profile_mode: int = 0,
        warmup_iters: int = 2,
    ):
        super().__init__(
            hf_config=hf_config, profile_mode=profile_mode, warmup_iters=warmup_iters
        )
        with MemoryTrackerContext("Embedding init") as memory_tracker_ctx:
            self.op = LanguageModelEmbeddingForTest(
                tf_config,
                hf_config,
                scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                tp_group=tp_group,
                hook_activation=profile_mode == ProfileMode.collect_data,
            )
        self.module_name = "Embedding"

        if profile_mode == ProfileMode.collect_data:
            self.memory_db["weights"][
                self.module_name
            ] = memory_tracker_ctx.get_result()

    @override
    def prepare_input(self, test_case: InputTestCase, batch_data_generator: Iterator):
        batch = next(batch_data_generator)
        batch = batch.to(torch.cuda.current_device())
        batch = batch.contiguous()
        input_ids_rmpad, attention_mask, position_ids_rmpad, packed_seq_params = (
            get_thd_model_input_from_bshd(batch)
        )
        return input_ids_rmpad, position_ids_rmpad

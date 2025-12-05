import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from tensordict import TensorDict
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.utils.memory import MemoryTracker, MemoryTrackerContext, get_memory_str
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
            theoretical_flops=theoretical_flops,
            theoretical_activations=theoretical_activations,
            tp_comm_overlap_cfg=tp_comm_overlap_cfg,
        )
        self.module_name = "Embedding"

        if profile_mode == ProfileMode.collect_data:
            with MemoryTrackerContext(self.module_name) as memory_tracker_ctx:
                self.op = LanguageModelEmbeddingForTest(
                    tf_config,
                    hf_config,
                    scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                    tp_group=tp_group,
                    hook_activation=(profile_mode == ProfileMode.collect_data),
                )

            detailed_mem_report = memory_tracker_ctx.get_result()
            vocab_size = self.hf_config.vocab_size
            hidden_size = self.hf_config.hidden_size
            tp_size = mpu.get_tensor_model_parallel_world_size()
            embedding_weight = self.op.word_embeddings.weight
            bytes_per_param = torch.finfo(embedding_weight.dtype).bits // 8

            estimated_weight_mem_bytes = 0
            if mpu.get_pipeline_model_parallel_rank() == 0:
                estimated_weight_mem_bytes = (
                    (vocab_size // tp_size) * hidden_size * bytes_per_param
                )

            estimated_weight_mem_str = get_memory_str(
                estimated_weight_mem_bytes, human_readable=True
            )
            detailed_mem_report["estimated_peak_mem_diff"] = estimated_weight_mem_str
            self.memory_db["weights"][self.module_name] = detailed_mem_report

        else:
            self.op = LanguageModelEmbeddingForTest(
                tf_config,
                hf_config,
                scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                tp_group=tp_group,
                hook_activation=False,
            )

    @override
    def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
        if test_case.shape == "bshd":
            micro_batch = micro_batch.to(torch.cuda.current_device())
            micro_batch = micro_batch.contiguous()
            input_ids_bshd = micro_batch.get("input_ids")
            position_ids_bshd = micro_batch.get("position_ids")
            return input_ids_bshd, position_ids_bshd
        else:
            micro_batch = micro_batch.to(torch.cuda.current_device())
            micro_batch = micro_batch.contiguous()
            input_ids_rmpad, attention_mask, position_ids_rmpad, packed_seq_params = (
                get_thd_model_input_from_bshd(micro_batch)
            )
            return input_ids_rmpad, position_ids_rmpad

    @override
    def calculate_tokens(
        self, test_case: InputTestCase, micro_batch: TensorDict, inputs: Any
    ) -> int:
        attention_mask = micro_batch["attention_mask"]
        return attention_mask.sum().item()

    @override
    def calc_theoretical_memory(self, test_case: InputTestCase) -> Dict[str, int]:
        """
        Calculate theoretical memory usage from the perspective of a single rank.
        """
        cp_size = test_case.context_parallel_size

        # Calculate activation memory
        if test_case.shape == "bshd":
            # activations: input_mask(dtype=bool), masked_input(dtype=int64)
            activation_mem = (
                test_case.micro_batch_size * test_case.seqlen * 8
                + test_case.micro_batch_size * test_case.seqlen * 1
            )
        else:  # thd
            activation_mem = test_case.max_token_len * 8 + test_case.max_token_len * 1

        activation_mem = activation_mem // cp_size

        return {"activations": {"activations": activation_mem}}

    @override
    def calc_theoretical_flops(self, test_case: InputTestCase) -> Dict[str, float]:
        """
        Calculate theoretical FLOPS from the perspective of a single rank.
        """
        micro_batch_size = test_case.micro_batch_size
        seq_len = test_case.seqlen
        hidden_size = self.hf_config.hidden_size
        cp_size = test_case.context_parallel_size

        local_seq_len = seq_len // cp_size
        forward_flops = micro_batch_size * local_seq_len * hidden_size

        backward_flops = 2 * forward_flops

        return {"forward": forward_flops, "backward": backward_flops}

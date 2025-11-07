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
            vocab_size = self.model_config.vocab_size
            hidden_size = self.model_config.hidden_size
            tp_size = mpu.get_tensor_model_parallel_world_size()
            embedding_weight = self.op.word_embeddings.weight
            bytes_per_param = torch.finfo(embedding_weight.dtype).bits // 8

            estimated_weight_mem_bytes = 0
            if mpu.get_pipeline_model_parallel_rank() == 0:
                estimated_weight_mem_bytes = (vocab_size // tp_size) * hidden_size * bytes_per_param

            estimated_weight_mem_str = get_memory_str(estimated_weight_mem_bytes, human_readable=True)
            detailed_mem_report['estimated_peak_mem_diff'] = estimated_weight_mem_str
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
        micro_batch = micro_batch.to(torch.cuda.current_device())
        micro_batch = micro_batch.contiguous()
        input_ids_rmpad, attention_mask, position_ids_rmpad, packed_seq_params = (
            get_thd_model_input_from_bshd(micro_batch)
        )
        return input_ids_rmpad, position_ids_rmpad

    @override
    def calc_theoretical_memory(self, test_case: InputTestCase) -> Dict[str, int]:
        """
        Calculate theoretical memory usage from the perspective of a single rank.
        """
        # Get dimensions and configuration
        hidden_size = self.model_config.hidden_size
        micro_batch_size = test_case.micro_batch_size
        seq_len = test_case.seqlen
        dtype = getattr(self.model_config, "dtype", torch.float16)
        bytes_per_param = torch.tensor([], dtype=dtype).element_size()
        
        # Get all parallel parameters
        tp_size = test_case.tensor_model_parallel_size
        sp_is_enabled = self.op.config.sequence_parallel
        cp_size = test_case.context_parallel_size
        
        # Calculate activation memory
        activation_mem = micro_batch_size * seq_len * hidden_size * bytes_per_param
        if sp_is_enabled:
            activation_mem = activation_mem // tp_size

        if cp_size > 1:
            activation_mem = activation_mem // cp_size
        return {
            "activations": {
                "activations": activation_mem
            }
        }

    @override
    def calc_theoretical_flops(self, test_case: InputTestCase) -> Dict[str, float]:
        """
        Calculate theoretical FLOPS from the perspective of a single rank.
        """
        micro_batch_size = test_case.micro_batch_size
        seq_len = test_case.seqlen
        hidden_size = self.model_config.hidden_size
        cp_size = test_case.context_parallel_size

        local_seq_len = seq_len // cp_size
        forward_flops = micro_batch_size * local_seq_len * hidden_size
        
        backward_flops = 2 * forward_flops
        
        return {
            "forward": forward_flops,
            "backward": backward_flops
        }
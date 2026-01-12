import os
from typing import Any, Dict, Optional

import torch
from megatron.core import parallel_state as mpu
from megatron.core.models.common.embeddings.language_model_cpu_embedding import (
    LanguageModelCPUEmbedding,
)
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tensordict import TensorDict
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.utils.memory import MemoryTrackerContext, get_memory_str
from AutoTuner.utils.model_inputs import get_thd_model_input_from_bshd
from AutoTuner.utils.structs import InputTestCase

from ..ops.preprocess_enhanced import PreprocessEnhancedForTest
from ..profile.configs.config_struct import ProfileMode
from .common import TestCommon

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestPreprocessEnhanced(TestCommon):
    """Test class for PreprocessEnhanced with CPU embedding support."""

    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        scatter_to_sequence_parallel: bool = True,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        profile_mode: int = 0,
        warmup_iters: int = 2,
        profile_iters: int = 2,
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        seq_len_interpolation_factor: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        theoretical_flops: bool = False,
        theoretical_activations: bool = False,
        tp_comm_overlap_cfg: str = None,
        use_cpu_embedding: bool = False,
    ):
        super().__init__(
            tf_config=tf_config,
            hf_config=hf_config,
            profile_mode=profile_mode,
            warmup_iters=warmup_iters,
            profile_iters=profile_iters,
            theoretical_flops=theoretical_flops,
            theoretical_activations=theoretical_activations,
            tp_comm_overlap_cfg=tp_comm_overlap_cfg,
        )
        self.module_name = "PreprocessEnhanced"
        self.use_cpu_embedding = use_cpu_embedding

        if profile_mode == ProfileMode.collect_data:
            with MemoryTrackerContext(self.module_name) as memory_tracker_ctx:
                embedding = self._build_embedding(
                    tf_config=tf_config,
                    hf_config=hf_config,
                    scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                    tp_group=tp_group,
                    use_cpu_embedding=use_cpu_embedding,
                )
                self.op = PreprocessEnhancedForTest(
                    embedding=embedding,
                    rotary_pos_emb=self.build_rotary_embedding(
                        tf_config=tf_config,
                        hf_config=hf_config,
                        rotary_percent=rotary_percent,
                        rotary_base=rotary_base,
                        seq_len_interpolation_factor=seq_len_interpolation_factor,
                    ),
                    config=tf_config,
                    hook_activation=(profile_mode == ProfileMode.collect_data),
                )
            detailed_mem_report = memory_tracker_ctx.get_result()
            # For CPU embedding, GPU weight memory is 0
            if use_cpu_embedding:
                estimated_weight_mem_bytes = 0
            else:
                # Regular GPU embedding weight memory
                vocab_size = hf_config.vocab_size
                hidden_size = hf_config.hidden_size
                tp_size = mpu.get_tensor_model_parallel_world_size()
                dtype = next(self.op.embedding.parameters()).dtype
                bytes_per_param = torch.tensor([], dtype=dtype).element_size()
                estimated_weight_mem_bytes = (
                    (vocab_size // tp_size) * hidden_size * bytes_per_param
                )

            estimated_weight_mem_str = get_memory_str(
                estimated_weight_mem_bytes, human_readable=True
            )
            detailed_mem_report["estimated_peak_mem_diff"] = estimated_weight_mem_str
            self.memory_db["weights"][self.module_name] = detailed_mem_report
        else:
            embedding = self._build_embedding(
                tf_config=tf_config,
                hf_config=hf_config,
                scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                tp_group=tp_group,
                use_cpu_embedding=use_cpu_embedding,
            )
            self.op = PreprocessEnhancedForTest(
                embedding=embedding,
                rotary_pos_emb=self.build_rotary_embedding(
                    tf_config=tf_config,
                    hf_config=hf_config,
                    rotary_percent=rotary_percent,
                    rotary_base=rotary_base,
                    seq_len_interpolation_factor=seq_len_interpolation_factor,
                ),
                config=tf_config,
            )

    @staticmethod
    def _build_embedding(
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        scatter_to_sequence_parallel: bool,
        tp_group: Optional[torch.distributed.ProcessGroup],
        use_cpu_embedding: bool,
    ):
        """Build either CPU or GPU embedding based on configuration."""
        if use_cpu_embedding:
            return LanguageModelCPUEmbedding(
                config=tf_config,
                vocab_size=hf_config.vocab_size,
                max_sequence_length=hf_config.max_position_embeddings,
                position_embedding_type="rope",
                scatter_to_sequence_parallel=scatter_to_sequence_parallel,
            )
        else:
            return LanguageModelEmbedding(
                config=tf_config,
                vocab_size=hf_config.vocab_size,
                max_sequence_length=hf_config.max_position_embeddings,
                position_embedding_type="rope",
                num_tokentypes=0,
                scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                tp_group=tp_group,
            )

    @staticmethod
    def build_rotary_embedding(
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        rotary_percent: float,
        rotary_base: float,
        seq_len_interpolation_factor: Optional[float],
    ):
        rope_scaling = getattr(hf_config, "rope_scaling", None)

        if rope_scaling is not None and rope_scaling.get("type", None) == "yarn":
            return YarnRotaryEmbedding(
                kv_channels=tf_config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=tf_config.rotary_interleaved,
                rotary_base=rotary_base,
                scaling_factor=rope_scaling["factor"],
                original_max_position_embeddings=rope_scaling[
                    "original_max_position_embeddings"
                ],
                beta_fast=rope_scaling["beta_fast"],
                beta_slow=rope_scaling["beta_slow"],
                mscale=rope_scaling.get("mscale", 1.0),
                mscale_all_dim=rope_scaling.get("mscale_all_dim", 1.0),
                use_cpu_initialization=tf_config.use_cpu_initialization,
                cp_group=None,
            )
        else:
            return RotaryEmbedding(
                kv_channels=tf_config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=tf_config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                use_cpu_initialization=tf_config.use_cpu_initialization,
                cp_group=None,
            )

    @override
    def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
        micro_batch = micro_batch.to(torch.cuda.current_device())
        micro_batch = micro_batch.contiguous()
        input_ids_rmpad, attention_mask, position_ids_rmpad, packed_seq_params = (
            get_thd_model_input_from_bshd(micro_batch)
        )
        return input_ids_rmpad, position_ids_rmpad, attention_mask, packed_seq_params

    @override
    def calculate_tokens(
        self, test_case: InputTestCase, micro_batch: TensorDict, inputs: Any
    ) -> int:
        attention_mask = micro_batch["attention_mask"]
        return attention_mask.sum().item()

    @override
    def calc_theoretical_flops(self, test_case: InputTestCase) -> Dict[str, float]:
        """Calculate theoretical FLOPS for preprocessing."""
        micro_batch_size = test_case.micro_batch_size
        seq_len = test_case.seqlen
        hidden_size = self.hf_config.hidden_size
        cp_size = test_case.context_parallel_size

        local_seq_len = seq_len // cp_size
        # Embedding lookup is essentially a memory operation, minimal compute
        forward_flops = micro_batch_size * local_seq_len * hidden_size

        backward_flops = 2 * forward_flops

        return {"forward": forward_flops, "backward": backward_flops}

    @override
    def calc_theoretical_memory(self, test_case: InputTestCase) -> Dict[str, int]:
        """Calculate theoretical activation memory."""
        cp_size = test_case.context_parallel_size
        hidden_size = self.hf_config.hidden_size

        if test_case.shape == "bshd":
            # Activation memory for embedding output
            activation_mem = (
                test_case.micro_batch_size * test_case.seqlen * hidden_size * 2
            )  # bf16
        else:  # thd
            activation_mem = test_case.max_token_len * hidden_size * 2  # bf16

        activation_mem = activation_mem // cp_size

        return {"activations": {"activations": activation_mem}}

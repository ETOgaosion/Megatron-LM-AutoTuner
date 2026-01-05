import logging
from typing import Optional

import torch
from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.language_model_cpu_embedding import (
    LanguageModelCPUEmbedding,
)
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor
from transformers import PretrainedConfig

from AutoTuner.utils.memory import ActivationHook, MemoryTracker
from AutoTuner.utils.nvtx import nvtx_decorator, nvtx_range_pop, nvtx_range_push

from .common import CommonOpsForTest

# from AutoTunner.utils.timing import Timer


class LanguageModelCPUEmbeddingForTest(LanguageModelCPUEmbedding, CommonOpsForTest):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        scatter_to_sequence_parallel: bool = True,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        hook_activation=False,
    ):
        LanguageModelCPUEmbedding.__init__(
            self,
            config=tf_config,
            vocab_size=hf_config.vocab_size,
            max_sequence_length=hf_config.max_position_embeddings,
            position_embedding_type="rope",
            num_tokentypes=0,
            scatter_to_sequence_parallel=scatter_to_sequence_parallel,
            tp_group=tp_group,
        )
        CommonOpsForTest.__init__(
            self,
            hook_activation=hook_activation,
            module_name="LanguageModelCPUEmbedding",
            logging_level=logging.INFO,
        )

    # @MemoryTracker.track_decorator()
    # @Timer.time_decorator()
    @nvtx_decorator(message="LanguageModelCPUEmbedding forward")
    def _forward(
        self, input_ids: Tensor, position_ids: Tensor, tokentype_ids: int = None
    ) -> Tensor:
        """Forward pass of the embedding module.

        Args:
            input_ids (Tensor): The input tokens
            position_ids (Tensor): The position id's used to calculate position embeddings
            tokentype_ids (int): The token type ids. Used when args.bert_binary_head is
                set to True. Defaults to None

        Returns:
            Tensor: The output embeddings
        """
        nvtx_range_push(suffix="word_embeddings")
        word_embeddings = self.word_embeddings(input_ids)
        nvtx_range_pop(suffix="word_embeddings")
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = word_embeddings + position_embeddings
        else:
            embeddings = word_embeddings

        if not self.reduce_scatter_embeddings:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            embeddings = embeddings.transpose(0, 1).contiguous()

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            # [b s h] -> [s b h] (So that it can be added with embeddings)
            tokentype_embedding = self.tokentype_embeddings(tokentype_ids).permute(
                1, 0, 2
            )
            embeddings = embeddings + tokentype_embedding
        else:
            assert self.tokentype_embeddings is None

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.config.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.config.sequence_parallel:
            nvtx_range_push(suffix="sequence_parallel")
            if not self.reduce_scatter_embeddings and self.scatter_to_sequence_parallel:
                embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                    embeddings, group=self.tp_group
                )
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if (
                self.config.clone_scatter_output_in_embedding
                and self.scatter_to_sequence_parallel
            ):
                embeddings = embeddings.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
            nvtx_range_pop(suffix="sequence_parallel")
        else:
            nvtx_range_push(suffix="dropout")
            embeddings = self.embedding_dropout(embeddings)
            nvtx_range_pop(suffix="dropout")

        return embeddings

    def forward(
        self, input_ids: Tensor, position_ids: Tensor, tokentype_ids: int = None
    ) -> Tensor:
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(input_ids, position_ids, tokentype_ids)
        return ret

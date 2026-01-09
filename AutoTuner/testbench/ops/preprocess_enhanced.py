import logging

import torch
import torch.nn as nn
from megatron.core.models.common.embeddings.language_model_cpu_embedding import (
    LanguageModelCPUEmbedding,
)
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_decorator, nvtx_range_pop, nvtx_range_push
from torch import Tensor

from .common import CommonOpsForTest


# Preprocess Enhanced = embedding (CPU or GPU) + rope
# Supports both regular and CPU embeddings
class PreprocessEnhancedForTest(nn.Module, CommonOpsForTest):
    """Enhanced Preprocess module with CPU embedding support.

    This module supports both regular (GPU) and CPU embeddings, determined by the
    type of embedding passed in during initialization.
    """

    def __init__(
        self,
        embedding: LanguageModelEmbedding,  # Can be LanguageModelEmbedding or LanguageModelCPUEmbedding
        rotary_pos_emb: RotaryEmbedding,
        config: TransformerConfig,
        hook_activation: bool = False,
    ):
        nn.Module.__init__(self)
        CommonOpsForTest.__init__(
            self,
            hook_activation=hook_activation,
            module_name="PreprocessEnhanced",
            logging_level=logging.INFO,
        )
        self.embedding = embedding
        self.rotary_pos_emb = rotary_pos_emb
        self.config = config
        self.is_cpu_embedding = isinstance(embedding, LanguageModelCPUEmbedding)

    @nvtx_decorator(message="PreprocessEnhanced forward")
    def _forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        packed_seq_params: PackedSeqParams,
    ):
        """Forward pass of the enhanced preprocess module.

        Args:
            input_ids: Input token IDs (can be on GPU or CPU)
            position_ids: Position IDs (can be on GPU or CPU)
            attention_mask: Attention mask tensor
            packed_seq_params: Packed sequence parameters

        Returns:
            Tuple of (decoder_input, rotary_pos_emb, sequence_len_offset, attention_mask, packed_seq_params)
        """
        rotary_pos_emb = None

        # Embedding forward pass
        # For CPU embeddings, input can be on GPU and will be handled internally
        nvtx_range_push(suffix="embedding")
        if self.is_cpu_embedding:
            nvtx_range_push(suffix="cpu_embedding")
            decoder_input = self.embedding(input_ids, position_ids)
            nvtx_range_pop(suffix="cpu_embedding")
        else:
            nvtx_range_push(suffix="gpu_embedding")
            decoder_input = self.embedding(input_ids, position_ids)
            nvtx_range_pop(suffix="gpu_embedding")
        nvtx_range_pop(suffix="embedding")

        # Rotary positional embeddings (assuming using rope)
        if not self.config.multi_latent_attention:
            nvtx_range_push(suffix="rotary_embedding")
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_context=None,
                transformer=None,
                transformer_input=decoder_input,
                transformer_config=self.config,
                packed_seq_params=packed_seq_params,
            )

            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd",
            )
            nvtx_range_pop(suffix="rotary_embedding")

        sequence_len_offset = None

        return (
            decoder_input,
            rotary_pos_emb,
            sequence_len_offset,
            attention_mask,
            packed_seq_params,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> Tensor:
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(
                input_ids, position_ids, attention_mask, packed_seq_params
            )
        return ret

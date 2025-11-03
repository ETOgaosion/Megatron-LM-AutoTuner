import logging

import torch
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_decorator, nvtx_range_pop, nvtx_range_push
from torch import Tensor

from .common import CommonOpsForTest


# Preprocess = embedding + rope
# Preprocess has no operator in megatron, so we do not inherit from it
class PreprocessForTest(CommonOpsForTest):
    def __init__(
        self,
        embedding: LanguageModelEmbedding,
        rotary_pos_emb: RotaryEmbedding,
        config: TransformerConfig,
    ):
        super().__init__(
            hook_activation=False,
            module_name="Preprocess",
            logging_level=logging.INFO,
        )
        self.embedding = embedding
        self.rotary_pos_emb = rotary_pos_emb
        self.config = config

    @nvtx_decorator(message="Preprocess forward")
    def _forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> Tensor:
        # need decorator to track memory and time
        nvtx_range_push(suffix="embedding")
        decoder_input = self.embedding(input_ids, position_ids)
        nvtx_range_pop(suffix="embedding")

        # we assume using rope
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

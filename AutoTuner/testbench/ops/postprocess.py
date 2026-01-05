import logging
from typing import Optional, Union

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.inference.contexts.base_context import BaseInferenceContext
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
    roll_tensor,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_decorator, nvtx_range_pop, nvtx_range_push
from torch import Tensor

from .common import CommonOpsForTest

try:
    from megatron.core.extensions.transformer_engine import te_parallel_cross_entropy
except:
    te_parallel_cross_entropy = None

from megatron.core import tensor_parallel
from megatron.core.utils import (
    WrappedTensor,
    make_viewless_tensor,
)


class PostprocessForTest(torch.nn.Module, CommonOpsForTest):
    def __init__(
        self,
        tf_config: TransformerConfig,
        share_embeddings_and_output_weights: bool = False,
        mtp: MultiTokenPredictionBlock = None,
        post_process: bool = True,
        mtp_process: bool = False,
        output_layer: Optional[tensor_parallel.ColumnParallelLinear] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        embedding: LanguageModelEmbedding = None,
        hook_activation: bool = False,
    ):
        torch.nn.Module.__init__(self)
        CommonOpsForTest.__init__(
            self,
            hook_activation=hook_activation,
            module_name="Postprocess",
            logging_level=logging.INFO,
        )
        self.tf_config = tf_config
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.mtp = mtp
        self.post_process = post_process
        self.mtp_process = mtp_process
        self.pre_process = (
            mtp_process  # pre_process and mtp_process are equivalent in this context
        )
        self.output_layer = output_layer
        self.cp_group = cp_group
        self.pg_collection = pg_collection
        self.embedding = embedding

    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the embedding weight or output logit weights when share input embedding and
        output weights set to True or when use Multi-Token Prediction (MTP) feature.

        Returns:
            Tensor: During pre processing or MTP process it returns the input embeddings weight.
            Otherwise, during post processing it returns the final output layers weight.
        """
        if self.pre_process or self.mtp_process:
            # Multi-Token Prediction (MTP) need both embedding layer and output layer.
            # So there will be both embedding layer and output layer in the mtp process stage.
            # In this case, if share_embeddings_and_output_weights is True, the shared weights
            # will be stored in embedding layer, and output layer will not have any weight.
            assert hasattr(
                self, "embedding"
            ), f"embedding is needed in this pipeline stage, but it is not initialized."
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    @nvtx_decorator(message="Postprocess forward")
    def _forward(
        self,
        hidden_states: Tensor,
        input_ids,
        position_ids,
        rotary_pos_emb,
        mtp_in_postprocess=None,
        attention_mask=None,
        packed_seq_params=None,
        extra_block_kwargs=None,
    ):
        # if isinstance(hidden_states, WrappedTensor):
        #     hidden_states = hidden_states.unwrap()
        # hidden_states = make_viewless_tensor(
        #     inp=hidden_states, requires_grad=True, keep_graph=True
        # )

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        nvtx_range_push(suffix="mtp")
        if mtp_in_postprocess:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=None,  # Training: always None
                rotary_pos_sin=None,  # Training: always None
                packed_seq_params=packed_seq_params,  # Pass packed_seq_params to support THD format
                sequence_len_offset=None,  # Training: always None
                embedding=self.embedding,
                **(extra_block_kwargs or {}),
            )
        nvtx_range_pop(suffix="mtp")

        nvtx_range_push(suffix="output layer")
        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=False
        )
        nvtx_range_pop(suffix="output layer")

        return logits
        # return loss

    def forward(
        self,
        mtp_in_postprocess: bool,
        input_ids: Optional[Tensor],
        position_ids: Optional[Tensor],
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        rotary_pos_emb: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(
                input_ids=input_ids,
                position_ids=position_ids,
                mtp_in_postprocess=mtp_in_postprocess,
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
            )
        return ret

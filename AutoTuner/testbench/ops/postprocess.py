import logging
from typing import Optional

import torch
from megatron.core.utils import nvtx_decorator, nvtx_range_pop, nvtx_range_push
from torch import Tensor

from .common import CommonOpsForTest
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
    roll_tensor,
)
try:
    from megatron.core.extensions.transformer_engine import te_parallel_cross_entropy
except:
    te_parallel_cross_entropy = None
    
from megatron.core.utils import (
    WrappedTensor,
    make_viewless_tensor,
)

from megatron.core import tensor_parallel

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
        hook_activation=False,
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
        self.pre_process = mtp_process  # pre_process and mtp_process are equivalent in this context
        self.output_layer = output_layer
        self.cp_group = cp_group
        self.pg_collection = pg_collection
        self.embedding = embedding
    # copy from LanguageModule
    # @nvtx_decorator(message="Postprocess loss")
    def compute_language_model_loss(self, labels: Tensor, logits: Tensor) -> Tensor:
        """Computes the language model loss (Cross entropy across vocabulary)

        Args:
            labels (Tensor): The labels of dimension [batch size, seq length] or [total_tokens] for THD
            logits (Tensor): The final logits returned by the output layer of the transformer model

        Returns:
            Tensor: Loss tensor of dimensions [batch size, sequence_length] or [total_tokens] for THD
        """
        # Handle both BSHD format (2D) and THD format (1D)
        is_thd_format = labels.dim() == 1
        
        if not is_thd_format:
            # [b s] => [s b] only for BSHD format
            labels = labels.transpose(0, 1).contiguous()
        
        if self.tf_config.cross_entropy_loss_fusion:
            if self.tf_config.cross_entropy_fusion_impl == 'te':
                if te_parallel_cross_entropy is not None:
                    if not is_thd_format:
                        # as_strided only works for 2D tensors
                        labels = torch.as_strided(labels, labels.size(), (labels.size()[1], 1))
                    loss = te_parallel_cross_entropy(
                        logits, labels, self.pg_collection.tp, False  # is_cg_capturable=False for training
                    )
                else:
                    raise RuntimeError("Trying to use a TE block when it's not present.")
            elif self.tf_config.cross_entropy_fusion_impl == 'native':
                loss = fused_vocab_parallel_cross_entropy(logits, labels, self.pg_collection.tp)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(logits, labels)

        if not is_thd_format:
            # [s b] => [b, s] only for BSHD format
            loss = loss.transpose(0, 1).contiguous()
        return loss

    # copy form gpt_model
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
                self, 'embedding'
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
        labels,
        rotary_pos_emb,
        mtp_in_postprocess=None,
        attention_mask=None,
        packed_seq_params=None,
        extra_block_kwargs=None,
    ):
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )
        
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
            hidden_states, weight=output_weight,runtime_gather_output = False
        )
        nvtx_range_pop(suffix="output layer")

        return logits
        # return loss


    def forward(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        mtp_in_postprocess=None,
        attention_mask=None,
        packed_seq_params=None,
        extra_block_kwargs=None,
    ) -> Tensor:
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                rotary_pos_emb=rotary_pos_emb,
                mtp_in_postprocess=mtp_in_postprocess,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
            )
        return ret

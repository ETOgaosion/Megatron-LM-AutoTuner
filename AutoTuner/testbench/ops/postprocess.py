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



class PostprocessForTest(CommonOpsForTest):
    def __init__(
        self,
        tf_config: TransformerConfig,
        share_embeddings_and_output_weights: bool = False,
        mtp: MultiTokenPredictionBlock = None,
        post_process: bool = True,
        mtp_process: bool = False,
        output_layer=None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        embedding: LanguageModelEmbedding = None,
        hook_activation=False,
    ):
        super().__init__(
            hook_activation=hook_activation,
            module_name="Postprocess",
            logging_level=logging.INFO,
        )
        self.tf_config = tf_config
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.mtp = mtp
        self.post_process = post_process
        self.mtp_process = mtp_process
        self.output_layer = output_layer
        self.cp_group = cp_group
        self.pg_collection = pg_collection
        self.embedding = embedding

    # copy from LanguageModule
    # @nvtx_decorator(message="Postprocess loss")
    def compute_language_model_loss(self, labels: Tensor, logits: Tensor) -> Tensor:
        """Computes the language model loss (Cross entropy across vocabulary)

        Args:
            labels (Tensor): The labels of dimension [batch size, seq length]
            logits (Tensor): The final logits returned by the output layer of the transformer model

        Returns:
            Tensor: Loss tensor of dimensions [batch size, sequence_length]
        """
        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        
        if self.tf_config.cross_entropy_loss_fusion:
            if self.tf_config.cross_entropy_fusion_impl == 'te':
                if te_parallel_cross_entropy is not None:
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

        # [s b] => [b, s]
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
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        mtp_in_postprocess=None,
        loss_mask=None,
        attention_mask=None,
        packed_seq_params=None,
        extra_block_kwargs=None,
    ):
        if not self.post_process:
            return hidden_states
        
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
                packed_seq_params=packed_seq_params,
                sequence_len_offset=None,  # Training: always None
                embedding=self.embedding,
                **(extra_block_kwargs or {}),
            )
        nvtx_range_pop(suffix="mtp")
        
        nvtx_range_push(suffix="loss computation")
        if self.mtp_process:
            
            mtp_labels = labels.clone()
            hidden_states_list = torch.chunk(hidden_states, 1 + self.tf_config.mtp_num_layers, dim=0)
            hidden_states = hidden_states_list[0]
            
            if loss_mask is None:
                loss_mask = torch.ones_like(mtp_labels)
            
            mtp_loss_scale = self.tf_config.mtp_loss_scaling_factor / self.tf_config.mtp_num_layers
            
            for mtp_layer_number in range(self.tf_config.mtp_num_layers):
                # output
                mtp_logits, _ = self.output_layer(
                    hidden_states_list[mtp_layer_number + 1],
                    weight=output_weight,
                )
                # Calc loss for the current Multi-Token Prediction (MTP) layers.
                mtp_labels, _ = roll_tensor(mtp_labels, shifts=-1, dims=-1, cp_group=self.cp_group)
                loss_mask, num_tokens = roll_tensor(
                    loss_mask, shifts=-1, dims=-1, cp_group=self.cp_group
                )
                mtp_loss = self.compute_language_model_loss(mtp_labels, mtp_logits)
                mtp_loss = loss_mask * mtp_loss
                if self.training:
                    # after moving loss logging to loss_func in pretrain_gpt.py
                    MTPLossLoggingHelper.save_loss_to_tracker(
                        torch.sum(mtp_loss) / num_tokens,
                        mtp_layer_number,
                        self.tf_config.mtp_num_layers,
                        avg_group=parallel_state.get_data_parallel_group(
                            with_context_parallel=True
                        ),
                    )
                
                if self.tf_config.calculate_per_token_loss:
                    hidden_states = MTPLossAutoScaler.apply(
                        hidden_states, mtp_loss_scale * mtp_loss
                    )
                else:
                    hidden_states = MTPLossAutoScaler.apply(
                        hidden_states, mtp_loss_scale * mtp_loss / num_tokens
                    )
        nvtx_range_pop(suffix="loss computation")
        
        nvtx_range_push(suffix="output layer")
        logits, _ = self.output_layer(
            hidden_states, weight=output_weight
        )
        nvtx_range_pop(suffix="output layer")

        loss = self.compute_language_model_loss(labels, logits)

        return loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        mtp_in_postprocess=None,
        loss_mask=None,
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
                loss_mask=loss_mask,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
            )
        return ret
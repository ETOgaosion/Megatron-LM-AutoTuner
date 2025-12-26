import logging
from contextlib import nullcontext
from typing import Optional, Union

from megatron.core.transformer.identity_op import IdentityOp
import torch
from megatron.core import tensor_parallel
from megatron.core.inference.contexts.base_context import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerLayerSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import (
    WrappedTensor,
    make_viewless_tensor,
    nvtx_decorator,
    nvtx_range_pop,
    nvtx_range_push,
)
from torch import Tensor

from .common import CommonOpsForTest


class TransformerLayerForTest(CommonOpsForTest, TransformerLayer):
    def __init__(
        self,
        tf_config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        hook_activation: bool = False,
    ):
        TransformerLayer.__init__(
            self,
            config=tf_config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        CommonOpsForTest.__init__(
            self,
            hook_activation=hook_activation,
            module_name="TransformerLayer",
            logging_level=logging.INFO,
        )

    @nvtx_decorator(message="TransformerLayer forward")
    def _forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        packed_seq_params: Optional[object] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:

        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )

        nvtx_range_push(suffix="Attention Layer")
        hidden_states, context = self._forward_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            context=context,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            rotary_pos_cos_sin=None,
            attention_bias=None,
            packed_seq_params=packed_seq_params,
        )
        nvtx_range_pop(suffix="Attention Layer")
        
        nvtx_range_push(suffix="Mlp Layer")
        output = self._forward_mlp(hidden_states)
        nvtx_range_pop(suffix="Mlp Layer")

        return output
    
    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        rotary_pos_emb: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        dynamic_inference_decode_only: Optional[bool] = None,
    ):
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                packed_seq_params,
                context=context,
                context_mask=context_mask,
            )
        return ret

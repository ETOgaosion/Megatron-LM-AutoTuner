import logging
from contextlib import nullcontext
from typing import Optional, Tuple, Union

import torch
from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import TEFusedMLP
from megatron.core.inference.contexts.base_context import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    WrappedTensor,
    make_viewless_tensor,
    nvtx_decorator,
    nvtx_range_pop,
    nvtx_range_push,
)
from torch import Tensor

from .common import CommonOpsForTest


class MLPDenseForTest(CommonOpsForTest, TEFusedMLP):
    def __init__(
        self,
        tf_config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: int = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        hook_activation=False,
    ):
        TEFusedMLP.__init__(
            self,
            config=tf_config,
            submodules=submodules,
            is_expert=is_expert,
            input_size=input_size,
            ffn_hidden_size=ffn_hidden_size,
            tp_group=tp_group,
        )
        CommonOpsForTest.__init__(
            self,
            hook_activation=hook_activation,
            module_name="MLPDense",
            logging_level=logging.INFO,
        )

    @nvtx_decorator(message="MLP forward")
    def _forward(self, hidden_states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:

        if self._fused_impl is None:
            self._fused_impl = (self._make_fused_impl(),)

        # Apply fused impl
        nvtx_range_push(suffix="MLP Layer Fused Impl")
        out = self._fused_impl[0](hidden_states)
        nvtx_range_pop(suffix="MLP Layer Fused Impl")

        # Return bias tensor if requested
        bias = None
        if self.linear_fc2.te_return_bias:
            bias = self.linear_fc2.bias
            if isinstance(bias, torch.Tensor) and bias.numel() == 0:
                bias = None

        return out, bias

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
    ) -> Tuple[Tensor, Optional[Tensor]]:
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(
                hidden_states,
            )
            return ret

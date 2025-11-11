import logging
from contextlib import nullcontext
from typing import Optional, Union

import torch
from megatron.core import tensor_parallel
from megatron.core.enums import Fp8Recipe
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts.base_context import BaseInferenceContext
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_block import (
    TransformerBlock,
)
from megatron.core.transformer.transformer_config import TransformerConfig

# from tests.unit_tests.test_utilities import Utils
from megatron.core.utils import (
    WrappedTensor,
    make_viewless_tensor,
    nvtx_decorator,
    nvtx_range_pop,
    nvtx_range_push,
)
from torch import Tensor

from .common import CommonOpsForTest


class DecoderForTest(TransformerBlock, CommonOpsForTest):
    def __init__(
        self,
        config: TransformerConfig,
    ):
        TransformerBlock.__init__(
            self,
            config,
            spec=get_gpt_layer_with_transformer_engine_spec(),
            post_process=False,
        )
        CommonOpsForTest.__init__(
            self,
            hook_activation=False,
            module_name="Decoder",
            logging_level=logging.INFO,
        )
        self.config = config

    @nvtx_decorator(message="Decoder forward")
    def _forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        sequence_len_offset: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """

        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        # For FP4: NVFP4BlockScaling doesn't have delayed scaling, always uses inner context
        if self.config.fp8:
            use_outer_quantization_context = self.config.fp8_recipe == Fp8Recipe.delayed
            use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
            outer_quantization_context = (
                get_fp8_context(self.config)
                if use_outer_quantization_context
                else nullcontext()
            )
        elif self.config.fp4:
            # use_outer_quantization_context = False
            use_inner_quantization_context = True
            outer_quantization_context = nullcontext()
        else:
            # No quantization
            # use_outer_quantization_context = False
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()

        with rng_context, outer_quantization_context:
            # Forward pass.
            nvtx_range_push(suffix="Transformer Layers")
            for l_no, layer in enumerate(self.layers):
                # Get appropriate inner quantization context
                if use_inner_quantization_context:
                    if self.config.fp8:
                        inner_quantization_context = get_fp8_context(
                            self.config, layer.layer_number - 1
                        )
                    elif self.config.fp4:
                        inner_quantization_context = get_fp4_context(
                            self.config, layer.layer_number - 1
                        )
                    else:
                        inner_quantization_context = nullcontext()
                else:
                    inner_quantization_context = nullcontext()
                with self.offload_context, inner_quantization_context:
                    nvtx_range_push(suffix="Transformer Layer")
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset,
                    )
                    nvtx_range_pop(suffix="Transformer Layer")
                if (
                    torch.is_grad_enabled()
                    and self.config.cpu_offloading
                    and self.group_prefetch_offload_commit_async is not None
                ):
                    hidden_states = self.group_prefetch_offload_commit_async(
                        hidden_states
                    )
            nvtx_range_pop(suffix="Transformer Layers")

        return hidden_states

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
    ) -> Tensor:
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                packed_seq_params,
                sequence_len_offset,
            )
        return ret

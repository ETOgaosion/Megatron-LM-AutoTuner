import logging
from typing import Optional, Tuple, Union

import torch
from einops import rearrange
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.contexts.base_context import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import (
    apply_rotary_pos_emb,
)
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    _yarn_get_concentration_factor_from_config,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import HAVE_FUSED_QKV_ROPE, SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    nvtx_decorator,
    nvtx_range_pop,
    nvtx_range_push,
)
from torch import Tensor

from .common import CommonOpsForTest

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    from transformer_engine.pytorch.attention.rope import apply_fused_qkv_rotary_pos_emb

    HAVE_FUSED_QKV_ROPE = True
except ImportError:
    HAVE_FUSED_QKV_ROPE = False


class SelfAttentionForTest(SelfAttention, CommonOpsForTest):
    def __init__(
        self,
        config: TransformerConfig,
        cp_com_type: str = None,
        pg_collection: ProcessGroupCollection = None,
        hook_activation: bool = False,
        submodules: ModuleSpec = None,
    ):
        SelfAttention.__init__(
            self,
            config=config,
            submodules=submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,  # TODO: check the input is THD or BSHD
            cp_comm_type=cp_com_type,
            pg_collection=pg_collection,
        )
        CommonOpsForTest.__init__(
            self,
            hook_activation=hook_activation,
            module_name="SelfAttention",
            logging_level=logging.INFO,
        )

    @nvtx_decorator(message="SelfAttention forward")
    def _forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the attention module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """
        # Check if we need to skip RoPE
        # no_rope is 0-indexed array and self.layer_number is 1-indexed
        no_rope = (
            self.config.no_rope_freq[self.layer_number - 1]
            if self.config.no_rope_freq
            else False
        )
        if no_rope:
            rotary_pos_emb = None

        assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        nvtx_range_push(suffix="qkv")
        split_qkv = (self.attention_type == "cross") or not all(
            [
                not self.config.test_mode,
                self.config.fused_single_qkv_rope,
                inference_context is None,
                packed_seq_params is None,
                (
                    rotary_pos_emb is not None
                    and rotary_pos_emb[0] is not None
                    and rotary_pos_emb[1] is not None
                ),
                not self.config.flash_decode,
                HAVE_FUSED_QKV_ROPE,
                self.q_layernorm is None or isinstance(self.q_layernorm, IdentityOp),
                self.k_layernorm is None or isinstance(self.k_layernorm, IdentityOp),
            ]
        )
        # Check if fused_single_qkv_rope is requested but either unavailable or not
        # supported for the current use case.
        if self.attention_type != "cross":
            assert not (
                self.config.fused_single_qkv_rope and split_qkv
            ), "fused_single_qkv_rope requested but not available/supported for the config."

        qkv_output = self.get_query_key_value_tensors(
            hidden_states, key_value_states, split_qkv=split_qkv
        )
        attn_mask_type = self.attn_mask_type
        block_table = None
        if split_qkv:
            query, key, value = qkv_output
        else:
            mixed_qkv, qkv_split_arg_list = qkv_output
        nvtx_range_pop(suffix="qkv")

        # ===================================================
        # Adjust key, value, and rotary_pos_emb
        # ===================================================
        nvtx_range_push(suffix="adjust_key_value")
        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        nvtx_range_pop(suffix="adjust_key_value")

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        nvtx_range_push(suffix="rotary_pos_emb")
        if rotary_pos_emb is not None and (
            not self.config.flash_decode or inference_context is None
        ):
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if split_qkv:
                if q_pos_emb is not None:
                    # TODO VIJAY: simplify
                    if (
                        inference_context is None
                        or inference_context.is_static_batching()
                    ):
                        query = apply_rotary_pos_emb(
                            query,
                            q_pos_emb,
                            config=self.config,
                            cu_seqlens=cu_seqlens_q,
                            mscale=_yarn_get_concentration_factor_from_config(
                                self.config
                            ),
                            cp_group=self.pg_collection.cp,
                        )
                    else:
                        query = inference_context.apply_rotary_emb_query(
                            query,
                            q_pos_emb,
                            self.config,
                            cu_seqlens_q,
                            self.pg_collection.cp,
                        )
                if k_pos_emb is not None:
                    key = apply_rotary_pos_emb(
                        key,
                        k_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv,
                        mscale=_yarn_get_concentration_factor_from_config(self.config),
                        cp_group=self.pg_collection.cp,
                    )
            else:
                query, key, value = apply_fused_qkv_rotary_pos_emb(
                    mixed_qkv, q_pos_emb, k_pos_emb, qkv_split_arg_list
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        nvtx_range_pop(suffix="rotary_pos_emb")

        # ==================================
        # core attention computation
        # ==================================

        nvtx_range_push(suffix="core_attention")
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                # Static batching attention kernel.
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )

            else:
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = (
                    inference_context.cu_kv_lengths()
                )

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                )
                core_attn_out = rearrange(core_attn_out, "s b h d -> s b (h d)")

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        nvtx_range_pop(suffix="core_attention")

        # =================
        # Output. [sq, b, h]
        # =================

        nvtx_range_push(suffix="linear_proj")
        output, bias = self.linear_proj(core_attn_out)
        nvtx_range_pop(suffix="linear_proj")

        return output, bias

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        return ret

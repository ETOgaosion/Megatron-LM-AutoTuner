import logging

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    nvtx_decorator,
)
from torch import Tensor
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    AttnFuncWithCPAndKVP2P,
)

from TransformerEngine.transformer_engine.pytorch.attention.dot_product_attention.context_parallel import *

from .common import CommonOpsForTest


class AttnFuncWithCPAndKVP2PWrapper:
    def forward(
        self,
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        dropout_p,
        cp_group,
        cp_global_ranks,
        cp_stream,
        cp_comm_type,
        softmax_scale=None,
        qkv_format="bshd",
        attn_mask_type="causal",
        attn_bias_type="no_bias",
        attn_bias=None,
        deterministic=False,
        use_fused_attention=False,
        window_size=None,
        fp8=False,
        fp8_meta=None,
        quantizers=None,
        pad_between_seqs=False,
        use_flash_attn_3=False,
    ):

        args = [
            is_training,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            dropout_p,
            softmax_scale,
            qkv_format,
            attn_mask_type,
            attn_bias_type,
            attn_bias,
            deterministic,
            use_fused_attention,
        ]

        args += [
            fp8,
            fp8_meta,
            cp_group,
            cp_global_ranks,
            cp_stream,
            quantizers,
            pad_between_seqs,
            use_flash_attn_3,
        ]

        return AttnFuncWithCPAndKVP2P.apply(*args)

    __call__ = forward


class AttnFuncWithCPAndKVP2PForTest(AttnFuncWithCPAndKVP2PWrapper, CommonOpsForTest):
    def __init__(
        self,
        config: TransformerConfig,
    ):
        CommonOpsForTest.__init__(
            self,
            hook_activation=False,
            module_name="AttnFuncWithCPAndKVP2P",
            logging_level=logging.INFO,
        )
        self.config = config

    @nvtx_decorator(message="AttnFuncWithCPAndKVP2P forward")
    def _forward(
        self,
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        dropout_p,
        cp_group,
        cp_global_ranks,
        cp_stream,
        cp_comm_type,
        softmax_scale=None,
        qkv_format="bshd",
        attn_mask_type="causal",
        attn_bias_type="no_bias",
        attn_bias=None,
        deterministic=False,
        use_fused_attention=False,
        window_size=None,
        fp8=False,
        fp8_meta=None,
        quantizers=None,
        pad_between_seqs=False,
        use_flash_attn_3=False,
    ):

        args = [
            is_training,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            dropout_p,
            cp_group,
            cp_global_ranks,
            cp_stream,
            cp_comm_type,
            softmax_scale,
            qkv_format,
            attn_mask_type,
            attn_bias_type,
            attn_bias,
            deterministic,
            use_fused_attention,
            window_size,
            fp8,
            fp8_meta,
            quantizers,
            pad_between_seqs,
            use_flash_attn_3,
        ]

        return super().forward(*args)

    def __call__(
        self,
        q,
        k,
        v,
        is_training,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        dropout_p,
        cp_group,
        cp_global_ranks,
        cp_stream,
        cp_comm_type,
        qkv_format="bshd",
    ) -> Tensor:
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            if qkv_format == "bshd":
                attn_mask_type = "causal"
            elif qkv_format == "thd":
                attn_mask_type = "padding_causal"
            ret = self._forward(
                is_training,
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q_padded,
                cu_seqlens_kv_padded,
                dropout_p,
                cp_group,
                cp_global_ranks,
                cp_stream,
                cp_comm_type,
                softmax_scale=None,
                qkv_format=qkv_format,
                attn_mask_type=attn_mask_type,
                attn_bias_type="no_bias",
                attn_bias=None,
                deterministic=False,
                use_fused_attention=False,
                window_size=None,
                fp8=False,
                fp8_meta=None,
                quantizers=None,
                pad_between_seqs=False,
                use_flash_attn_3=False,
            )
        return ret

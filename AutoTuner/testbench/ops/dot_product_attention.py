import torch
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    get_te_version,
    is_te_min_version,
)
from torch import Tensor

from .common import CommonOpsForTest


class TEDotProductAttentionForTest(CommonOpsForTest, TEDotProductAttention):
    # 这个继承顺序只是为了补全方便，实际上没有影响

    def __init__(
        self,
        config: TransformerConfig,
        layer_number,  # layer_number: layer number of the current `DotProductAttention` when multiple such modules are concatenated, for instance in consecutive transformer blocks.
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        attention_type: str = "self",  # default self for above 95% llm's attn type is self
        hook_activation: bool = False,
    ):
        TEDotProductAttention.__init__(
            self, config, layer_number, attn_mask_type, attention_type
        )
        CommonOpsForTest.__init__(
            self,
            hook_activation=hook_activation,
            module_name="TEDotProductAttention",
        )
        self.config = config

    def _forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Forward."""
        packed_seq_kwargs = (
            {
                key: getattr(packed_seq_params, key)
                for key in self.kept_packed_seq_params
            }
            if packed_seq_params is not None
            else {}
        )
        qkv_format = packed_seq_kwargs.get("qkv_format", self.qkv_format)
        attention_bias_kwargs = {}
        if attention_bias is not None:
            assert is_te_min_version("1.2.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 1.2.0 to support"
                "`attention_bias`."
            )
            attention_bias_kwargs = dict(
                core_attention_bias_type="post_scale_bias",
                core_attention_bias=attention_bias,
            )

        # this is for decoding, so commented
        # if attn_mask_type == AttnMaskType.no_mask and self.config.window_size is not None:
        #     if (qkv_format == "bshd" and query.size(1) == 1) or (
        #         qkv_format == "sbhd" and query.size(0) == 1
        #     ):
        #         #  need to change mask type for SWA inference decode stage.
        #         attn_mask_type = AttnMaskType.causal_bottom_right
        if self.te_forward_mask_type:
            if qkv_format == "thd" and is_te_min_version("1.7.0"):
                # thd format uses flash attention with cuDNN kernel which requires is_padding=True,
                # so the only acceptable mask types are `padding_causal` and `padding`. These do not
                # necessarily indicate there are padded tokens in the sequence.
                if attn_mask_type == AttnMaskType.causal:
                    attn_mask_type = AttnMaskType.padding_causal
                elif attn_mask_type == AttnMaskType.no_mask:
                    attn_mask_type = AttnMaskType.padding

            # 这里的调用方法是对的，下面也一样，就是调用TEDotProductionAttention的父类的forward方法
            core_attn_out = super(TEDotProductAttention, self).forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type.name,
                **attention_bias_kwargs,
                **packed_seq_kwargs,
            )
        else:
            core_attn_out = super(TEDotProductAttention, self).forward(
                query,
                key,
                value,
                attention_mask,
                **attention_bias_kwargs,
                **packed_seq_kwargs,
            )

        return core_attn_out

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        packed_seq_params: PackedSeqParams = None,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,  # 先设置默认值，未来可能修改
        attention_bias: Tensor = None,
    ) -> Tensor:
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type,
                attention_bias,
                packed_seq_params,
            )
            return ret

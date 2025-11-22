import logging
from typing import Optional

import torch
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from torch import Tensor
from transformers import PretrainedConfig

from AutoTuner.utils.memory import ActivationHook, MemoryTracker
from AutoTuner.utils.nvtx import nvtx_decorator, nvtx_range_pop, nvtx_range_push

from .common import CommonOpsForTest
from megatron.core.extensions.transformer_engine import TENorm

class LayerNormForTest(torch.nn.Module, CommonOpsForTest):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        hook_activation=False,
    ):
        torch.nn.Module.__init__(self)
        CommonOpsForTest.__init__(
            self,
            hook_activation=hook_activation,
            module_name="LayerNorm",
            logging_level=logging.INFO,
        )

        if tf_config.transformer_impl != "transformer_engine":
            raise ValueError(f"Expected transformer_impl to be 'transformer_engine', got '{tf_config.transformer_impl}'")
        self.norm = TENorm(
            config=tf_config,
            hidden_size=hf_config.hidden_size,
            eps=tf_config.layernorm_epsilon,
        )

    @nvtx_decorator(message="LayerNorm forward")
    def _forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass of LayerNorm.
        Args:
            hidden_states: Input tensor of shape [*, hidden_size]
        Returns:
            Normalized tensor with same shape as input
        """
        return self.norm(hidden_states)

    def forward(self, hidden_states: Tensor) -> Tensor:
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(hidden_states)
        return ret

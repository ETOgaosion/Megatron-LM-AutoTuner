import torch
import torch.nn.functional as F
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.shared_experts import (
    SharedExpertMLP,
    set_tensor_grad_fn_sequence_sr,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_decorator, nvtx_range_pop, nvtx_range_push
from torch import Tensor

from .common import CommonOpsForTest


class SharedExpertMLPForTest(SharedExpertMLP, CommonOpsForTest):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        gate: bool = False,
        hook_activation: bool = False,
    ):
        SharedExpertMLP.__init__(self, config, submodules, gate=gate)
        CommonOpsForTest.__init__(
            self, hook_activation=hook_activation, module_name="SharedExpertMLP"
        )

        if self.config.moe_shared_expert_overlap:
            self._forward = self._forward_all2all_overlap_part
        else:
            self._forward = self._forward_common

    @nvtx_decorator(message="SharedExpertMLPForward_common")
    def _forward_common(self, hidden_states):
        """Forward function"""
        nvtx_range_push(suffix="expert_forward")
        output, _ = super(SharedExpertMLP, self).forward(hidden_states)
        nvtx_range_pop(suffix="expert_forward")
        if self.use_shared_expert_gate:
            nvtx_range_push(suffix="gate")
            logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
            gate_score = torch.nn.functional.sigmoid(logits)
            output = output * gate_score
            nvtx_range_pop(suffix="gate")
        return output

    @nvtx_decorator(message="SharedExpertMLPForward_all2all_overlap")
    def _forward_all2all_overlap_part(self, hidden_states):
        """Forward function"""
        nvtx_range_push(suffix="pre_forward")
        self.pre_forward_comm(hidden_states)
        nvtx_range_pop(suffix="pre_forward")
        self.linear_fc1_forward_and_act(None)

        nvtx_range_push(suffix="fc2")
        self.linear_fc2_forward(None)
        nvtx_range_pop(suffix="fc2")

        nvtx_range_push(suffix="post_forward")
        self.post_forward_comm()
        nvtx_range_pop(suffix="post_forward")

        nvtx_range_push(suffix="gating")
        ret = self.get_output()
        nvtx_range_pop(suffix="gating")
        return ret

    def linear_fc1_forward_and_act(self, overlapped_comm_output=None):
        """
        Do Linear FC1 and activation function forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_fc1_input is not None
        if overlapped_comm_output is not None:
            set_tensor_grad_fn_sequence_sr(
                overlapped_comm_output, torch.iinfo(torch.int).max
            )
        with torch.cuda.stream(self.stream):
            # [s, b, 4 * h/p]
            nvtx_range_push(suffix="fc1")
            intermediate_parallel, bias_parallel = self.linear_fc1(
                self.cached_fc1_input
            )
            nvtx_range_pop(suffix="fc1")
            self.cached_fc1_input = None
            nvtx_range_push(suffix="activation")
            if self.config.bias_activation_fusion:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(
                            intermediate_parallel, bias_parallel
                        )
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
            else:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                if self.config.gated_linear_unit:

                    def glu(x):
                        x = torch.chunk(x, 2, dim=-1)
                        return self.config.activation_func(x[0]) * x[1]

                    intermediate_parallel = glu(intermediate_parallel)
                else:
                    intermediate_parallel = self.activation_func(intermediate_parallel)

            self.cached_fc2_input = intermediate_parallel
            nvtx_range_pop(suffix="activation")

    def forward(self, hidden_states: Tensor) -> Tensor:
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(hidden_states)
        return ret

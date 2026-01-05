from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.activations import squared_relu
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    nvtx_decorator,
    nvtx_range_pop,
    nvtx_range_push,
)

from .common import CommonOpsForTest


class TEGroupedMLPForTest(TEGroupedMLP, CommonOpsForTest):

    def __init__(
        self,
        num_local_experts,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
        hook_activation: bool = False,
    ):
        TEGroupedMLP.__init__(
            self,
            num_local_experts=num_local_experts,
            config=config,
            submodules=submodules,
            model_comm_pgs=model_comm_pgs,
        )
        CommonOpsForTest.__init__(
            self, module_name="TEGroupedMLP", hook_activation=hook_activation
        )

    @nvtx_decorator(message="TEGroupedMLPForward")
    def _forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of TEGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.
            permuted_probs (torch.Tensor): The permuted probs of each token produced by the router.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        tokens_per_expert = tokens_per_expert.tolist()
        if self.config.fp8:
            actual_tokens_per_expert = tokens_per_expert
            permuted_local_hidden_states, tokens_per_expert = self.fp8_padding(
                permuted_local_hidden_states, tokens_per_expert
            )
            permuted_probs, _ = self.fp8_padding(
                permuted_probs.unsqueeze(-1), actual_tokens_per_expert
            )
        else:
            permuted_probs = permuted_probs.unsqueeze(-1)

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(
                original_dtype
            )
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, bias_parallel = self.linear_fc1(
            permuted_local_hidden_states, tokens_per_expert
        )
        nvtx_range_pop(suffix="linear_fc1")

        def bias_act_func(intermediate_parallel, bias_parallel, permuted_probs):
            nvtx_range_push(suffix="activation")
            if self.config.bias_activation_fusion:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                    )
                else:
                    raise ValueError("Only support fusion of swiglu in TEGroupedMLP.")
            elif (
                self.activation_func == squared_relu
                and self.config.use_fused_weighted_squared_relu
            ):
                assert bias_parallel is None
                intermediate_parallel = weighted_squared_relu_impl(
                    intermediate_parallel, permuted_probs
                )
            else:
                if bias_parallel is not None:
                    shape = intermediate_parallel.shape
                    intermediate_parallel = torch.cat(
                        [
                            t + b
                            for t, b in zip(
                                torch.split(
                                    intermediate_parallel.view(-1, shape[-1]),
                                    tokens_per_expert,
                                ),
                                bias_parallel,
                            )
                        ]
                    ).view(shape)
                if self.config.gated_linear_unit:

                    def glu(x):
                        x = torch.chunk(x, 2, dim=-1)
                        return self.config.activation_func(x[0]) * x[1]

                    intermediate_parallel = glu(intermediate_parallel)
                else:
                    intermediate_parallel = self.activation_func(intermediate_parallel)
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * permuted_probs
                intermediate_parallel = intermediate_parallel.to(original_dtype)
            nvtx_range_pop(suffix="activation")
            return intermediate_parallel

        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            intermediate_parallel = self.activation_checkpoint.checkpoint(
                bias_act_func, intermediate_parallel, bias_parallel, permuted_probs
            )
            nvtx_range_push(suffix="linear_fc2")
            output, output_bias = self.linear_fc2(
                intermediate_parallel, tokens_per_expert
            )
            nvtx_range_pop(suffix="linear_fc2")
            self.activation_checkpoint.discard_output_and_register_recompute(output)
        else:
            intermediate_parallel = bias_act_func(
                intermediate_parallel, bias_parallel, permuted_probs
            )
            nvtx_range_push(suffix="linear_fc2")
            output, output_bias = self.linear_fc2(
                intermediate_parallel, tokens_per_expert
            )
            nvtx_range_pop(suffix="linear_fc2")

        # upad and concat the output
        if self.config.fp8:
            output = self.fp8_unpadding(output, actual_tokens_per_expert)

        return output, output_bias

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(
                permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )
        return ret

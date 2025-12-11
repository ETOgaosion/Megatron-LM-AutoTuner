from typing import Optional, Union

import torch
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.transformer_config import TransformerConfig

from .common import CommonOpsForTest

try:
    import transformer_engine as te  # pylint: disable=unused-import
    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from megatron.core import parallel_state, tensor_parallel, utils
from megatron.core.utils import nvtx_decorator, nvtx_range_pop, nvtx_range_push


class MoELayerForTest(MoELayer, CommonOpsForTest):
    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        layer_number: int = 1,
        hook_activation=False,
    ):
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm
        )
        """
            If you trigger the assertion below, then carefully set the tf_config.num_moe_experts.
            You can simply set it at AutoTuner/testbench/profile/configs/local/override_tf_config.json
            Also, please carefully check the value, there might be more strictions to be satisfied.
        """
        assert config.num_moe_experts is not None
        assert config.num_moe_experts % utils.get_pg_size(pg_collection.ep) == 0
        MoELayer.__init__(
            self,
            config,
            submodules=transformer_layer_spec.submodules.mlp.submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
        )
        CommonOpsForTest.__init__(
            self, hook_activation=hook_activation, module_name="MoELayer"
        )

    @nvtx_decorator(message="MoELayer forward")
    def _forward(self, hidden_states: torch.Tensor):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        # we will implement the instrumentation according to the docs above
        if (
            self.training
            and self.attn_tp_group.size() > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # MoE forward: route -> dispatch -> compute -> combine
        def custom_forward(hidden_states):
            nvtx_range_push(suffix="routing")
            hidden_states, probs, residual = self.router_and_preprocess(hidden_states)
            nvtx_range_pop(suffix="routing")
            nvtx_range_push(suffix="dispatch")
            dispatched_input, probs = self.dispatch(hidden_states, probs)
            nvtx_range_pop(suffix="dispatch")
            nvtx_range_push(suffix="expert compute")
            output, shared_expert_output, mlp_bias = self.experts_compute(
                dispatched_input, probs, residual
            )
            nvtx_range_pop(suffix="expert compute")
            nvtx_range_push(suffix="combine")
            output = self.combine(output, shared_expert_output)
            nvtx_range_pop(suffix="combine")
            return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8:
                output, mlp_bias = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            else:
                output, mlp_bias = tensor_parallel.checkpoint(
                    custom_forward, False, hidden_states
                )
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

    def forward(self, hidden_states: torch.Tensor):
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(hidden_states)
        return ret

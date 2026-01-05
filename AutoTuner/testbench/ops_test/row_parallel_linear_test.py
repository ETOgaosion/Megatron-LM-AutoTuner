from typing import Dict, Optional

import torch
import torch.nn.functional as F
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.testbench.profile.configs.config_struct import ProfileMode
from AutoTuner.utils.memory import MemoryTrackerContext, get_memory_str
from AutoTuner.utils.structs import InputTestCase

from ..ops.row_parallel_linear import TERowParallelLinearForTest
from .test_with_hiddens import TestWithHiddenInputs

try:
    import transformer_engine as te  # pylint: disable=unused-import
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    from megatron.core.extensions.kitchen import KitchenSpecProvider

    HAVE_KITCHEN = True
except ImportError:
    HAVE_KITCHEN = False

from megatron.core.fusions.fused_bias_geglu import (
    bias_geglu_impl,
    quick_gelu,
    weighted_bias_quick_geglu_impl,
)
from megatron.core.fusions.fused_bias_swiglu import (
    bias_swiglu_impl,
    weighted_bias_swiglu_impl,
)


class MLPBeforeFc2(MLP):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: int = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(
            config, submodules, is_expert, input_size, ffn_hidden_size, tp_group
        )

    def forward(self, hidden_states, per_token_scale=None):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = (
                    intermediate_parallel * per_token_scale.unsqueeze(-1)
                )
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                elif (
                    self.activation_func == quick_gelu and self.config.gated_linear_unit
                ):
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu with per_token_scale in MLP."
                    )
            else:
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
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.activation_func(x_glu) * (
                        x_linear + self.config.glu_linear_offset
                    )

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = (
                    intermediate_parallel * per_token_scale.unsqueeze(-1)
                )
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        return intermediate_parallel


class TestTERowParallelLinear(TestWithHiddenInputs):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        profile_mode: int = 0,
        warmup_iters: int = 2,
        theoretical_flops: bool = False,
        theoretical_activations: bool = False,
    ):
        # Dense MLP w/ or w/o TE modules.
        backend: BackendSpecProvider = (
            KitchenSpecProvider(fallback=TESpecProvider())
            if tf_config.use_kitchen
            else TESpecProvider()
        )
        linear_fc2 = backend.row_parallel_linear()
        activation_func = backend.activation_func()

        if backend.fuse_layernorm_and_linear():
            linear_fc1 = backend.column_parallel_layer_norm_linear()
            assert linear_fc1 is not None
        else:
            linear_fc1 = backend.column_parallel_linear()

        self.fc1_act = MLPBeforeFc2(
            tf_config, MLPSubmodules(linear_fc1, activation_func, linear_fc2)
        )
        super().__init__(
            hf_config=hf_config,
            tf_config=tf_config,
            profile_mode=profile_mode,
            warmup_iters=warmup_iters,
            tp_group=tp_group,
            theoretical_flops=theoretical_flops,
            theoretical_activations=theoretical_activations,
        )
        self.module_name = "TERowParallelLinear"
        if profile_mode == ProfileMode.collect_data:
            with MemoryTrackerContext(self.module_name) as memory_tracker_ctx:
                self.op = TERowParallelLinearForTest(
                    tf_config.ffn_hidden_size,
                    tf_config.hidden_size,
                    config=tf_config,
                    init_method=tf_config.output_layer_init_method,
                    bias=tf_config.add_bias_linear,
                    tp_group=tp_group,
                    hook_activation=False,
                )

            detailed_mem_report = memory_tracker_ctx.get_result()

            # TODO: theoretical weight memory
            estimated_weight_mem_bytes = 0
            estimated_weight_mem_str = get_memory_str(
                estimated_weight_mem_bytes, human_readable=True
            )
            detailed_mem_report["estimated_peak_mem_diff"] = estimated_weight_mem_str
            self.memory_db["weights"][self.module_name] = detailed_mem_report

        else:
            self.op = TERowParallelLinearForTest(
                tf_config.ffn_hidden_size,
                tf_config.hidden_size,
                config=tf_config,
                init_method=tf_config.output_layer_init_method,
                bias=tf_config.add_bias_linear,
                tp_group=tp_group,
                hook_activation=False,
            )

    def prepare_input(self, test_case, micro_batch):
        (
            decoder_input,
            attention_mask,
            rotary_pos_emb,
            packed_seq_params,
            sequence_len_offset,
        ) = super().prepare_input(test_case, micro_batch)
        return (self.fc1_act(decoder_input),)

    @override
    def calc_theoretical_flops(self, test_case: InputTestCase) -> Dict[str, float]:
        """
        TODO: theoretical FLOPS
        """
        return {"forward": 0, "backward": 0}

    @override
    def calc_theoretical_memory(self, test_case: InputTestCase) -> Dict[str, int]:
        """
        TODO: theoretical activation memory
        """
        return {"activations": {"activations": 0}}

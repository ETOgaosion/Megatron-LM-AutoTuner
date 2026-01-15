"""Test class for GPTModelModuleQueue operator.

This test class is designed for testing the last pipeline stage model
(pre_process=False, post_process=True) with module queue memory optimization.
"""

import os
from typing import Any, Dict, Optional

import torch
from megatron.core import parallel_state as mpu
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tensordict import TensorDict
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.testbench.ops.gpt_model_module_queue import GPTModelModuleQueueForTest
from AutoTuner.utils.memory import MemoryTrackerContext, get_memory_str
from AutoTuner.utils.structs import InputTestCase

from ..profile.configs.config_struct import ProfileMode
from .test_with_hiddens import TestWithHiddenInputs

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestGPTModelModuleQueue(TestWithHiddenInputs):
    """Test class for GPTModelModuleQueue.

    This test class handles the last pipeline stage scenario where:
    - pre_process=False: No embedding, receives hidden states from previous stage
    - post_process=True: Has output layer for final logits
    - Module queue enabled: Memory-efficient CPU offloading of layers
    """

    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        transformer_layer_spec: ModuleSpec,
        scatter_to_sequence_parallel: bool = True,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        profile_mode: int = 0,
        warmup_iters: int = 2,
        theoretical_flops: bool = False,
        theoretical_activations: bool = False,
        tp_comm_overlap_cfg: str = None,
        **kwargs,
    ):
        super().__init__(
            hf_config=hf_config,
            tf_config=tf_config,
            tp_group=tp_group,
            profile_mode=profile_mode,
            warmup_iters=warmup_iters,
            scatter_to_sequence_parallel=scatter_to_sequence_parallel,
            pg_collection=pg_collection,
            theoretical_flops=theoretical_flops,
            theoretical_activations=theoretical_activations,
            tp_comm_overlap_cfg=tp_comm_overlap_cfg,
        )
        self.module_name = "GPTModelModuleQueue"
        self.transformer_layer_spec = transformer_layer_spec
        self.scatter_to_sequence_parallel = scatter_to_sequence_parallel
        self.tp_group = tp_group
        self.pg_collection = pg_collection

        if profile_mode == ProfileMode.collect_data:
            with MemoryTrackerContext(self.module_name) as memory_tracker_ctx:
                self.op = GPTModelModuleQueueForTest(
                    tf_config=tf_config,
                    hf_config=hf_config,
                    transformer_layer_spec=transformer_layer_spec,
                    hook_activation=True,
                    scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                    tp_group=tp_group,
                    **kwargs,
                )
            detailed_mem_report = memory_tracker_ctx.get_result()
            self._calculate_weight_memory(detailed_mem_report, tf_config, hf_config)
            self.memory_db["weights"][self.module_name] = detailed_mem_report
        else:
            self.op = GPTModelModuleQueueForTest(
                tf_config=tf_config,
                hf_config=hf_config,
                transformer_layer_spec=transformer_layer_spec,
                hook_activation=False,
                scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                tp_group=tp_group,
                **kwargs,
            )

    def _calculate_weight_memory(
        self,
        detailed_mem_report: Dict,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
    ):
        """Calculate estimated weight memory for last pipeline stage.

        Since this is the last pipeline stage (pre_process=False, post_process=True):
        - No embedding weights
        - Has transformer layer weights
        - Has output layer weights
        """
        hidden_size = hf_config.hidden_size
        num_layers = hf_config.num_hidden_layers
        vocab_size = hf_config.vocab_size
        ffn_hidden_size = hf_config.intermediate_size
        num_attention_heads = hf_config.num_attention_heads
        kv_channels = hidden_size // num_attention_heads

        tp_size = mpu.get_tensor_model_parallel_world_size()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        dtype = next(self.op.parameters()).dtype
        bytes_per_param = torch.tensor([], dtype=dtype).element_size()

        estimated_weight_bytes = 0

        # Transformer layers (this is last stage, so it has some layers)
        layers_per_rank = num_layers // pp_size

        # Input layer norm per layer
        if tf_config.normalization == "RMSNorm":
            estimated_weight_bytes += layers_per_rank * hidden_size * bytes_per_param
        else:
            estimated_weight_bytes += (
                layers_per_rank * 2 * hidden_size * bytes_per_param
            )

        # Self attention per layer
        query_projection_size = hidden_size
        kv_projection_size = kv_channels * tf_config.num_query_groups
        qkv_size = query_projection_size + 2 * kv_projection_size
        estimated_weight_bytes += (
            layers_per_rank * hidden_size * (qkv_size // tp_size) * bytes_per_param
        )

        # Output projection per layer
        estimated_weight_bytes += (
            layers_per_rank * (hidden_size // tp_size) * hidden_size * bytes_per_param
        )

        # MLP per layer
        # Column parallel
        estimated_weight_bytes += (
            layers_per_rank
            * hidden_size
            * (ffn_hidden_size // tp_size)
            * bytes_per_param
        )
        # Row parallel
        estimated_weight_bytes += (
            layers_per_rank
            * (ffn_hidden_size // tp_size)
            * hidden_size
            * bytes_per_param
        )

        # Post-attention layer norm per layer
        if tf_config.normalization == "RMSNorm":
            estimated_weight_bytes += layers_per_rank * hidden_size * bytes_per_param
        else:
            estimated_weight_bytes += (
                layers_per_rank * 2 * hidden_size * bytes_per_param
            )

        # Final layer norm (last pipeline stage)
        if tf_config.normalization == "RMSNorm":
            estimated_weight_bytes += hidden_size * bytes_per_param
        else:
            estimated_weight_bytes += 2 * hidden_size * bytes_per_param

        # Output layer weights (last pipeline stage has output layer)
        estimated_weight_bytes += (
            hidden_size * (vocab_size // tp_size) * bytes_per_param
        )

        detailed_mem_report["estimated_peak_mem_diff"] = get_memory_str(
            estimated_weight_bytes, human_readable=True
        )

    @override
    def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
        """Prepare hidden states input for last pipeline stage.

        Uses HiddenStatusGenerator to create decoder_input (hidden states)
        that would come from the previous pipeline stage.
        """
        # Get hidden states and other inputs from parent class
        inputs = super().prepare_input(test_case, micro_batch)
        # inputs format: (hidden_states, attention_mask, rotary_pos_emb, ...)
        # We need: (decoder_input, attention_mask, labels, packed_seq_params, ...)

        hidden_states = inputs[0]  # decoder_input
        attention_mask = inputs[1]

        # Return in format expected by GPTModelModuleQueueForTest.forward()
        return (
            hidden_states,  # decoder_input
            attention_mask,
            None,  # labels
            None,  # packed_seq_params
            None,  # extra_block_kwargs
            None,  # runtime_gather_output
        )

    @override
    def calculate_tokens(
        self, test_case: InputTestCase, micro_batch: TensorDict, inputs: Any
    ) -> int:
        """Calculate number of tokens processed."""
        decoder_input = inputs[0]
        # decoder_input shape: [seq_len, batch_size, hidden_size] for thd format
        tokens = decoder_input.shape[0] * decoder_input.shape[1]
        return tokens

    @override
    def calc_theoretical_memory(self, test_case: InputTestCase) -> Dict[str, int]:
        """Calculate theoretical activation memory for last pipeline stage."""
        hidden_size = self.hf_config.hidden_size
        micro_batch_size = test_case.micro_batch_size
        seq_len = test_case.seqlen
        dtype = next(self.op.parameters()).dtype
        bytes_per_param = torch.tensor([], dtype=dtype).element_size()

        # Base activation memory
        # For last stage: decoder layers + output layer activations
        activation_mem = 18 * micro_batch_size * seq_len * hidden_size * bytes_per_param

        # Adjust for tensor parallelism (sequence parallel)
        if test_case.sequence_parallel_enabled:
            tp_size = mpu.get_tensor_model_parallel_world_size()
            activation_mem = activation_mem // tp_size

        # Adjust for context parallelism
        if test_case.context_parallel_size > 1:
            activation_mem = activation_mem // test_case.context_parallel_size

        return {"activations": {"activations": activation_mem}}

    @override
    def calc_theoretical_flops(self, test_case: InputTestCase) -> Dict[str, float]:
        """Calculate theoretical FLOPS for last pipeline stage.

        Last stage has:
        - Transformer layers (subset based on PP)
        - Final layer norm
        - Output layer
        - No embedding (pre_process=False)
        """
        hidden_size = self.hf_config.hidden_size
        num_layers = self.hf_config.num_hidden_layers
        vocab_size = self.hf_config.vocab_size
        ffn_hidden_size = self.hf_config.intermediate_size
        micro_batch_size = test_case.micro_batch_size
        seq_len = test_case.seqlen

        tp_size = mpu.get_tensor_model_parallel_world_size()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        cp_size = test_case.context_parallel_size

        tokens = micro_batch_size * seq_len
        if cp_size > 1:
            tokens = tokens // cp_size
        layers_per_rank = num_layers // pp_size

        forward_flops = 0.0

        # Transformer layers FLOPS
        if layers_per_rank > 0:
            # Self-attention FLOPS per layer
            qkv_flops = (
                2
                * tokens
                * hidden_size
                * (
                    hidden_size
                    + 2 * (hidden_size // self.hf_config.num_attention_heads)
                )
            )
            attn_flops = 2 * tokens * tokens * hidden_size
            proj_flops = 2 * tokens * hidden_size * hidden_size

            # MLP FLOPS per layer
            mlp_flops = (
                2 * tokens * hidden_size * ffn_hidden_size
                + 2 * tokens * ffn_hidden_size * hidden_size
            )

            # Layer norm FLOPS (approx 5 FLOPS per element)
            ln_flops = 5 * tokens * hidden_size

            # Total FLOPS per layer
            layer_flops = qkv_flops + attn_flops + proj_flops + mlp_flops + 2 * ln_flops

            # Adjust for tensor parallelism
            layer_flops = layer_flops / tp_size

            # Total for all layers on this rank
            forward_flops += layers_per_rank * layer_flops

        # Final layer norm FLOPS (last pipeline stage)
        forward_flops += 5 * tokens * hidden_size

        # Output layer FLOPS (last pipeline stage)
        forward_flops += 2 * tokens * hidden_size * (vocab_size // tp_size)

        # Backward FLOPS is approximately 2x forward FLOPS
        backward_flops = 2 * forward_flops

        return {"forward": forward_flops, "backward": backward_flops}

"""GPTModelModuleQueue operator for AutoTuner testing.

This module provides a test wrapper for GPTModelModuleQueue, which is designed
for the last pipeline stage (pre_process=False, post_process=True) with
memory-efficient CPU offloading of transformer layers.
"""

import logging
from typing import Optional

import torch
from megatron.core.inference.contexts.base_context import BaseInferenceContext
from megatron.core.models.gpt.gpt_model_module_queue import GPTModelModuleQueue
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor
from transformers import PretrainedConfig

from AutoTuner.utils.nvtx import nvtx_decorator

from .common import CommonOpsForTest
from .gpt_model import NVTXDecoder


class GPTModelModuleQueueForTest(GPTModelModuleQueue, CommonOpsForTest):
    """GPTModelModuleQueue with testing support.

    This is designed for the last pipeline stage:
    - pre_process=False: No embedding layer, receives hidden states from previous stage
    - post_process=True: Has output layer for final logits

    The module queue enables memory-efficient training by offloading transformer
    layers to CPU and loading output layer weights in chunks.
    """

    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        transformer_layer_spec: ModuleSpec,
        hook_activation: bool = False,
        scatter_to_sequence_parallel: bool = True,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        **kwargs,
    ):
        # GPTModelModuleQueue is for last pipeline stage
        GPTModelModuleQueue.__init__(
            self,
            config=tf_config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=hf_config.vocab_size,
            max_sequence_length=hf_config.max_position_embeddings,
            pre_process=False,  # Last pipeline stage: no embedding
            post_process=True,  # Last pipeline stage: has output layer
            fp16_lm_cross_entropy=False,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type="rope",
            scatter_embedding_sequence_parallel=scatter_to_sequence_parallel,
            seq_len_interpolation_factor=None,
            **kwargs,
        )
        CommonOpsForTest.__init__(
            self,
            hook_activation=hook_activation,
            module_name="GPTModelModuleQueue",
            logging_level=logging.INFO,
        )
        # Replace decoder with NVTX-instrumented version for profiling
        self.decoder = NVTXDecoder(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=self.pg_collection,
        )
        # Re-initialize module queue state after decoder replacement
        if self._module_queue_enabled:
            self.num_layers = len(self.decoder.layers)
            self.layers_on_cpu = [False] * self.num_layers
            self._forward_hooks = []
            self._backward_hooks = []
            self._register_hooks()

    @nvtx_decorator(message="GPTModelModuleQueue forward")
    def _forward(
        self,
        decoder_input: Tensor,
        attention_mask: Tensor,
        labels: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for last pipeline stage.

        Since pre_process=False, this model receives decoder_input (hidden states)
        from the previous pipeline stage instead of input_ids.
        """
        # Call parent forward with decoder_input
        # input_ids and position_ids are None since we're not the first stage
        return super().forward(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            loss_mask=loss_mask,
        )

    def forward(
        self,
        decoder_input: Tensor,
        attention_mask: Tensor,
        labels: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward with activation hook for memory tracking."""
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                labels=labels,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
                runtime_gather_output=runtime_gather_output,
                inference_params=inference_params,
                loss_mask=loss_mask,
            )
        return ret

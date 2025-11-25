import logging
from typing import Optional

import torch
from megatron.core.inference.contexts.base_context import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor
from transformers import PretrainedConfig

from AutoTuner.utils.memory import ActivationHook, MemoryTracker
from AutoTuner.utils.nvtx import nvtx_decorator, nvtx_range_pop, nvtx_range_push

from .common import CommonOpsForTest


class GPTModelForTest(GPTModel, CommonOpsForTest):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        transformer_layer_spec: ModuleSpec,
        hook_activation=False,
        scatter_to_sequence_parallel: bool = True,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        **kwargs,
    ):
        GPTModel.__init__(
            self,
            config=tf_config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=hf_config.vocab_size,
            max_sequence_length=hf_config.max_position_embeddings,
            pre_process=True,
            post_process=True,
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
            module_name="GPTModel",
            logging_level=logging.INFO,
        )

    @nvtx_decorator(message="GPTModel forward")
    def _forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,
            loss_mask=loss_mask,
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
                runtime_gather_output=runtime_gather_output,
                inference_params=inference_params,
                loss_mask=loss_mask,
            )
        return ret

import logging
from collections import OrderedDict
from contextlib import nullcontext
from typing import Optional, Union

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import te_checkpoint
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts.base_context import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    roll_tensor,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import (
    TransformerBlock,
    TransformerBlockSubmodules,
    _get_block_submodules,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    get_pg_rank,
    make_viewless_tensor,
)
from torch import Tensor
from transformer_engine.pytorch.cpu_offload import get_cpu_offload_context
from transformers import PretrainedConfig

from AutoTuner.utils.memory import ActivationHook, MemoryTracker
from AutoTuner.utils.nvtx import nvtx_decorator, nvtx_range_pop, nvtx_range_push

from .common import CommonOpsForTest
from .gpt_model import NVTXDecoder


class GPTModelEnhancedForTest(GPTModel, CommonOpsForTest):
    """GPT Model with CPU Embedding support for testing.

    This enhanced version uses CPU embeddings when config.cpu_embedding is True,
    keeping embedding weights on CPU to reduce GPU memory usage.
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
            module_name="GPTModelEnhanced",
            logging_level=logging.INFO,
        )
        self.decoder = NVTXDecoder(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=self.pg_collection,
        )

    @nvtx_decorator(message="GPTModelEnhanced forward")
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
        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=None,
            packed_seq_params=packed_seq_params,
        )

        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        ) = preproc_output[:5]

        rotary_pos_cos_sin = preproc_output[5] if len(preproc_output) == 6 else None

        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=None,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
        )

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=None,
        )

    @nvtx_decorator(message="preprocess")
    def _preprocess(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        decoder_input: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Preprocesses inputs for the transformer decoder.

        Applies embeddings to input tokens, or uses `decoder_input` from a previous
        pipeline stage. Also sets up rotary positional embeddings.

        Note: When using CPU embeddings, input_ids and position_ids can be on GPU,
        the embedding layer will handle moving them to CPU for lookup.
        """
        in_inference_mode = inference_context is not None and not self.training

        # Decoder embedding
        nvtx_range_push(suffix="word_embedding")
        if decoder_input is not None:
            pass
        elif self.pre_process:
            # CPU embedding handles CPU/GPU movement internally
            decoder_input = self.embedding(
                input_ids=input_ids, position_ids=position_ids
            )
        else:
            # intermediate stage of pipeline
            decoder_input = None
        nvtx_range_pop(suffix="word_embedding")

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        nvtx_range_push(suffix="rotary_embedding")
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        rotary_pos_cos_sin = None

        if (
            self.position_embedding_type == "rope"
            and not self.config.multi_latent_attention
        ):
            use_flash_infer_fused_rope = (
                hasattr(inference_context, "use_flashinfer_fused_rope")
                and inference_context.use_flashinfer_fused_rope
            )
            if in_inference_mode and (
                self.config.flash_decode or use_flash_infer_fused_rope
            ):
                assert (
                    not self.config.flash_decode
                ) or inference_context.is_static_batching(), (
                    "Flash decode is only applicable to static batching."
                )
                if self.config.flash_decode:
                    rotary_pos_cos, rotary_pos_sin = (
                        self.rotary_pos_emb_cache.setdefault(
                            inference_context.max_sequence_length,
                            self.rotary_pos_emb.get_cos_sin(
                                inference_context.max_sequence_length
                            ),
                        )
                    )
                elif use_flash_infer_fused_rope:
                    assert (
                        not self.mtp_process
                    ), "MTP not tested with flashinfer_fused_rope"
                    rotary_pos_cos_sin = self.rotary_pos_emb_cache.setdefault(
                        inference_context.max_sequence_length,
                        torch.cat(
                            self.rotary_pos_emb.get_cos_sin(
                                inference_context.max_sequence_length
                            ),
                            -1,
                        ),
                    )
            else:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_context,
                    self.decoder,
                    decoder_input,
                    self.config,
                    packed_seq_params,
                )
                rotary_pos_emb = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None
                    and packed_seq_params.qkv_format == "thd",
                )
        elif self.position_embedding_type == "yarn":
            if self.training or not self.config.flash_decode:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_context,
                    self.decoder,
                    decoder_input,
                    self.config,
                    packed_seq_params,
                )
                rotary_pos_emb, _ = self.rotary_pos_emb(rotary_seq_len)
            else:
                raise NotImplementedError(
                    "Flash decoding uses precomputed cos and sin for RoPE, not implemented in "
                    "YarnRotaryEmbedding yet."
                )
        elif (
            self.position_embedding_type == "mrope"
            and not self.config.multi_latent_attention
        ):
            if self.training or not self.config.flash_decode:
                rotary_pos_emb = self.rotary_pos_emb(position_ids, self.mrope_section)
            else:
                raise NotImplementedError(
                    "Flash decoding uses precomputed cos and sin for RoPE, not implemented in "
                    "MultimodalRotaryEmbedding yet."
                )
        nvtx_range_pop(suffix="rotary_embedding")

        if (
            in_inference_mode
            and (
                (
                    self.config.enable_cuda_graph
                    and self.config.cuda_graph_scope != "full_iteration"
                )
                or self.config.flash_decode
            )
            and rotary_pos_cos is not None
            and inference_context.is_static_batching()
        ):
            current_batch_size = input_ids.shape[0]
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,
            )
        else:
            sequence_len_offset = None

        if in_inference_mode and not has_config_logger_enabled(self.config):
            decoder_input = WrappedTensor(decoder_input)

        preproc_output = (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )
        if rotary_pos_cos_sin is not None:
            preproc_output += (rotary_pos_cos_sin,)

        return preproc_output

    @nvtx_decorator(message="postprocess")
    def _postprocess(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        mtp_in_postprocess=None,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
        inference_context=None,
    ):
        """Postprocesses decoder hidden states to generate logits or compute loss."""
        in_inference_mode = inference_context is not None and not self.training
        if in_inference_mode:
            assert runtime_gather_output, "Inference must always gather TP logits"

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if not self.post_process:
            return hidden_states

        if mtp_in_postprocess:
            nvtx_range_push(suffix="mtp_forward")
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=self.embedding,
                **(extra_block_kwargs or {}),
            )
            nvtx_range_pop(suffix="mtp_forward")

        if self.mtp_process:
            nvtx_range_push(suffix="mtp_loss")
            mtp_labels = labels.clone()
            hidden_states_list = torch.chunk(
                hidden_states, 1 + self.config.mtp_num_layers, dim=0
            )
            hidden_states = hidden_states_list[0]
            if loss_mask is None:
                loss_mask = torch.ones_like(mtp_labels)
            for mtp_layer_number in range(self.config.mtp_num_layers):
                mtp_logits, _ = self.output_layer(
                    hidden_states_list[mtp_layer_number + 1],
                    weight=output_weight,
                    runtime_gather_output=runtime_gather_output,
                )
                mtp_labels, _ = roll_tensor(
                    mtp_labels, shifts=-1, dims=-1, cp_group=self.cp_group
                )
                loss_mask, num_tokens = roll_tensor(
                    loss_mask, shifts=-1, dims=-1, cp_group=self.cp_group
                )
                mtp_loss = self.compute_language_model_loss(mtp_labels, mtp_logits)
                mtp_loss = loss_mask * mtp_loss
                if self.training:
                    MTPLossLoggingHelper.save_loss_to_tracker(
                        torch.sum(mtp_loss) / num_tokens,
                        mtp_layer_number,
                        self.config.mtp_num_layers,
                        avg_group=parallel_state.get_data_parallel_group(
                            with_context_parallel=True
                        ),
                    )
                mtp_loss_scale = (
                    self.config.mtp_loss_scaling_factor / self.config.mtp_num_layers
                )
                if self.config.calculate_per_token_loss:
                    hidden_states = MTPLossAutoScaler.apply(
                        hidden_states, mtp_loss_scale * mtp_loss
                    )
                else:
                    hidden_states = MTPLossAutoScaler.apply(
                        hidden_states, mtp_loss_scale * mtp_loss / num_tokens
                    )
            nvtx_range_pop(suffix="mtp_loss")

        nvtx_range_push(suffix="output_layer")
        sequence_parallel_override = False
        if in_inference_mode and inference_context.materialize_only_last_token_logits:
            if inference_context.is_static_batching():
                hidden_states = hidden_states[-1:, :, :]
            else:
                if self.output_layer.sequence_parallel:
                    hidden_states = gather_from_sequence_parallel_region(
                        hidden_states, group=self.pg_collection.tp
                    )
                    self.output_layer.sequence_parallel = False
                    sequence_parallel_override = True

                hidden_states = inference_context.last_token_logits(
                    hidden_states.squeeze(1).unsqueeze(0)
                ).unsqueeze(1)

        logits, _ = self.output_layer(
            hidden_states,
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        )
        nvtx_range_pop(suffix="output_layer")

        if sequence_parallel_override:
            assert (
                in_inference_mode
                and inference_context.is_dynamic_batching()
                and inference_context.materialize_only_last_token_logits
            )
            self.output_layer.sequence_parallel = True

        if has_config_logger_enabled(self.config):
            payload = OrderedDict(
                {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "decoder_input": decoder_input,
                    "logits": logits,
                }
            )
            log_config_to_disk(self.config, payload, prefix="input_and_logits")

        if labels is None:
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)

        return loss

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

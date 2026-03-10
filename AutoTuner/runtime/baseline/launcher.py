import itertools
import json
import logging
import os
import time
from dataclasses import asdict
from typing import Iterable, Optional

import torch
from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from transformers import PretrainedConfig

from AutoTuner.runtime.baseline.simulator import (
    StageParamStats,
    StageTimingStats,
    build_pp_stage_layer_counts,
    get_dp_allreduce_comm_config,
    simulate_full_iteration,
)
from AutoTuner.testbench.ops.gpt_model import GPTModelForTest
from AutoTuner.utils.config import (
    get_hf_model_config,
    get_mcore_model_config_from_hf_config,
)
from AutoTuner.utils.gpu_info import GPU_PEAK_FLOPS
from AutoTuner.utils.logging import log_rank0, log_with_rank
from AutoTuner.utils.memory import (
    get_all_rank_peak_memory_stats,
    get_memory_str,
    reset_peak_memory_stats,
)
from AutoTuner.utils.model_inputs import DataSets
from AutoTuner.utils.structs import InputTestCase
from AutoTuner.utils.tp_overlap import destroy_ub, initialize_tp_communicators
from verl.models.mcore import get_mcore_forward_fn, get_mcore_forward_fused_fn
from verl.models.mcore.model_forward_fused import patch_fused_forward
from verl.utils.flops_counter import FlopsCounter
from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits
from verl.utils.megatron_utils import get_model, unwrap_model


class _ModuleCudaTimer:
    def __init__(self):
        self._forward_start = None
        self._backward_start = None
        self.forward_event_pairs = []
        self.backward_event_pairs = []

    def reset(self):
        self._forward_start = None
        self._backward_start = None
        self.forward_event_pairs.clear()
        self.backward_event_pairs.clear()

    def forward_pre_hook(self, module, inputs):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._forward_start = start

    def forward_hook(self, module, inputs, output):
        if self._forward_start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.forward_event_pairs.append((self._forward_start, end))
        self._forward_start = None

    def backward_pre_hook(self, module, grad_output):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._backward_start = start

    def backward_hook(self, module, grad_input, grad_output):
        if self._backward_start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.backward_event_pairs.append((self._backward_start, end))
        self._backward_start = None

    @staticmethod
    def _sum_elapsed_time_s(event_pairs) -> float:
        total_ms = 0.0
        for start, end in event_pairs:
            total_ms += start.elapsed_time(end)
        return total_ms / 1e3

    def summary(self) -> tuple[float, float]:
        return (
            self._sum_elapsed_time_s(self.forward_event_pairs),
            self._sum_elapsed_time_s(self.backward_event_pairs),
        )


class RuntimeLauncher:
    def __init__(
        self,
        model_name: str,
        test_cases: list[InputTestCase],
        override_model_kwargs: dict,
        override_tf_config_kwargs: dict,
        tp_comm_overlap_cfg: str | None = None,
        share_embeddings_and_output_weights: Optional[bool] = None,
        wrap_with_ddp: bool = True,
        use_distributed_optimizer: bool = False,
        fix_compute_amount: bool = True,
    ) -> None:
        self.model_name = model_name
        self._log_all_ranks = os.getenv("AUTOTUNER_LOG_ALL_RANKS", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._log_microbatch_every = 1
        self._log_microbatch_io = True
        self._current_iteration = None
        self._microbatch_counter = 0
        self._tp_overlap_tokens = None
        self._tp_overlap_disabled_logged = False
        self.test_cases = test_cases
        self.tp_comm_overlap_cfg = tp_comm_overlap_cfg
        self.wrap_with_ddp = wrap_with_ddp

        self.hf_config: PretrainedConfig = get_hf_model_config(
            model_name, **override_model_kwargs
        )
        # default transformer config optimization
        override_tf_config_kwargs.setdefault("persist_layer_norm", True)
        override_tf_config_kwargs.setdefault("bias_activation_fusion", True)
        override_tf_config_kwargs.setdefault("apply_rope_fusion", True)
        override_tf_config_kwargs.setdefault("moe_permute_fusion", True)
        override_tf_config_kwargs.setdefault("deallocate_pipeline_outputs", True)
        override_tf_config_kwargs.setdefault("gradients_accumulation_fusion", True)

        self.tf_config = get_mcore_model_config_from_hf_config(
            self.hf_config, **override_tf_config_kwargs
        )
        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        # Ensure TF config matches actual VPP setup; avoid asserting on missing vp_stage.
        self.tf_config.virtual_pipeline_model_parallel_size = vpp_size
        if vpp_size is None:
            self._log(
                "vpp disabled: forcing tf_config.virtual_pipeline_model_parallel_size=None"
            )

        if share_embeddings_and_output_weights is None:
            share_embeddings_and_output_weights = bool(
                getattr(self.hf_config, "tie_word_embeddings", False)
            )
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

        assert (
            torch.distributed.is_initialized()
        ), "torch.distributed is not initialized"
        self.tp_group = mpu.get_tensor_model_parallel_group()
        self._configure_lightweight_pipeline_layers()

        self.datasets = DataSets(
            self.hf_config,
            self.test_cases,
            fix_compute_amount=fix_compute_amount,
            use_dynamic_bsz_balance=True,
            vpp_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
        )
        log_rank0(
            "runtime dataset ready: "
            f"test_cases={len(self.test_cases)} "
            f"original_layers={self.original_total_layers} "
            f"runtime_layers={self.runtime_total_layers} "
            f"vpp={mpu.get_virtual_pipeline_model_parallel_world_size()} "
            f"fix_compute_amount={fix_compute_amount}"
        )

        self.flops_counter = FlopsCounter(self.hf_config)
        self.use_fused_kernels = os.getenv(
            "AUTOTUNER_RUNTIME_USE_FUSED_KERNELS", "1"
        ).lower() in ("1", "true", "yes", "on")
        self._forward_fn = get_mcore_forward_fn(self.hf_config)
        self._forward_fused_fn = get_mcore_forward_fused_fn(self.hf_config)

        self.model = self._build_model(
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=use_distributed_optimizer,
        )
        self._maybe_patch_fused_forward()
        self._stage_timers = self._register_stage_timers()
        self.simulation_context = self._build_simulation_context()
        self.latest_run_report = None
        log_rank0(
            "runtime model built: "
            f"wrap_with_ddp={wrap_with_ddp} "
            f"use_distributed_optimizer={use_distributed_optimizer} "
            f"use_fused_kernels={self.use_fused_kernels}"
        )

    def _log(self, message: str, level: int = logging.INFO, all_ranks: bool = False):
        if all_ranks or self._log_all_ranks:
            log_with_rank(message, level=level)
        else:
            log_rank0(message, level=level)

    def _configure_lightweight_pipeline_layers(self):
        """Build only one transformer layer per virtual chunk to avoid OOM."""
        pp_size = max(1, mpu.get_pipeline_model_parallel_world_size())
        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1

        cfg = getattr(self.hf_config, "text_config", self.hf_config)
        self.original_total_layers = int(
            getattr(cfg, "num_hidden_layers", self.tf_config.num_layers)
        )
        self.original_total_layers = max(1, self.original_total_layers)
        self.runtime_total_layers = pp_size * vpp_size
        self.runtime_layers_per_pp_rank = vpp_size
        self.flops_layer_scale = self.runtime_total_layers / max(
            1, self.original_total_layers
        )

        # Force lightweight transformer depth: one layer per virtual chunk.
        self.tf_config.num_layers = self.runtime_total_layers
        self._log(
            "runtime lightweight layer mode: "
            f"original_total_layers={self.original_total_layers} "
            f"runtime_total_layers={self.runtime_total_layers} "
            f"pp={pp_size} vpp={vpp_size} "
            f"layers_per_pp_rank={self.runtime_layers_per_pp_rank}"
        )

    def _representative_layer_number(
        self, pp_rank: int, vp_stage: int, local_layer_idx: int = 0
    ) -> int:
        """Pick first layer of each theoretical chunk from the full model."""
        pp_size = max(1, mpu.get_pipeline_model_parallel_world_size())
        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
        total_chunks = pp_size * vpp_size
        chunk_id = vp_stage * pp_size + pp_rank
        first_layer = (chunk_id * self.original_total_layers) // total_chunks + 1
        return first_layer + local_layer_idx

    @staticmethod
    def _set_layer_number(layer, layer_number: int):
        layer.layer_number = layer_number
        self_attention = getattr(layer, "self_attention", None)
        if self_attention is not None and hasattr(self_attention, "layer_number"):
            self_attention.layer_number = layer_number
        cross_attention = getattr(layer, "cross_attention", None)
        if cross_attention is not None and hasattr(cross_attention, "layer_number"):
            cross_attention.layer_number = layer_number
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            if hasattr(mlp, "set_layer_number"):
                mlp.set_layer_number(layer_number)
            elif hasattr(mlp, "layer_number"):
                mlp.layer_number = layer_number

    def _assign_representative_layer_numbers(self, model):
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        for chunk_idx, model_chunk in enumerate(model):
            unwrapped = unwrap_model(model_chunk)
            vp_stage = getattr(unwrapped, "vp_stage", None)
            if vp_stage is None:
                vp_stage = chunk_idx if len(model) > 1 else 0
            decoder = getattr(unwrapped, "decoder", None)
            layers = [] if decoder is None else list(getattr(decoder, "layers", []))
            for local_layer_idx, layer in enumerate(layers):
                representative_layer = self._representative_layer_number(
                    pp_rank=pp_rank,
                    vp_stage=vp_stage,
                    local_layer_idx=local_layer_idx,
                )
                self._set_layer_number(layer, representative_layer)
                self._log(
                    "layer remap: "
                    f"pp_rank={pp_rank} vp_stage={vp_stage} chunk_idx={chunk_idx} "
                    f"local_layer_idx={local_layer_idx} layer_number={representative_layer}",
                    all_ranks=True,
                )

    def _build_model(self, wrap_with_ddp: bool, use_distributed_optimizer: bool):
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=self.tf_config.num_moe_experts,
            multi_latent_attention=self.tf_config.multi_latent_attention,
            qk_layernorm=self.tf_config.qk_layernorm,
            moe_grouped_gemm=self.tf_config.moe_grouped_gemm,
        )

        def model_provider(pre_process: bool, post_process: bool, vp_stage: int = None):
            return GPTModelForTest(
                tf_config=self.tf_config,
                hf_config=self.hf_config,
                transformer_layer_spec=transformer_layer_spec,
                pre_process=pre_process,
                post_process=post_process,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                hook_activation=False,
                scatter_to_sequence_parallel=True,
                tp_group=self.tp_group,
                vp_stage=vp_stage,
            )

        model = get_model(
            model_provider,
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=use_distributed_optimizer,
            transformer_config=self.tf_config,
        )
        self._assign_representative_layer_numbers(model)
        return model

    def _iter_model_chunks(self):
        if isinstance(self.model, list):
            return self.model
        return [self.model]

    def _register_stage_timers(self) -> list[_ModuleCudaTimer]:
        timers: list[_ModuleCudaTimer] = []
        for model_chunk in self._iter_model_chunks():
            timer = _ModuleCudaTimer()
            module = unwrap_model(model_chunk)
            module.register_forward_pre_hook(timer.forward_pre_hook)
            module.register_forward_hook(timer.forward_hook)
            module.register_full_backward_pre_hook(timer.backward_pre_hook)
            module.register_full_backward_hook(timer.backward_hook)
            timers.append(timer)
        return timers

    def _reset_stage_timers(self):
        for timer in self._stage_timers:
            timer.reset()

    def _collect_local_stage_timing_stats(
        self, num_microbatches: int
    ) -> StageTimingStats:
        forward_total_time_s = 0.0
        backward_total_time_s = 0.0
        runtime_layer_count = 0
        for timer, model_chunk in zip(self._stage_timers, self._iter_model_chunks()):
            chunk_forward_time_s, chunk_backward_time_s = timer.summary()
            forward_total_time_s += chunk_forward_time_s
            backward_total_time_s += chunk_backward_time_s
            runtime_layer_count += len(
                list(getattr(getattr(unwrap_model(model_chunk), "decoder", None), "layers", []))
            )

        runtime_forward_time_s = 0.0
        runtime_backward_time_s = 0.0
        if num_microbatches > 0:
            runtime_forward_time_s = forward_total_time_s / float(num_microbatches)
            runtime_backward_time_s = backward_total_time_s / float(num_microbatches)

        return StageTimingStats(
            pp_rank=mpu.get_pipeline_model_parallel_rank(),
            runtime_layer_count=runtime_layer_count,
            runtime_forward_time_s=runtime_forward_time_s,
            runtime_backward_time_s=runtime_backward_time_s,
            runtime_forward_total_time_s=forward_total_time_s,
            runtime_backward_total_time_s=backward_total_time_s,
            num_microbatches=num_microbatches,
        )

    def _gather_stage_timing_stats(
        self, num_microbatches: int
    ) -> list[StageTimingStats]:
        local_stats = self._collect_local_stage_timing_stats(num_microbatches)
        gathered_stats = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_stats, local_stats)

        stage_stats_by_pp_rank: dict[int, StageTimingStats] = {}
        for stage_stats in gathered_stats:
            existing = stage_stats_by_pp_rank.get(stage_stats.pp_rank)
            existing_total = 0.0 if existing is None else (
                existing.runtime_forward_total_time_s + existing.runtime_backward_total_time_s
            )
            current_total = (
                stage_stats.runtime_forward_total_time_s
                + stage_stats.runtime_backward_total_time_s
            )
            if existing is None or current_total > existing_total:
                stage_stats_by_pp_rank[stage_stats.pp_rank] = stage_stats

        pp_size = max(1, mpu.get_pipeline_model_parallel_world_size())
        missing_pp_ranks = [
            pp_rank
            for pp_rank in range(pp_size)
            if pp_rank not in stage_stats_by_pp_rank
        ]
        if missing_pp_ranks:
            raise RuntimeError(
                f"Missing stage timing stats for pp ranks: {missing_pp_ranks}"
            )
        return [stage_stats_by_pp_rank[pp_rank] for pp_rank in range(pp_size)]

    def _maybe_patch_fused_forward(self):
        if not self.use_fused_kernels:
            self._log("runtime fused forward disabled")
            return
        if getattr(self.tf_config, "overlap_moe_expert_parallel_comm", False):
            self._log(
                "runtime fused forward disabled because overlap_moe_expert_parallel_comm is enabled"
            )
            self.use_fused_kernels = False
            return
        for model_chunk in self._iter_model_chunks():
            patch_fused_forward(model_chunk)
        self._log("runtime fused forward patch enabled on all model chunks")

    @staticmethod
    def _build_next_token_labels(
        input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.roll(input_ids, shifts=-1, dims=-1)
        label_mask = attention_mask.to(bool).clone()
        label_mask[:, :-1] &= attention_mask[:, 1:].to(bool)
        label_mask[:, -1] = False
        return labels, label_mask

    @staticmethod
    def _extract_log_probs(output):
        if isinstance(output, dict):
            return output.get("log_probs")
        return getattr(output, "log_probs", None)

    def _collect_local_stage_param_stats(self) -> StageParamStats:
        decoder_param_ids: set[int] = set()
        runtime_layer_count = 0
        for model_chunk in self._iter_model_chunks():
            unwrapped = unwrap_model(model_chunk)
            decoder = getattr(unwrapped, "decoder", None)
            layers = [] if decoder is None else list(getattr(decoder, "layers", []))
            runtime_layer_count += len(layers)
            for layer in layers:
                for param in layer.parameters():
                    if param.requires_grad:
                        decoder_param_ids.add(id(param))

        total_param_bytes = 0
        decoder_param_bytes = 0
        seen_param_ids: set[int] = set()
        for model_chunk in self._iter_model_chunks():
            for param in model_chunk.parameters():
                if not param.requires_grad:
                    continue
                param_id = id(param)
                if param_id in seen_param_ids:
                    continue
                seen_param_ids.add(param_id)
                param_bytes = param.numel() * param.element_size()
                total_param_bytes += param_bytes
                if param_id in decoder_param_ids:
                    decoder_param_bytes += param_bytes

        return StageParamStats(
            pp_rank=mpu.get_pipeline_model_parallel_rank(),
            runtime_layer_count=runtime_layer_count,
            runtime_total_param_bytes=total_param_bytes,
            runtime_decoder_param_bytes=decoder_param_bytes,
        )

    def _gather_stage_param_stats(self) -> list[StageParamStats]:
        local_stats = self._collect_local_stage_param_stats()
        gathered_stats = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_stats, local_stats)

        stage_stats_by_pp_rank: dict[int, StageParamStats] = {}
        for stage_stats in gathered_stats:
            existing = stage_stats_by_pp_rank.get(stage_stats.pp_rank)
            if (
                existing is None
                or stage_stats.runtime_total_param_bytes
                > existing.runtime_total_param_bytes
            ):
                stage_stats_by_pp_rank[stage_stats.pp_rank] = stage_stats

        pp_size = max(1, mpu.get_pipeline_model_parallel_world_size())
        missing_pp_ranks = [
            pp_rank
            for pp_rank in range(pp_size)
            if pp_rank not in stage_stats_by_pp_rank
        ]
        if missing_pp_ranks:
            raise RuntimeError(
                f"Missing stage param stats for pp ranks: {missing_pp_ranks}"
            )
        return [stage_stats_by_pp_rank[pp_rank] for pp_rank in range(pp_size)]

    def _build_simulation_context(self) -> dict:
        pp_size = max(1, mpu.get_pipeline_model_parallel_world_size())
        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
        runtime_stage_layer_counts = build_pp_stage_layer_counts(
            total_layers=self.runtime_total_layers,
            pp_size=pp_size,
            vpp_size=vpp_size,
        )
        full_stage_layer_counts = build_pp_stage_layer_counts(
            total_layers=self.original_total_layers,
            pp_size=pp_size,
            vpp_size=vpp_size,
        )
        stage_param_stats = self._gather_stage_param_stats()
        dp_world_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
        dp_comm_cfg = get_dp_allreduce_comm_config()
        simulation_context = {
            "pp_size": pp_size,
            "vpp_size": vpp_size,
            "runtime_stage_layer_counts": runtime_stage_layer_counts,
            "full_stage_layer_counts": full_stage_layer_counts,
            "dp_world_size": dp_world_size,
            "dp_allreduce_bandwidth_gbps": dp_comm_cfg["bandwidth_gbps"],
            "dp_allreduce_latency_s": dp_comm_cfg["latency_s"],
            "stage_param_stats": [asdict(stats) for stats in stage_param_stats],
            "_stage_param_stats": stage_param_stats,
        }
        self._log(
            "runtime simulator ready: "
            f"pp={pp_size} vpp={vpp_size} dp={dp_world_size} "
            f"runtime_stage_layers={runtime_stage_layer_counts} "
            f"full_stage_layers={full_stage_layer_counts} "
            f"dp_bw_gbps={dp_comm_cfg['bandwidth_gbps']:.2f} "
            f"dp_latency_s={dp_comm_cfg['latency_s']:.6f}"
        )
        return simulation_context

    def _limit_iterator(self, data_iterator: Iterable, num_items: int):
        if isinstance(data_iterator, list):
            return [itertools.islice(it, num_items) for it in data_iterator]
        return itertools.islice(data_iterator, num_items)

    def _build_forward_step(self, test_case: InputTestCase):
        def forward_step(data_iterator, model, checkpoint_activations_microbatch=None):
            self._microbatch_counter += 1
            log_this = self._microbatch_counter == 1 or (
                self._log_microbatch_every > 0
                and self._microbatch_counter % self._log_microbatch_every == 0
            )
            data_iter = data_iterator
            unwrapped = unwrap_model(model)
            if isinstance(data_iterator, list):
                vp_stage = getattr(unwrapped, "vp_stage", 0)
                data_iter = data_iterator[vp_stage]

            if log_this:
                self._log(
                    f"iter {self._current_iteration} microbatch {self._microbatch_counter}: "
                    "fetching batch"
                )
            micro_batch = next(data_iter)
            micro_batch = micro_batch.to(torch.cuda.current_device())
            micro_batch = micro_batch.contiguous()
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"].to(bool)
            position_ids = micro_batch["position_ids"]
            labels, label_mask = self._build_next_token_labels(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            if test_case.shape == "bshd":
                self._maybe_init_tp_overlap(tokens=int(attention_mask.sum().item()))
            elif not self._tp_overlap_disabled_logged and getattr(
                self.tf_config, "tp_comm_overlap", False
            ):
                self._log(
                    "tp overlap disabled for this run: THD/variable sequence lengths "
                    "are not supported by TransformerEngine user buffers"
                )
                self._tp_overlap_disabled_logged = True
            if log_this and self._log_microbatch_io:
                self._log(
                    "iter {iteration} microbatch {mb}: "
                    "input_ids={input_shape} attn_mask={mask_shape} pos_ids={pos_shape}".format(
                        iteration=self._current_iteration,
                        mb=self._microbatch_counter,
                        input_shape=tuple(input_ids.shape),
                        mask_shape=tuple(attention_mask.shape),
                        pos_shape=tuple(position_ids.shape),
                    )
                )

            if log_this:
                self._log(
                    f"iter {self._current_iteration} microbatch {self._microbatch_counter}: "
                    "model forward start"
                )
            if self.use_fused_kernels:
                output = self._forward_fused_fn(
                    model=model,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    labels_mask=label_mask,
                    temperature=1.0,
                    multi_modal_inputs={},
                )
            else:

                def logits_processor(logits, label, label_mask):
                    log_probs = vocab_parallel_log_probs_from_logits(logits, label)
                    return {"log_probs": log_probs.masked_fill(~label_mask, 0.0)}

                output = self._forward_fn(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    multi_modal_inputs={},
                    logits_processor=logits_processor,
                    logits_processor_args={
                        "label": labels,
                        "label_mask": label_mask,
                    },
                    data_format=test_case.shape,
                )
            if log_this:
                self._log(
                    f"iter {self._current_iteration} microbatch {self._microbatch_counter}: "
                    "model forward done"
                )

            def loss_func(output_tensor, non_loss_data=False):
                log_probs = self._extract_log_probs(output_tensor)
                if log_probs is not None:
                    if non_loss_data:
                        return {"log_probs": log_probs}
                    valid_log_probs = log_probs.masked_select(label_mask)
                    if valid_log_probs.numel() == 0:
                        loss = log_probs.float().sum() * 0.0
                    else:
                        loss = -valid_log_probs.float().mean()
                    return loss, {"loss": loss.detach()}
                if non_loss_data:
                    return {"loss": output_tensor}
                valid_loss = output_tensor.masked_select(label_mask)
                if valid_loss.numel() == 0:
                    loss = output_tensor.float().sum() * 0.0
                else:
                    loss = valid_loss.float().mean()
                return loss, {"loss": loss.detach()}

            return output, loss_func

        return forward_step

    def _collect_batch_seqlens(
        self, test_case: InputTestCase, num_microbatches: int
    ) -> list[int]:
        micro_batches = self.datasets.data[test_case][:num_microbatches]
        batch_seqlens: list[int] = []
        for micro_batch in micro_batches:
            attention_mask = micro_batch["attention_mask"]
            if attention_mask.is_floating_point():
                attention_mask = attention_mask > 0
            seqlens = attention_mask.to(torch.int64).sum(dim=1).tolist()
            batch_seqlens.extend(seqlens)
        return batch_seqlens

    def _compute_perf_metrics(
        self, batch_seqlens: list[int], delta_time: float, world_size: int
    ) -> dict:
        total_tokens = sum(batch_seqlens)
        total_sequences = len(batch_seqlens)
        throughput_tokens_s = (
            total_tokens / delta_time if delta_time > 0 else float("inf")
        )
        throughput_tokens_s_per_gpu = throughput_tokens_s / world_size
        throughput_seqs_s = (
            total_sequences / delta_time if delta_time > 0 else float("inf")
        )
        throughput_seqs_s_per_gpu = throughput_seqs_s / world_size

        estimated_flops, promised_flops = self.flops_counter.estimate_flops(
            batch_seqlens, delta_time
        )
        if estimated_flops == 0.0:
            estimated_flops = self._estimate_generic_flops(batch_seqlens, delta_time)
        else:
            estimated_flops *= self.flops_layer_scale
        if promised_flops in (0, float("inf")):
            promised_flops = GPU_PEAK_FLOPS / 1e12
        if promised_flops == 0:
            mfu = 0.0
        else:
            mfu = estimated_flops / promised_flops / world_size

        return {
            "total_tokens": total_tokens,
            "total_sequences": total_sequences,
            "time_s": delta_time,
            "throughput_tokens_s": throughput_tokens_s,
            "throughput_tokens_s_per_gpu": throughput_tokens_s_per_gpu,
            "throughput_sequences_s": throughput_seqs_s,
            "throughput_sequences_s_per_gpu": throughput_seqs_s_per_gpu,
            "mfu": mfu,
        }

    def _estimate_generic_flops(
        self, batch_seqlens: list[int], delta_time: float
    ) -> float:
        if delta_time <= 0:
            return float("inf")
        config = getattr(self.hf_config, "text_config", self.hf_config)
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_hidden_layers = self.runtime_total_layers
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(
            config, "num_key_value_heads", num_attention_heads
        )
        intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)

        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        tokens_sum = sum(batch_seqlens)
        seqlen_square_sum = sum(seqlen * seqlen for seqlen in batch_seqlens)

        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (
            q_size + k_size + v_size + num_attention_heads * head_dim
        )
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        dense_N_flops = 6 * dense_N * tokens_sum
        attn_qkv_flops = (
            12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers
        )
        flops_all_token = dense_N_flops + attn_qkv_flops
        return flops_all_token * (1.0 / delta_time) / 1e12

    def _maybe_init_tp_overlap(self, tokens: int):
        if mpu.get_tensor_model_parallel_world_size() <= 1 or not getattr(
            self.tf_config, "tp_comm_overlap", False
        ):
            return
        if tokens <= 0:
            return
        if self._tp_overlap_tokens == tokens:
            return
        if self._tp_overlap_tokens is not None:
            destroy_ub()
        initialize_tp_communicators(
            tp_comm_overlap_cfg=self.tp_comm_overlap_cfg,
            op_name="ColumnParallelLinear",
            tokens=tokens,
            hidden_size=self.hf_config.hidden_size,
            inputs_are_cp_sharded=False,
        )
        self._tp_overlap_tokens = tokens
        self._log(
            "tp overlap initialized: "
            f"tokens={tokens} "
            f"tp={mpu.get_tensor_model_parallel_world_size()} "
            f"cp={mpu.get_context_parallel_world_size()} "
            f"cfg={self.tp_comm_overlap_cfg}"
        )

    def _maybe_destroy_tp_overlap(self):
        if mpu.get_tensor_model_parallel_world_size() <= 1 or not getattr(
            self.tf_config, "tp_comm_overlap", False
        ):
            return
        if self._tp_overlap_tokens is not None:
            destroy_ub()
            self._tp_overlap_tokens = None
            self._log("tp overlap user buffers destroyed")

    def _summarize_rank_memory(self, valid_metrics: list[dict]) -> list[dict]:
        summary_by_rank: dict[int, dict] = {}
        for metrics in valid_metrics:
            for rank_stats in metrics.get("memory_by_rank", []):
                rank = rank_stats["rank"]
                if rank not in summary_by_rank:
                    summary_by_rank[rank] = {
                        "rank": rank,
                        "world_size": rank_stats["world_size"],
                        "device_index": rank_stats["device_index"],
                        "peak_allocated_bytes": rank_stats["peak_allocated_bytes"],
                        "peak_reserved_bytes": rank_stats["peak_reserved_bytes"],
                        "real_detected_bytes": rank_stats["real_detected_bytes"],
                        "total_device_bytes": rank_stats["total_device_bytes"],
                    }
                else:
                    summary_by_rank[rank]["peak_allocated_bytes"] = max(
                        summary_by_rank[rank]["peak_allocated_bytes"],
                        rank_stats["peak_allocated_bytes"],
                    )
                    summary_by_rank[rank]["peak_reserved_bytes"] = max(
                        summary_by_rank[rank]["peak_reserved_bytes"],
                        rank_stats["peak_reserved_bytes"],
                    )
                    summary_by_rank[rank]["real_detected_bytes"] = max(
                        summary_by_rank[rank]["real_detected_bytes"],
                        rank_stats["real_detected_bytes"],
                    )

        summary_list = []
        for rank in sorted(summary_by_rank.keys()):
            item = summary_by_rank[rank]
            item["peak_allocated"] = get_memory_str(item["peak_allocated_bytes"])
            item["peak_reserved"] = get_memory_str(item["peak_reserved_bytes"])
            item["real_detected"] = get_memory_str(item["real_detected_bytes"])
            item["total_device_memory"] = get_memory_str(item["total_device_bytes"])
            summary_list.append(item)
        return summary_list

    def run_pipeline(
        self,
        num_test_cases: Optional[int] = None,
        run_one_data: bool = False,
        max_iterations: int = 10,
        warmup_iterations: int = 3,
        output_dir: Optional[str] = None,
    ):
        if num_test_cases is None:
            test_case_idxs = list(range(len(self.test_cases)))
        else:
            test_case_idxs = list(range(min(num_test_cases, len(self.test_cases))))

        forward_backward_func = get_forward_backward_func()
        metrics_by_test_case = []
        world_size = torch.distributed.get_world_size()
        run_report = {
            "model_name": self.model_name,
            "world_size": world_size,
            "original_total_layers": self.original_total_layers,
            "runtime_total_layers": self.runtime_total_layers,
            "runtime_layers_per_pp_rank": self.runtime_layers_per_pp_rank,
            "simulation": {
                key: value
                for key, value in self.simulation_context.items()
                if key != "_stage_param_stats"
            },
            "num_test_cases": len(test_case_idxs),
            "run_one_data": run_one_data,
            "max_iterations": max_iterations,
            "warmup_iterations": warmup_iterations,
            "test_cases": [],
        }
        self._log(
            "runtime pipeline start: "
            f"world_size={world_size} "
            f"num_test_cases={len(test_case_idxs)} "
            f"run_one_data={run_one_data} "
            f"max_iterations={max_iterations} "
            f"warmup_iterations={warmup_iterations}"
        )

        for idx in test_case_idxs:
            test_case = self.test_cases[idx]
            num_microbatches = len(self.datasets.data[test_case])
            if run_one_data:
                num_microbatches = 1
            test_case_report = {
                "test_case_idx": idx,
                "test_case": asdict(test_case),
                "num_microbatches": num_microbatches,
                "iterations": [],
            }

            batch_seqlens = self._collect_batch_seqlens(test_case, num_microbatches)
            if batch_seqlens:
                min_seqlen = min(batch_seqlens)
                max_seqlen = max(batch_seqlens)
                avg_seqlen = sum(batch_seqlens) / len(batch_seqlens)
            else:
                min_seqlen = max_seqlen = avg_seqlen = 0
            self._log(
                "test case start: "
                f"idx={idx} "
                f"{test_case} "
                f"microbatches={num_microbatches} "
                f"batch_seqlens(min/avg/max)={min_seqlen}/{avg_seqlen:.1f}/{max_seqlen}"
            )
            iteration_metrics = []
            for iteration in range(max_iterations):
                data_iterator = self.datasets.get_batch_generator(test_case)
                if run_one_data:
                    data_iterator = self._limit_iterator(
                        data_iterator, num_microbatches
                    )

                self._current_iteration = iteration
                self._microbatch_counter = 0
                self._log(f"iter {iteration}: entering pre-barrier")
                torch.distributed.barrier()
                self._log(f"iter {iteration}: exited pre-barrier")
                torch.cuda.synchronize()
                reset_peak_memory_stats(synchronize=False)
                self._reset_stage_timers()
                self._log(f"iter {iteration}: starting forward/backward")
                start_time = time.perf_counter()
                forward_backward_func(
                    forward_step_func=self._build_forward_step(test_case),
                    data_iterator=data_iterator,
                    model=self.model,
                    num_microbatches=num_microbatches,
                    seq_length=test_case.seqlen,
                    micro_batch_size=test_case.micro_batch_size,
                    decoder_seq_length=test_case.seqlen,
                    forward_only=False,
                )
                torch.cuda.synchronize()
                self._log(f"iter {iteration}: finished forward/backward")
                self._log(f"iter {iteration}: entering post-barrier")
                torch.distributed.barrier()
                self._log(f"iter {iteration}: exited post-barrier")
                delta_time = time.perf_counter() - start_time
                memory_by_rank = get_all_rank_peak_memory_stats(synchronize=False)
                stage_timing_stats = self._gather_stage_timing_stats(num_microbatches)

                metrics = self._compute_perf_metrics(
                    batch_seqlens, delta_time, world_size
                )
                simulation = simulate_full_iteration(
                    observed_iteration_time_s=delta_time,
                    num_microbatches=num_microbatches,
                    full_stage_layer_counts=self.simulation_context[
                        "full_stage_layer_counts"
                    ],
                    runtime_stage_layer_counts=self.simulation_context[
                        "runtime_stage_layer_counts"
                    ],
                    stage_param_stats=self.simulation_context["_stage_param_stats"],
                    stage_timing_stats=stage_timing_stats,
                    dp_world_size=self.simulation_context["dp_world_size"],
                    include_runtime_dp_in_observed_time=self.wrap_with_ddp,
                    bandwidth_gbps=self.simulation_context[
                        "dp_allreduce_bandwidth_gbps"
                    ],
                    latency_s=self.simulation_context["dp_allreduce_latency_s"],
                )
                simulated_perf_metrics = self._compute_perf_metrics(
                    batch_seqlens, simulation["simulated_time_s"], world_size
                )
                metrics["iteration"] = iteration
                metrics["memory_by_rank"] = memory_by_rank
                metrics["simulation"] = simulation
                metrics["simulated_time_s"] = simulation["simulated_time_s"]
                metrics["simulated_pp_compute_time_s"] = simulation[
                    "simulated_pp_compute_time_s"
                ]
                metrics["simulated_dp_allreduce_time_s"] = simulation[
                    "simulated_dp_allreduce_time_s"
                ]
                metrics["simulated_throughput_tokens_s"] = simulated_perf_metrics[
                    "throughput_tokens_s"
                ]
                metrics["simulated_throughput_tokens_s_per_gpu"] = (
                    simulated_perf_metrics["throughput_tokens_s_per_gpu"]
                )
                metrics["simulated_throughput_sequences_s"] = simulated_perf_metrics[
                    "throughput_sequences_s"
                ]
                metrics["simulated_throughput_sequences_s_per_gpu"] = (
                    simulated_perf_metrics["throughput_sequences_s_per_gpu"]
                )
                metrics["simulated_mfu"] = simulated_perf_metrics["mfu"]
                iteration_metrics.append(metrics)
                test_case_report["iterations"].append(metrics)
                max_peak_allocated = max(
                    rank_stats["peak_allocated_bytes"] for rank_stats in memory_by_rank
                )
                max_peak_reserved = max(
                    rank_stats["peak_reserved_bytes"] for rank_stats in memory_by_rank
                )
                max_real_detected = max(
                    rank_stats["real_detected_bytes"] for rank_stats in memory_by_rank
                )
                self._log(
                    "iter {iteration} metrics: time_s={time_s:.4f} "
                    "sim_time_s={sim_time_s:.4f} sim_pp_s={sim_pp_s:.4f} sim_dp_s={sim_dp_s:.4f} "
                    "tokens_per_s={tps:.2f} tokens_per_s_per_gpu={tps_pg:.2f} "
                    "sim_tokens_per_s={sim_tps:.2f} sim_tokens_per_s_per_gpu={sim_tps_pg:.2f} "
                    "seqs_per_s={sps:.2f} seqs_per_s_per_gpu={sps_pg:.2f} mfu={mfu:.4f} "
                    "sim_mfu={sim_mfu:.4f} "
                    "peak_alloc={peak_alloc} peak_reserved={peak_reserved} real_detected={real_detected}".format(
                        iteration=iteration,
                        time_s=metrics["time_s"],
                        sim_time_s=metrics["simulated_time_s"],
                        sim_pp_s=metrics["simulated_pp_compute_time_s"],
                        sim_dp_s=metrics["simulated_dp_allreduce_time_s"],
                        tps=metrics["throughput_tokens_s"],
                        tps_pg=metrics["throughput_tokens_s_per_gpu"],
                        sim_tps=metrics["simulated_throughput_tokens_s"],
                        sim_tps_pg=metrics["simulated_throughput_tokens_s_per_gpu"],
                        sps=metrics["throughput_sequences_s"],
                        sps_pg=metrics["throughput_sequences_s_per_gpu"],
                        mfu=metrics["mfu"],
                        sim_mfu=metrics["simulated_mfu"],
                        peak_alloc=get_memory_str(max_peak_allocated),
                        peak_reserved=get_memory_str(max_peak_reserved),
                        real_detected=get_memory_str(max_real_detected),
                    )
                )
            self._maybe_destroy_tp_overlap()

            warmup_count = min(warmup_iterations, max_iterations)
            valid_metrics = iteration_metrics[warmup_count:]
            if not valid_metrics:
                valid_metrics = iteration_metrics

            def _avg(key: str) -> float:
                return sum(m[key] for m in valid_metrics) / len(valid_metrics)

            summary = {
                "test_case_idx": idx,
                "num_microbatches": num_microbatches,
                "max_iterations": max_iterations,
                "warmup_iterations": warmup_count,
                "total_tokens": iteration_metrics[-1]["total_tokens"],
                "total_sequences": iteration_metrics[-1]["total_sequences"],
                "time_s": _avg("time_s"),
                "throughput_tokens_s": _avg("throughput_tokens_s"),
                "throughput_tokens_s_per_gpu": _avg("throughput_tokens_s_per_gpu"),
                "throughput_sequences_s": _avg("throughput_sequences_s"),
                "throughput_sequences_s_per_gpu": _avg(
                    "throughput_sequences_s_per_gpu"
                ),
                "mfu": _avg("mfu"),
                "simulated_time_s": _avg("simulated_time_s"),
                "simulated_pp_compute_time_s": _avg("simulated_pp_compute_time_s"),
                "simulated_dp_allreduce_time_s": _avg("simulated_dp_allreduce_time_s"),
                "simulated_throughput_tokens_s": _avg("simulated_throughput_tokens_s"),
                "simulated_throughput_tokens_s_per_gpu": _avg(
                    "simulated_throughput_tokens_s_per_gpu"
                ),
                "simulated_throughput_sequences_s": _avg(
                    "simulated_throughput_sequences_s"
                ),
                "simulated_throughput_sequences_s_per_gpu": _avg(
                    "simulated_throughput_sequences_s_per_gpu"
                ),
                "simulated_mfu": _avg("simulated_mfu"),
                "simulation": valid_metrics[-1]["simulation"],
                "memory_by_rank": self._summarize_rank_memory(valid_metrics),
            }
            metrics_by_test_case.append(summary)
            test_case_report["summary"] = summary
            run_report["test_cases"].append(test_case_report)
            max_case_peak_allocated = max(
                rank_stats["peak_allocated_bytes"]
                for rank_stats in summary["memory_by_rank"]
            )
            max_case_peak_reserved = max(
                rank_stats["peak_reserved_bytes"]
                for rank_stats in summary["memory_by_rank"]
            )
            max_case_real_detected = max(
                rank_stats["real_detected_bytes"]
                for rank_stats in summary["memory_by_rank"]
            )
            log_rank0(
                "[runtime] test_case_idx={idx} microbatches={num_microbatches} "
                "iters={max_iterations} warmup={warmup} "
                "tokens={tokens} seqs={seqs} time_s={time_s:.4f} "
                "sim_time_s={sim_time_s:.4f} sim_pp_s={sim_pp_s:.4f} sim_dp_s={sim_dp_s:.4f} "
                "tokens_per_s={tps:.2f} tokens_per_s_per_gpu={tps_pg:.2f} "
                "sim_tokens_per_s={sim_tps:.2f} sim_tokens_per_s_per_gpu={sim_tps_pg:.2f} "
                "seqs_per_s={sps:.2f} seqs_per_s_per_gpu={sps_pg:.2f} mfu={mfu:.4f} "
                "sim_mfu={sim_mfu:.4f} "
                "peak_alloc={peak_alloc} peak_reserved={peak_reserved} real_detected={real_detected}".format(
                    idx=idx,
                    num_microbatches=num_microbatches,
                    max_iterations=max_iterations,
                    warmup=warmup_count,
                    tokens=summary["total_tokens"],
                    seqs=summary["total_sequences"],
                    time_s=summary["time_s"],
                    sim_time_s=summary["simulated_time_s"],
                    sim_pp_s=summary["simulated_pp_compute_time_s"],
                    sim_dp_s=summary["simulated_dp_allreduce_time_s"],
                    tps=summary["throughput_tokens_s"],
                    tps_pg=summary["throughput_tokens_s_per_gpu"],
                    sim_tps=summary["simulated_throughput_tokens_s"],
                    sim_tps_pg=summary["simulated_throughput_tokens_s_per_gpu"],
                    sps=summary["throughput_sequences_s"],
                    sps_pg=summary["throughput_sequences_s_per_gpu"],
                    mfu=summary["mfu"],
                    sim_mfu=summary["simulated_mfu"],
                    peak_alloc=get_memory_str(max_case_peak_allocated),
                    peak_reserved=get_memory_str(max_case_peak_reserved),
                    real_detected=get_memory_str(max_case_real_detected),
                )
            )

        self.latest_run_report = run_report
        if output_dir is not None and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "runtime_baseline.json")
            with open(output_path, "w") as fp:
                json.dump(run_report, fp, indent=2)
            log_rank0(f"runtime baseline report saved to {output_path}")

        return metrics_by_test_case

import itertools
import time
from typing import Iterable, Optional

import torch
from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from transformers import PretrainedConfig

from AutoTuner.testbench.ops.gpt_model import GPTModelForTest
from AutoTuner.utils.config import (
    get_hf_model_config,
    get_mcore_model_config_from_hf_config,
)
from AutoTuner.utils.gpu_info import GPU_PEAK_FLOPS
from AutoTuner.utils.model_inputs import DataSets, get_thd_model_input_from_bshd
from AutoTuner.utils.structs import InputTestCase
from verl.utils.flops_counter import FlopsCounter
from verl.utils.megatron_utils import get_model, unwrap_model


class RuntimeLauncher:
    def __init__(
        self,
        model_name: str,
        test_cases: list[InputTestCase],
        override_model_kwargs: dict,
        override_tf_config_kwargs: dict,
        share_embeddings_and_output_weights: Optional[bool] = None,
        wrap_with_ddp: bool = True,
        use_distributed_optimizer: bool = False,
        fix_compute_amount: bool = True,
    ) -> None:
        self.model_name = model_name
        self.test_cases = test_cases

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

        if share_embeddings_and_output_weights is None:
            share_embeddings_and_output_weights = bool(
                getattr(self.hf_config, "tie_word_embeddings", False)
            )
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

        assert torch.distributed.is_initialized(), "torch.distributed is not initialized"
        self.tp_group = mpu.get_tensor_model_parallel_group()

        self.datasets = DataSets(
            self.hf_config,
            self.test_cases,
            fix_compute_amount=fix_compute_amount,
            use_dynamic_bsz_balance=True,
            vpp_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
        )

        self.flops_counter = FlopsCounter(self.hf_config)

        self.model = self._build_model(
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=use_distributed_optimizer,
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

        return get_model(
            model_provider,
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=use_distributed_optimizer,
            transformer_config=self.tf_config,
        )

    def _limit_iterator(self, data_iterator: Iterable, num_items: int):
        if isinstance(data_iterator, list):
            return [itertools.islice(it, num_items) for it in data_iterator]
        return itertools.islice(data_iterator, num_items)

    def _build_forward_step(self, test_case: InputTestCase):
        def forward_step(data_iterator, model, checkpoint_activations_microbatch=None):
            data_iter = data_iterator
            unwrapped = unwrap_model(model)
            if isinstance(data_iterator, list):
                vp_stage = getattr(unwrapped, "vp_stage", 0)
                data_iter = data_iterator[vp_stage]

            micro_batch = next(data_iter)
            micro_batch = micro_batch.to(torch.cuda.current_device())
            micro_batch = micro_batch.contiguous()

            (
                input_ids_rmpad,
                attention_mask,
                position_ids_rmpad,
                packed_seq_params,
            ) = get_thd_model_input_from_bshd(micro_batch)

            output = model(
                input_ids_rmpad,
                position_ids_rmpad,
                attention_mask,
                None,
                None,
                packed_seq_params,
                None,
                None,
            )

            def loss_func(output_tensor, non_loss_data=False):
                if non_loss_data:
                    return {"logits": output_tensor}
                loss = output_tensor.float().mean()
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
        throughput_tokens_s = total_tokens / delta_time if delta_time > 0 else float("inf")
        throughput_tokens_s_per_gpu = throughput_tokens_s / world_size
        throughput_seqs_s = total_sequences / delta_time if delta_time > 0 else float("inf")
        throughput_seqs_s_per_gpu = throughput_seqs_s / world_size

        estimated_flops, promised_flops = self.flops_counter.estimate_flops(
            batch_seqlens, delta_time
        )
        if estimated_flops == 0.0:
            estimated_flops = self._estimate_generic_flops(batch_seqlens, delta_time)
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
        num_hidden_layers = config.num_hidden_layers
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
        intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)

        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        tokens_sum = sum(batch_seqlens)
        seqlen_square_sum = sum(seqlen * seqlen for seqlen in batch_seqlens)

        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        dense_N_flops = 6 * dense_N * tokens_sum
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers
        flops_all_token = dense_N_flops + attn_qkv_flops
        return flops_all_token * (1.0 / delta_time) / 1e12

    def run_pipeline(
        self,
        num_test_cases: Optional[int] = None,
        run_one_data: bool = False,
        max_iterations: int = 10,
        warmup_iterations: int = 3,
    ):
        if num_test_cases is None:
            test_case_idxs = list(range(len(self.test_cases)))
        else:
            test_case_idxs = list(range(min(num_test_cases, len(self.test_cases))))

        forward_backward_func = get_forward_backward_func()
        metrics_by_test_case = []
        world_size = torch.distributed.get_world_size()

        for idx in test_case_idxs:
            test_case = self.test_cases[idx]
            num_microbatches = len(self.datasets.data[test_case])
            if run_one_data:
                num_microbatches = 1

            batch_seqlens = self._collect_batch_seqlens(test_case, num_microbatches)
            iteration_metrics = []
            for iteration in range(max_iterations):
                data_iterator = self.datasets.get_batch_generator(test_case)
                if run_one_data:
                    data_iterator = self._limit_iterator(data_iterator, num_microbatches)

                torch.distributed.barrier()
                torch.cuda.synchronize()
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
                torch.distributed.barrier()
                delta_time = time.perf_counter() - start_time

                metrics = self._compute_perf_metrics(
                    batch_seqlens, delta_time, world_size
                )
                metrics["iteration"] = iteration
                iteration_metrics.append(metrics)

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
            }
            metrics_by_test_case.append(summary)
            if torch.distributed.get_rank() == 0:
                print(
                    "[runtime] test_case_idx={idx} microbatches={num_microbatches} "
                    "iters={max_iterations} warmup={warmup} "
                    "tokens={tokens} seqs={seqs} time_s={time_s:.4f} "
                    "tokens_per_s={tps:.2f} tokens_per_s_per_gpu={tps_pg:.2f} "
                    "seqs_per_s={sps:.2f} seqs_per_s_per_gpu={sps_pg:.2f} mfu={mfu:.4f}".format(
                        idx=idx,
                        num_microbatches=num_microbatches,
                        max_iterations=max_iterations,
                        warmup=warmup_count,
                        tokens=summary["total_tokens"],
                        seqs=summary["total_sequences"],
                        time_s=summary["time_s"],
                        tps=summary["throughput_tokens_s"],
                        tps_pg=summary["throughput_tokens_s_per_gpu"],
                        sps=summary["throughput_sequences_s"],
                        sps_pg=summary["throughput_sequences_s_per_gpu"],
                        mfu=summary["mfu"],
                    )
                )

        return metrics_by_test_case

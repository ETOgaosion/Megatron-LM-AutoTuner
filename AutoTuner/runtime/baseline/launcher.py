import itertools
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
from AutoTuner.utils.model_inputs import DataSets, get_thd_model_input_from_bshd
from AutoTuner.utils.structs import InputTestCase
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

    def run_pipeline(self, test_case_idxs: Optional[list[int]] = None, run_one_data: bool = False):
        if test_case_idxs is None:
            test_case_idxs = list(range(len(self.test_cases)))

        forward_backward_func = get_forward_backward_func()

        for idx in test_case_idxs:
            test_case = self.test_cases[idx]
            data_iterator = self.datasets.get_batch_generator(test_case)
            num_microbatches = len(self.datasets.data[test_case])
            if run_one_data:
                num_microbatches = 1
                data_iterator = self._limit_iterator(data_iterator, num_microbatches)

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

import dataclasses
import os
from typing import Any, Dict, Optional

import torch
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tensordict import TensorDict
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.testbench.ops.atten_with_cp import AttnFuncWithCPAndKVP2PForTest
from AutoTuner.testbench.ops_test.test_with_hiddens import TestWithHiddenInputs
from AutoTuner.testbench.profile.configs.config_struct import ProfileMode
from AutoTuner.utils.memory import MemoryTrackerContext, get_memory_str
from AutoTuner.utils.structs import InputTestCase

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestAttnFuncWithCPAndKVP2P(TestWithHiddenInputs):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        profile_mode: int = 0,
        warmup_iters: int = 2,
        profile_iters: int = 2,
        theoretical_flops: bool = False,
        theoretical_activations: bool = False,
        tp_comm_overlap_cfg: str = None,
        #
        pg_collection=None,
    ):
        super().__init__(
            hf_config=hf_config,
            tf_config=tf_config,
            profile_mode=profile_mode,
            warmup_iters=warmup_iters,
            profile_iters=profile_iters,
            tp_group=tp_group,
            theoretical_flops=theoretical_flops,
            theoretical_activations=theoretical_activations,
            tp_comm_overlap_cfg=tp_comm_overlap_cfg,
        )
        # Initialize process group collection
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.kept_packed_seq_params = set(
            field.name for field in dataclasses.fields(PackedSeqParams)
        )
        self.module_name = "AttnFuncWithCPAndKVP2P"

        self.self_attention = SelfAttention(
            tf_config,
            get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

        if profile_mode == ProfileMode.collect_data:
            with MemoryTrackerContext(
                "AttnFuncWithCPAndKVP2P init"
            ) as memory_tracker_ctx:
                self.op = AttnFuncWithCPAndKVP2PForTest(tf_config)

            detailed_mem_report = memory_tracker_ctx.get_result()

            # TODO: theoretical weight memory
            estimated_weight_mem_bytes = 0
            estimated_weight_mem_str = get_memory_str(
                estimated_weight_mem_bytes, human_readable=True
            )
            detailed_mem_report["estimated_peak_mem_diff"] = estimated_weight_mem_str
            self.memory_db["weights"][self.module_name] = detailed_mem_report

        else:
            self.op = AttnFuncWithCPAndKVP2PForTest(tf_config)

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

    @override
    def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
        """
        args provided to forward:
            is_training,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            dropout_p,
            cp_group,
            cp_global_ranks,
            cp_stream,
            cp_comm_type,
        """
        micro_batch = micro_batch.to(torch.cuda.current_device())
        micro_batch = micro_batch.contiguous()
        (
            hidden_states,
            attention_mask,
            rotary_pos_emb,
            packed_seq_params,
            sequence_len_offset,
        ) = super().prepare_input(test_case, micro_batch)
        q, k, v = self.self_attention.get_query_key_value_tensors(hidden_states)
        if getattr(packed_seq_params, "qkv_format", "bshd") == "thd":
            q = q.reshape(q.shape[0] * q.shape[1], q.shape[2], q.shape[3])
            k = k.reshape(k.shape[0] * k.shape[1], k.shape[2], k.shape[3])
            v = v.reshape(v.shape[0] * v.shape[1], v.shape[2], v.shape[3])
        query_layer, key_layer, value_layer = [
            x.contiguous() if not x.is_contiguous() else x for x in [q, k, v]
        ]

        packed_seq_kwargs = (
            {
                key: getattr(packed_seq_params, key)
                for key in self.kept_packed_seq_params
            }
            if packed_seq_params is not None
            else {}
        )
        qkv_format = packed_seq_kwargs.get("qkv_format", "sbhd")
        import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils

        # get qkv's memory layout
        if all(
            isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]
        ):
            (
                qkv_layout,
                query_layer._data,
                key_layer._data,
                value_layer._data,
                q_format,
                kv_format,
            ) = dpa_utils.get_qkv_layout(
                query_layer._data,
                key_layer._data,
                value_layer._data,
                qkv_format=qkv_format,
                inference_params=None,
            )
        else:
            (
                qkv_layout,
                query_layer,
                key_layer,
                value_layer,
                q_format,
                kv_format,
            ) = dpa_utils.get_qkv_layout(
                query_layer,
                key_layer,
                value_layer,
                qkv_format=qkv_format,
                inference_params=None,
            )

        extra_kwargs: dict[str, Any] = {}
        # Get CP related args
        if self.tf_config.context_parallel_size > 1:
            cp_stream = torch.cuda.Stream()
            extra_kwargs["cp_group"] = self.pg_collection.cp
            extra_kwargs["cp_global_ranks"] = torch.distributed.get_process_group_ranks(
                self.pg_collection.cp
            )
            extra_kwargs["cp_stream"] = cp_stream
            extra_kwargs["cp_comm_type"] = "p2p"
        # Expand CP-related kwargs in a stable order (values, not dict keys).
        # If context-parallel isn't used, the values will be None.
        return (
            query_layer,
            key_layer,
            value_layer,
            True,
            packed_seq_params.cu_seqlens_q,
            packed_seq_params.cu_seqlens_kv,
            packed_seq_params.max_seqlen_q,
            packed_seq_params.max_seqlen_kv,
            packed_seq_params.cu_seqlens_q_padded,
            packed_seq_params.cu_seqlens_kv_padded,
            0.0,
            extra_kwargs.get("cp_group", None),
            extra_kwargs.get("cp_global_ranks", None),
            extra_kwargs.get("cp_stream", None),
            extra_kwargs.get("cp_comm_type", None),
            qkv_format,
        )

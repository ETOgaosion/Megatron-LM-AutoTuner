import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from AutoTuner.testbench.ops.gpt_model import GPTModelForTest
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tensordict import TensorDict
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.utils.memory import MemoryTracker, MemoryTrackerContext, get_memory_str
from AutoTuner.utils.model_inputs import get_thd_model_input_from_bshd
from AutoTuner.utils.nested_dict import NestedDict
from AutoTuner.utils.structs import InputTestCase
from AutoTuner.utils.timing import Timer, TimerContext

from ..profile.configs.config_struct import ProfileMode
from .common import TestCommon

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestGPTModel(TestCommon):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        transformer_layer_spec: ModuleSpec,
        scatter_to_sequence_parallel: bool = True,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
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
            profile_mode=profile_mode,
            warmup_iters=warmup_iters,
            theoretical_flops=theoretical_flops,
            theoretical_activations=theoretical_activations,
            tp_comm_overlap_cfg=tp_comm_overlap_cfg,
        )
        self.module_name = "GPTModel"
        self.transformer_layer_spec = transformer_layer_spec
        self.scatter_to_sequence_parallel = scatter_to_sequence_parallel
        self.tp_group = tp_group
        if profile_mode == ProfileMode.collect_data:
            with MemoryTrackerContext(self.module_name) as memory_tracker_ctx:
                self.op = GPTModelForTest(
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
            self.op = GPTModelForTest(
                tf_config=tf_config,
                hf_config=hf_config,
                transformer_layer_spec=transformer_layer_spec,
                hook_activation=False,
                scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                tp_group=tp_group,
                **kwargs,
            )
    def _calculate_weight_memory(self, detailed_mem_report: Dict, tf_config: TransformerConfig, hf_config: PretrainedConfig):
        hidden_size = hf_config.hidden_size
        num_layers = hf_config.num_hidden_layers
        vocab_size = hf_config.vocab_size
        ffn_hidden_size = hf_config.intermediate_size
        num_attention_heads = hf_config.num_attention_heads
        kv_channels = hidden_size // num_attention_heads
        
        tp_size = mpu.get_tensor_model_parallel_world_size()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        cp_size = tf_config.context_parallel_size
        dtype = next(self.op.parameters()).dtype
        bytes_per_param = torch.tensor([], dtype=dtype).element_size()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        estimated_weight_bytes = 0

        # Embedding (only on first pipeline stage)
        if pp_rank == 0:
            # Word embeddings
            estimated_weight_bytes += (vocab_size // tp_size) * hidden_size * bytes_per_param
            # Position embeddings
            estimated_weight_bytes += hf_config.max_position_embeddings * hidden_size * bytes_per_param
            # Tokentype embeddings
            if hasattr(self.op.embedding, 'tokentype_embeddings') and self.op.embedding.tokentype_embeddings is not None:
                num_tokentypes = self.op.embedding.tokentype_embeddings.num_embeddings
                estimated_weight_bytes += num_tokentypes * hidden_size * bytes_per_param
        # Transformer layers
        layers_per_rank = num_layers // pp_size
        start_layer = pp_rank * layers_per_rank
        end_layer = start_layer + layers_per_rank
        # input layer norm
        if tf_config.normalization == "RMSNorm":
            estimated_weight_bytes += layers_per_rank * hidden_size * bytes_per_param
        else:
            estimated_weight_bytes += layers_per_rank * 2 * hidden_size * bytes_per_param
        
        # Self attention
        query_projection_size = hidden_size
        kv_projection_size = kv_channels * tf_config.num_query_groups
        qkv_size = query_projection_size + 2 * kv_projection_size
        estimated_weight_bytes += layers_per_rank * hidden_size * (qkv_size // tp_size) * bytes_per_param
        
        # output projection
        estimated_weight_bytes += layers_per_rank * (hidden_size // tp_size) * hidden_size * bytes_per_param
        
        # MLP
        # column parallel
        estimated_weight_bytes += layers_per_rank * hidden_size * (ffn_hidden_size // tp_size) * bytes_per_param
        # row parallel
        estimated_weight_bytes += layers_per_rank * (ffn_hidden_size // tp_size) * hidden_size * bytes_per_param
        
        # Layer norms
        if tf_config.normalization == "RMSNorm":
            estimated_weight_bytes += layers_per_rank * hidden_size * bytes_per_param
        else:
            estimated_weight_bytes += layers_per_rank * 2 * hidden_size * bytes_per_param
        # Final layer norm
        if pp_rank == (pp_size - 1):
            if tf_config.normalization == "RMSNorm":
                estimated_weight_bytes += hidden_size * bytes_per_param
            else:
                estimated_weight_bytes += 2 * hidden_size * bytes_per_param
        
        # Output layer weights (only on last pipeline stage)
        if pp_rank == (pp_size - 1):
            estimated_weight_bytes += hidden_size * (vocab_size // tp_size) * bytes_per_param
        
        detailed_mem_report["estimated_peak_mem_diff"] = get_memory_str(estimated_weight_bytes, human_readable=True)

    @override
    def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
        micro_batch = micro_batch.to(torch.cuda.current_device())
        micro_batch = micro_batch.contiguous()
        
        input_ids_rmpad, attention_mask, position_ids_rmpad, packed_seq_params = (
            get_thd_model_input_from_bshd(micro_batch)
        )
        
        return (
            input_ids_rmpad,
            position_ids_rmpad,
            attention_mask, 
            None,
            None,
            packed_seq_params,
            None,
            None,
        )
    
    @override
    def calculate_tokens(
        self, test_case: InputTestCase, micro_batch: TensorDict, inputs: Any
    ) -> int:
        input_ids = inputs[0]
        
        if test_case.shape == "thd":
            num_tokens = input_ids.size(0)
        elif test_case.shape == "bshd":
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            num_tokens = batch_size * seq_len
            if test_case.sequence_parallel_enabled:
                num_tokens = num_tokens // mpu.get_tensor_model_parallel_world_size()
        else:
            raise ValueError(f"Unsupported shape format: {test_case.shape}")
        
        if test_case.context_parallel_size > 1:
            num_tokens = num_tokens // test_case.context_parallel_size
            
        return num_tokens

    @override
    def calc_theoretical_memory(self, test_case: InputTestCase) -> Dict[str, int]:
        hidden_size = self.hf_config.hidden_size
        micro_batch_size = test_case.micro_batch_size
        seq_len = test_case.seqlen
        dtype = next(self.op.parameters()).dtype
        bytes_per_param = torch.tensor([], dtype=dtype).element_size()
        
        # Base activation memory: 18 * batch_size * seq_len * hidden_size
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
        
        # Embedding (only on first pipeline stage)
        if mpu.get_pipeline_model_parallel_rank() == 0:
            forward_flops += tokens * hidden_size
        
        # Transformer layers FLOPS
        if layers_per_rank > 0:
            # Self-attention FLOPS per layer:
            # QKV projection: 2 * tokens * hidden_size * (hidden_size + 2*kv_channels)
            # Attention computation: 2 * tokens * tokens * hidden_size (approximation)
            # Output projection: 2 * tokens * hidden_size * hidden_size
            qkv_flops = 2 * tokens * hidden_size * (hidden_size + 2 * (hidden_size // self.hf_config.num_attention_heads))
            attn_flops = 2 * tokens * tokens * hidden_size
            proj_flops = 2 * tokens * hidden_size * hidden_size
            # MLP FLOPS per layer:
            # First linear: 2 * tokens * hidden_size * ffn_hidden_size
            # Second linear: 2 * tokens * ffn_hidden_size * hidden_size
            mlp_flops = 2 * tokens * hidden_size * ffn_hidden_size + 2 * tokens * ffn_hidden_size * hidden_size
            
            # Layer norm FLOPS (approx 5 FLOPS per element)
            ln_flops = 5 * tokens * hidden_size
            
            # Total FLOPS per layer
            layer_flops = qkv_flops + attn_flops + proj_flops + mlp_flops + 2 * ln_flops  # 2 layer norms per layer
            
            # Adjust for tensor parallelism
            layer_flops = layer_flops / tp_size
            
            # Total for all layers on this rank
            forward_flops += layers_per_rank * layer_flops
        # Final layer norm FLOPS (only on last pipeline stage)
        if mpu.get_pipeline_model_parallel_rank() == (pp_size - 1):
            forward_flops += 5 * tokens * hidden_size
            
            # Output layer FLOPS
            forward_flops += 2 * tokens * hidden_size * (vocab_size // tp_size)
        
        # Backward FLOPS is approximately 2x forward FLOPS
        backward_flops = 2 * forward_flops
        
        return {"forward": forward_flops, "backward": backward_flops}
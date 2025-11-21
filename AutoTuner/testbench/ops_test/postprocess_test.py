import os
from typing import Any, Dict, Literal, Optional

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from tensordict import TensorDict
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.utils.memory import MemoryTrackerContext, get_memory_str
from AutoTuner.utils.model_inputs import get_thd_model_input_from_bshd
from AutoTuner.utils.structs import InputTestCase

from ..ops.postprocess import PostprocessForTest
from ..profile.configs.config_struct import ProfileMode
from .common import TestCommon

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core import parallel_state
from megatron.core import tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_local_spec,
    get_gpt_mtp_block_spec,
)

from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
    roll_tensor,
    tie_output_layer_state_dict,
    tie_word_embeddings_state_dict,
)

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestPostprocess(TestCommon):
	def __init__(
		self,
		tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        profile_mode: int = 0,
        warmup_iters: int = 2,
        theoretical_flops: bool = False,
        theoretical_activations: bool = False,
        scatter_to_sequence_parallel: bool = True,
        tp_comm_overlap_cfg: str = None,
        share_embeddings_and_output_weights: Optional[bool] = None,
		parallel_output: bool = True,
	):
		super().__init__(
			tf_config=tf_config,
			hf_config=hf_config,
			profile_mode=profile_mode,
			warmup_iters=warmup_iters,
			theoretical_flops=theoretical_flops,
			theoretical_activations=theoretical_activations,
			tp_comm_overlap_cfg=tp_comm_overlap_cfg,
		)
		self.module_name = "Postprocess"
		self.tf_config = tf_config
		self.hf_config = hf_config
		self.profile_mode = profile_mode
		if share_embeddings_and_output_weights is None:
			share_embeddings_and_output_weights = getattr(hf_config, 'tie_word_embeddings', True)
		self.share_embeddings_and_output_weights = share_embeddings_and_output_weights		
		self.vp_stage = parallel_state.get_virtual_pipeline_model_parallel_rank()
		self.post_process = parallel_state.is_pipeline_last_stage()
		self.parallel_output = parallel_output
		self.pre_process = tf_config.mtp_num_layers is not None
		self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
		self.cp_group = parallel_state.get_context_parallel_group()
		self.tp_group = tp_group if tp_group is not None else parallel_state.get_tensor_model_parallel_group()
		
		# Prepare MTP block spec before weight allocation
		
		mtp_block_spec = None
		if tf_config.mtp_num_layers is not None:
			use_te = getattr(tf_config, "transformer_impl", "local") == "transformer_engine"
			if use_te:
				transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
			else:
				transformer_layer_spec = get_gpt_layer_local_spec()
    
			transformer_layer_spec_for_mtp = transformer_layer_spec
			if hasattr(transformer_layer_spec, 'layer_specs') and len(transformer_layer_spec.layer_specs) == 0:
				# derive a concrete transformer layer spec from the decoder block spec
				from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
				decoder_block_spec = get_gpt_decoder_block_spec(config=tf_config, use_transformer_engine=use_te, vp_stage=self.vp_stage)
				transformer_layer_spec_for_mtp = decoder_block_spec.layer_specs[-1]

			mtp_block_spec = get_gpt_mtp_block_spec(
				config=tf_config,
				spec=transformer_layer_spec_for_mtp,
				use_transformer_engine=use_te,
				vp_stage=self.vp_stage,
			)
		self.mtp_block_spec = mtp_block_spec
		self.mtp_process = mtp_block_spec is not None
		
		# Prepare position embedding config
		self.position_embedding_type = getattr(tf_config, 'position_embedding_type', 'rope')
		self.scatter_to_sequence_parallel = scatter_to_sequence_parallel
  
		if self.position_embedding_type == 'rope':
			rotary_percent = getattr(tf_config, 'rotary_percent', 1.0)
			rotary_base = getattr(tf_config, 'rotary_base', 10000)
			seq_len_interpolation_factor = getattr(tf_config, 'seq_len_interpolation_factor', None)
			rope_scaling = getattr(tf_config, 'rope_scaling', False)
			rope_scaling_factor = getattr(tf_config, 'rope_scaling_factor', 8.0)
				
			self.rotary_pos_emb = RotaryEmbedding(
				kv_channels=tf_config.kv_channels,
				rotary_percent=rotary_percent,
				rotary_interleaved=tf_config.rotary_interleaved,
				seq_len_interpolation_factor=seq_len_interpolation_factor,
				rotary_base=rotary_base,
				rope_scaling=rope_scaling,
				rope_scaling_factor=rope_scaling_factor,
				use_cpu_initialization=tf_config.use_cpu_initialization,
				cp_group=self.pg_collection.cp,
			)
		else:
			self.rotary_pos_emb = None
   
		if self.share_embeddings_and_output_weights:
			# Create output layer
			self.output_layer = tensor_parallel.ColumnParallelLinear(
				self.tf_config.hidden_size,
				getattr(self.hf_config, "vocab_size", 151936),
				config=self.tf_config,
				init_method=self.tf_config.init_method,
				bias=False,
				skip_bias_add=False,
				gather_output = not self.parallel_output,
				tp_group = parallel_state.get_tensor_model_parallel_group()
			)
		
		if self.share_embeddings_and_output_weights:
			# Create embedding
			self.embedding = None
			if self.pre_process or tf_config.mtp_num_layers is not None:
				self.embedding = LanguageModelEmbedding(
					config=tf_config,
					vocab_size=hf_config.vocab_size,
					max_sequence_length=hf_config.max_position_embeddings,
					position_embedding_type="rope",
					num_tokentypes=0,
					scatter_to_sequence_parallel=self.scatter_to_sequence_parallel,
					tp_group=self.tp_group,
				)
		

		if profile_mode == ProfileMode.collect_data:
			# Allocate all weights inside MemoryTrackerContext to measure from zero baseline
			with MemoryTrackerContext(self.module_name) as memory_tracker_ctx:
				if not self.share_embeddings_and_output_weights:
					# Create output layer
					self.output_layer = tensor_parallel.ColumnParallelLinear(
						self.tf_config.hidden_size,
						getattr(self.hf_config, "vocab_size", 151936),
						config=self.tf_config,
						init_method=self.tf_config.init_method,
						bias=False,
						skip_bias_add=False,
						gather_output = not self.parallel_output,
						skip_weight_param_allocation = self.pre_process and self.share_embeddings_and_output_weights,
						tp_group = parallel_state.get_tensor_model_parallel_group()
					)
				
				if not self.share_embeddings_and_output_weights:
					# Create embedding
					self.embedding = None
					if self.pre_process or tf_config.mtp_num_layers is not None:
						self.embedding = LanguageModelEmbedding(
							config=tf_config,
							vocab_size=hf_config.vocab_size,
							max_sequence_length=hf_config.max_position_embeddings,
							position_embedding_type="rope",
							num_tokentypes=0,
							scatter_to_sequence_parallel=self.scatter_to_sequence_parallel,
							tp_group=self.tp_group,
						)
      
				# Create the operator wrapper
				self.op = PostprocessForTest(
					tf_config=self.tf_config,
					share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
					mtp=MultiTokenPredictionBlock(
						config=self.tf_config, spec=self.mtp_block_spec, vp_stage=self.vp_stage
					),
					post_process=self.post_process,
					mtp_process=self.pre_process,
					output_layer=self.output_layer,
					cp_group=self.cp_group,
					pg_collection=self.pg_collection,
					embedding=self.embedding,
					hook_activation=(profile_mode == ProfileMode.collect_data),
				)

			detailed_mem_report = memory_tracker_ctx.get_result()
			
			# 计算 postprocess 层的权重内存占用
			vocab_size = hf_config.vocab_size
			hidden_size = tf_config.hidden_size
			ffn_hidden_size = tf_config.ffn_hidden_size
			tp_size = parallel_state.get_tensor_model_parallel_world_size()
			pp_rank = parallel_state.get_pipeline_model_parallel_rank()
			mtp_num_layers = tf_config.mtp_num_layers
			
			dtype = tf_config.params_dtype
			bytes_per_param = torch.finfo(dtype).bits // 8
			
			estimated_weight_mem_bytes = 0
			if not self.share_embeddings_and_output_weights:
				if self.output_layer.weight is not None:
					output_layer_weight = (vocab_size // tp_size) * hidden_size * bytes_per_param
					estimated_weight_mem_bytes += output_layer_weight
			
			if self.mtp_process and mtp_num_layers is not None and mtp_num_layers > 0:
				# Determine how many MTP layers are actually built from the spec.
				if self.mtp_block_spec is not None and hasattr(self.mtp_block_spec, 'layer_specs'):
					built_mtp_layers = len(self.mtp_block_spec.layer_specs)
				else:
					built_mtp_layers = mtp_num_layers

				# Tensor-parallel world size (fallback to 1)
				world_tp = tp_size if tp_size and tp_size > 0 else 1
				# local shard sizes used by Column/Row parallel layers
				hidden_shard = hidden_size // world_tp
				ffn_shard = ffn_hidden_size // world_tp

				# Sum params by walking the logical structure used to build MTP layers.
				mtp_block_params = 0
				for _ in range(built_mtp_layers):
					# MTP layer has: enorm (LN), hnorm (LN), eh_proj (2H->H column-parallel),
					# transformer_layer (input LN, self-attn qkv, attn out proj, pre-mlp LN, mlp fc1/fc2), final_layernorm
					# Count LayerNorm params: weight + bias
					layernorm_params = 3 * 2 * hidden_size

					# eh_proj: ColumnParallelLinear allocates weight of shape (output_size_per_partition, input_size)
					# ColumnParallelLinear divides output dimension across TP ranks. For 2H->H, output_size=H,
					# local output partition = H_shard => local weight params = H_shard * (2H)
					eh_proj_params = hidden_shard * (2 * hidden_size)

					# Transformer layer breakdown (follow TransformerLayer construction):
					# input layernorm (weight+bias)
					input_ln_params = 2 * hidden_size
					# QKV: column-parallel linear with output 3H -> local params = 3*H*H_shard
					qkv_params = 3 * hidden_size * hidden_shard
					# attention output projection: row-parallel linear (local weight shape = H_shard x H)
					attn_out_params = hidden_shard * hidden_size
					# pre-MLP layernorm
					pre_mlp_ln_params = 2 * hidden_size
					# MLP: fc1 local params = H * (FFN_shard), fc2 local params = FFN_shard * H
					mlp_fc1_params = hidden_size * ffn_shard
					mlp_fc2_params = ffn_shard * hidden_size

					transformer_layer_params = (
						input_ln_params
						+ qkv_params
						+ attn_out_params
						+ pre_mlp_ln_params
						+ mlp_fc1_params
						+ mlp_fc2_params
					)

					mtp_layer_params = layernorm_params + eh_proj_params + transformer_layer_params
					mtp_block_params += mtp_layer_params

				# Convert to bytes
				mtp_block_weight_bytes = mtp_block_params * bytes_per_param

				# Conservative FP8 overhead: 1 byte per param plus potential small extra tables
				fp8_overhead_bytes = 0
				if getattr(self.tf_config, 'fp8', False):
					fp8_overhead_bytes = mtp_block_params * 1

				estimated_weight_mem_bytes += mtp_block_weight_bytes + fp8_overhead_bytes
			
			if self.mtp_process and pp_rank != 0 and self.embedding is not None:
				if self.embedding.word_embeddings.weight is not None:
					embedding_replica_weight = (vocab_size // tp_size) * hidden_size * bytes_per_param
					estimated_weight_mem_bytes += embedding_replica_weight

			# 4. Rotary positional embedding (RoPE) inv_freq tensor
			if self.position_embedding_type == 'rope' and getattr(self, 'rotary_pos_emb', None) is not None:
				# inv_freq length = number of indices from 0..dim-1 step 2 -> ceil(dim/2)
				dim = getattr(self.tf_config, 'kv_channels', None)
				if dim is not None:
					rotary_percent = getattr(self.tf_config, 'rotary_percent', 1.0)
					if rotary_percent < 1.0:
						dim = int(dim * rotary_percent)
					inv_freq_len = (dim + 1) // 2
					rope_mem = inv_freq_len * bytes_per_param
					estimated_weight_mem_bytes += rope_mem
			
			estimated_weight_mem_str = get_memory_str(
				estimated_weight_mem_bytes, human_readable=True
			)
			# detailed_mem_report["estimated_weight_memory"] = mtp_block_weight
			detailed_mem_report["estimate_peak_mem_diff"] = estimated_weight_mem_str
			self.memory_db["weights"][self.module_name] = detailed_mem_report
		else:
			# Non-profile mode: allocate weights in original order
			self.output_layer = tensor_parallel.ColumnParallelLinear(
				self.tf_config.hidden_size,
				getattr(self.hf_config, "vocab_size", 151936),
				config=self.tf_config,
				init_method=self.tf_config.init_method,
				bias=False,
				skip_bias_add=False,
				gather_output = not self.parallel_output,
				skip_weight_param_allocation = self.pre_process and self.share_embeddings_and_output_weights,
				tp_group = parallel_state.get_tensor_model_parallel_group()
			)
			
			self.embedding = None
			if self.pre_process or tf_config.mtp_num_layers is not None:
				self.embedding = LanguageModelEmbedding(
					config=tf_config,
					vocab_size=hf_config.vocab_size,
					max_sequence_length=hf_config.max_position_embeddings,
					position_embedding_type="rope",
					num_tokentypes=0,
					scatter_to_sequence_parallel=self.scatter_to_sequence_parallel,
					tp_group=self.tp_group,
				)
			
			self.mtp = None
			if self.mtp_block_spec is not None:
				self.mtp = MultiTokenPredictionBlock(
					config=self.tf_config, spec=self.mtp_block_spec, vp_stage=self.vp_stage
				)
			
			self.op = PostprocessForTest(
					tf_config=self.tf_config,
					share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
					mtp=self.mtp,
					post_process=self.post_process,
					mtp_process=self.pre_process,
					output_layer=self.output_layer,
					cp_group=self.cp_group,
					pg_collection=self.pg_collection,
					embedding=self.embedding,
					hook_activation=(profile_mode == ProfileMode.collect_data),
				)

	@override
	def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
		micro_batch = micro_batch.to(torch.cuda.current_device())
		micro_batch = micro_batch.contiguous()
  
		# 禁用 sequence packing，使用标准的 BSHD 格式以支持 MTP
		# MTP + sequence packing 目前不被 Megatron 支持
		input_ids = micro_batch["input_ids"]
		attention_mask = micro_batch["attention_mask"]
		position_ids = micro_batch["position_ids"]
		packed_seq_params = None  # 禁用 sequence packing
  
		if self.pre_process or self.mtp_process:
			with torch.no_grad():
				# embedding 返回 [seq_len, batch_size, hidden_size] (SBH)
				decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
		else:
			# 当没有 embedding 时，生成模拟的 decoder hidden states
			# Shape: [seq_len, batch_size, hidden_size] (SBH) for MTP compatibility
			batch_size, seq_len = input_ids.shape
			hidden_size = self.tf_config.hidden_size
			decoder_input = torch.randn(
				seq_len, batch_size, hidden_size,
				dtype=self.tf_config.params_dtype,
				device=torch.cuda.current_device()
			)
		
		# 模拟 decoder 处理后的 hidden states (保持 SBH 格式)
		hidden_states = decoder_input.clone()
		
		# 从原始 micro_batch 获取 labels（保持 BSHD 格式）
		if "labels" in micro_batch:
			labels = micro_batch["labels"]
		else:
			labels = micro_batch.get("input_ids")
		
		rotary_pos_emb = None
		# 获取 RoPE embeddings（参考 gpt_model.py _preprocess 方法）
		if self.position_embedding_type == 'rope' and self.rotary_pos_emb is not None:
			rotary_seq_len = position_ids.shape[1] if position_ids.dim() > 1 else position_ids.shape[0]
			rotary_pos_emb = self.rotary_pos_emb(
				rotary_seq_len,
				packed_seq=False  # 不使用 sequence packing
			)
		
		# 当启用 MTP 时，根据情况设置 mtp_in_postprocess
		# mtp_in_postprocess=True 表示在 postprocess 中运行 MTP block
		# mtp_in_postprocess=False 表示 MTP 已在之前运行，现在只处理 MTP 的输出
		mtp_in_postprocess = self.mtp_process  # 如果启用了 MTP，则设置为 True
		
		loss_mask = torch.ones_like(labels, dtype=self.tf_config.params_dtype)
		extra_block_kwargs = None
		
		return (
			hidden_states,           
			input_ids,         
			position_ids,            
			labels,                
			rotary_pos_emb,     
			mtp_in_postprocess,      
			loss_mask,               
			attention_mask,        
			packed_seq_params, 
			extra_block_kwargs,      
		)

	@override
	def calculate_tokens(self, test_case: InputTestCase, micro_batch: TensorDict, inputs: Any) -> int:
		
		attention_mask = micro_batch["attention_mask"]
		return attention_mask.sum().item()

	@override
	def calc_theoretical_flops(self, test_case: InputTestCase) -> Dict[str, float]:
		micro_batch_size = test_case.micro_batch_size
		seq_len = test_case.seqlen
		hidden_size = self.tf_config.hidden_size
		vocab_size = self.hf_config.vocab_size
		tp_size = test_case.tensor_model_parallel_size
		cp_size = test_case.context_parallel_size
		local_seq_len = seq_len // cp_size
		total_tokens = micro_batch_size * local_seq_len
		
		forward_flops = 0.0
		
		# 1. output_layer
		if self.post_process and not self.share_embeddings_and_output_weights:
			output_layer_flops = 2 * total_tokens * hidden_size * (vocab_size // tp_size)
			forward_flops += output_layer_flops
		
		# 2. MTP
		if self.mtp_process and self.tf_config.mtp_num_layers is not None and self.tf_config.mtp_num_layers > 0:
			mtp_num_layers = self.tf_config.mtp_num_layers
			ffn_hidden_size = self.tf_config.ffn_hidden_size
			num_attention_heads = self.tf_config.num_attention_heads
			kv_channels = self.tf_config.kv_channels
			
			for _ in range(mtp_num_layers):
				eh_proj_flops = 2 * total_tokens * hidden_size * (2 * hidden_size // tp_size)
				qkv_flops = 2 * total_tokens * hidden_size * (3 * hidden_size // tp_size)
				attention_flops = 2 * micro_batch_size * num_attention_heads * local_seq_len * local_seq_len * kv_channels
				attention_flops += 2 * micro_batch_size * num_attention_heads * local_seq_len * local_seq_len * kv_channels
				attn_out_flops = 2 * total_tokens * (hidden_size // tp_size) * hidden_size
				mlp_fc1_flops = 2 * total_tokens * hidden_size * (ffn_hidden_size // tp_size)
				mlp_fc2_flops = 2 * total_tokens * (ffn_hidden_size // tp_size) * hidden_size
				
				layer_flops = eh_proj_flops + qkv_flops + attention_flops + attn_out_flops + mlp_fc1_flops + mlp_fc2_flops
				forward_flops += layer_flops
		
		# 3. Cross entropy loss
		if self.post_process:
			if self.parallel_output:
				local_vocab_for_compute = vocab_size // tp_size
			else:
				local_vocab_for_compute = vocab_size
			max_flops = total_tokens * (local_vocab_for_compute - 1)
			sub_max_flops = total_tokens * local_vocab_for_compute
			exp_flops = total_tokens * local_vocab_for_compute
			sum_exp_flops = total_tokens * (local_vocab_for_compute - 1)
			log_flops = total_tokens 
			nll_flops = total_tokens * 3
			loss_flops = (max_flops + sub_max_flops + exp_flops + 
			              sum_exp_flops + log_flops + nll_flops)
			
			forward_flops += loss_flops
		
		backward_flops = 2 * forward_flops
		
		return {"forward": forward_flops, "backward": backward_flops}

	@override
	def calc_theoretical_memory(self, test_case: InputTestCase) -> Dict[str, int]:
		"""
  		Calculate theoretical activation memory for postprocess operations.
		"""
		seq_len = test_case.seqlen
		micro_batch_size = test_case.micro_batch_size
		vocab_size = self.hf_config.vocab_size
		hidden_size = self.tf_config.hidden_size
		tp_size = test_case.tensor_model_parallel_size
		cp_size = test_case.context_parallel_size
		dtype = self.tf_config.params_dtype
		bytes_per_elem = torch.finfo(dtype).bits // 8
		local_seq_len = seq_len // cp_size
		total_tokens = micro_batch_size * local_seq_len
		
		total_activation_mem = 0
		
		if self.post_process:
			if self.parallel_output:
				logits_mem = total_tokens * (vocab_size // tp_size) * bytes_per_elem
			else:
				logits_mem = total_tokens * vocab_size * bytes_per_elem
			total_activation_mem += logits_mem
		
		if self.mtp_process and self.tf_config.mtp_num_layers is not None and self.tf_config.mtp_num_layers > 0:
			mtp_num_layers = self.tf_config.mtp_num_layers
			ffn_hidden_size = self.tf_config.ffn_hidden_size
			num_attention_heads = self.tf_config.num_attention_heads
			
			for _ in range(mtp_num_layers):
				eh_proj_mem = total_tokens * (2 * hidden_size // tp_size) * bytes_per_elem
				qkv_mem = total_tokens * (3 * hidden_size // tp_size) * bytes_per_elem
				attn_scores_mem = micro_batch_size * (num_attention_heads // tp_size) * local_seq_len * local_seq_len * bytes_per_elem
				attn_out_mem = total_tokens * (hidden_size // tp_size) * bytes_per_elem
				mlp_fc1_mem = total_tokens * (ffn_hidden_size // tp_size) * bytes_per_elem
				mlp_fc2_mem = total_tokens * hidden_size * bytes_per_elem
				
				# Layer total (peak memory, not all stored simultaneously)
				layer_mem = max(
					eh_proj_mem + qkv_mem, 
					attn_scores_mem + attn_out_mem,
					mlp_fc1_mem + mlp_fc2_mem 
				)
				total_activation_mem += layer_mem
		
		# 3. Cross entropy intermediate tensors(maybe don't need to add)
		if self.post_process:
			if self.parallel_output:
				local_vocab = vocab_size // tp_size
			else:
				local_vocab = vocab_size
			
			exp_logits_mem = total_tokens * local_vocab * bytes_per_elem
			sum_exp_mem = total_tokens * bytes_per_elem
			loss_mem = total_tokens * bytes_per_elem
			ce_mem = exp_logits_mem + sum_exp_mem + loss_mem
			total_activation_mem += ce_mem
		
		return {"activations": {"activations": total_activation_mem}}
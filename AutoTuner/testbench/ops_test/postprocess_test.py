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
from AutoTuner.utils.hidden_status_gen import HiddenStatusGenerator

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
		if tf_config.mtp_num_layers is not None and tf_config.mtp_num_layers > 0:
			use_te = getattr(tf_config, "transformer_impl", "local") == "transformer_engine"
			if use_te:
				transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
			else:
				transformer_layer_spec = get_gpt_layer_local_spec()

			transformer_layer_spec_for_mtp = transformer_layer_spec

			mtp_block_spec = get_gpt_mtp_block_spec(
				config=tf_config,
				spec=transformer_layer_spec_for_mtp,
				use_transformer_engine=use_te,
				vp_stage=self.vp_stage,
			)
		self.mtp_block_spec = mtp_block_spec
		self.mtp_process = mtp_block_spec is not None

		self.scatter_to_sequence_parallel = scatter_to_sequence_parallel

		if profile_mode == ProfileMode.collect_data:
			# Allocate all weights inside MemoryTrackerContext to measure from zero baseline
			with MemoryTrackerContext(self.module_name) as memory_tracker_ctx:
				# Create the operator wrapper
				self.op = PostprocessForTest(
					tf_config=self.tf_config,
					share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
					mtp=(
						MultiTokenPredictionBlock(
							config=self.tf_config,
							spec=self.mtp_block_spec,
							vp_stage=self.vp_stage,
						) if self.mtp_process else None
					),
					post_process=self.post_process,
					mtp_process=self.mtp_process,
					output_layer= tensor_parallel.ColumnParallelLinear(
						self.tf_config.hidden_size,
						getattr(self.hf_config, "vocab_size", 151936),
						config=self.tf_config,
						init_method=self.tf_config.init_method,
						bias=False,
						skip_bias_add=False,
						gather_output = not self.parallel_output,
						tp_group = parallel_state.get_tensor_model_parallel_group()
					),
					cp_group=self.cp_group,
					pg_collection=self.pg_collection,
					embedding=LanguageModelEmbedding(
						config=tf_config,
						vocab_size=hf_config.vocab_size,
						max_sequence_length=hf_config.max_position_embeddings,
						position_embedding_type="rope",
						num_tokentypes=0,
						scatter_to_sequence_parallel=self.scatter_to_sequence_parallel,
						tp_group=self.tp_group,
					),
					hook_activation=(profile_mode == ProfileMode.collect_data),
				)

			detailed_mem_report = memory_tracker_ctx.get_result()
			# Estimate weight memory
			vocab_size = hf_config.vocab_size
			hidden_size = tf_config.hidden_size
			ffn_hidden_size = tf_config.ffn_hidden_size
			tp_size = parallel_state.get_tensor_model_parallel_world_size()
			pp_rank = parallel_state.get_pipeline_model_parallel_rank()
			mtp_num_layers = tf_config.mtp_num_layers
			
			dtype = tf_config.params_dtype
			bytes_per_param = torch.finfo(dtype).bits // 8
			
			estimated_weight_mem_bytes = 0
			
			# 1.Output layer weights
			estimated_weight_mem_bytes += (vocab_size // tp_size) * hidden_size * bytes_per_param

			# 2. MTP weights
			if self.mtp_process and mtp_num_layers is not None and mtp_num_layers > 0:
				built_mtp_layers = len(self.mtp_block_spec.layer_specs)

				world_tp = tp_size if tp_size and tp_size > 0 else 1
				hidden_shard = hidden_size // world_tp
				ffn_shard = ffn_hidden_size // world_tp
				per_layer_params = 0
				# 1) RMSNorm: enorm, hnorm, final_layernorm
				per_layer_params += 3 * hidden_size
				
				# 2) eh_proj
				per_layer_params += 2 * hidden_size * hidden_shard
				
				# 3) TransformerLayer
				#(a) linear_qkv
				I_qkv = hidden_size
				O_qkv = 4 * hidden_shard
				per_layer_params += I_qkv * O_qkv
				
				# linear_proj
				I_proj = 2 * hidden_shard
				O_proj = hidden_size
				per_layer_params += O_proj * I_proj
				
				# MLP
				I_fc1 = hidden_size
				O_fc1 = ffn_shard * 2
				per_layer_params += I_fc1 * O_fc1
				I_fc2 = ffn_shard
				O_fc2 = hidden_size
				per_layer_params += O_fc2 * I_fc2
				total_mtp_params = per_layer_params * built_mtp_layers
				mtp_block_weight_bytes = total_mtp_params * bytes_per_param
				estimated_weight_mem_bytes += mtp_block_weight_bytes
			
			# 3. embedding
			embedding_weight = self.op.embedding.word_embeddings.weight
			bytes_per_param = torch.finfo(embedding_weight.dtype).bits // 8
			embedding_replica_weight = (vocab_size // tp_size) * hidden_size * bytes_per_param
			estimated_weight_mem_bytes += embedding_replica_weight
			
			estimated_weight_mem_str = get_memory_str(
				round(estimated_weight_mem_bytes), human_readable=True
			)
			# detailed_mem_report["estimated_weight_memory"] = mtp_block_weight
			detailed_mem_report["estimate_peak_mem_diff"] = estimated_weight_mem_str
			self.memory_db["weights"][self.module_name] = detailed_mem_report
		else:
			self.op = PostprocessForTest(
					tf_config=self.tf_config,
					share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
					mtp=MultiTokenPredictionBlock(
						config=self.tf_config, spec=self.mtp_block_spec, vp_stage=self.vp_stage
					) if self.mtp_process else None,
					post_process=self.post_process,
					mtp_process=self.mtp_process,
					output_layer= tensor_parallel.ColumnParallelLinear(
						self.tf_config.hidden_size,
						getattr(self.hf_config, "vocab_size", 151936),
						config=self.tf_config,
						init_method=self.tf_config.init_method,
						bias=False,
						skip_bias_add=False,
						gather_output = not self.parallel_output,
						tp_group = parallel_state.get_tensor_model_parallel_group()
					),
					cp_group=self.cp_group,
					pg_collection=self.pg_collection,
					embedding=LanguageModelEmbedding(
						config=self.tf_config,
						vocab_size=self.hf_config.vocab_size,
						max_sequence_length=self.hf_config.max_position_embeddings,
						position_embedding_type="rope",
						num_tokentypes=0,
						scatter_to_sequence_parallel=self.scatter_to_sequence_parallel,
						tp_group=self.tp_group,
					),
					hook_activation=(profile_mode == ProfileMode.collect_data),
				)


	@override
	def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):

		micro_batch = micro_batch.to(torch.cuda.current_device())
		micro_batch = micro_batch.contiguous()

		self.hiddenstatus_generator = HiddenStatusGenerator(
			tf_config=self.tf_config,
			hf_config=self.hf_config,
			tp_group=self.tp_group,
			scatter_to_sequence_parallel=self.scatter_to_sequence_parallel,
			rotary_percent=1.0,
			rotary_base=10000,
			rope_scaling= False,
			rope_scaling_factor= 8.0,
			seq_len_interpolation_factor= None,
			pg_collection=self.pg_collection,
		)

		input_ids = micro_batch["input_ids"]
		attention_mask = micro_batch["attention_mask"]
		position_ids = micro_batch["position_ids"]
		packed_seq_params = None  # Disable sequence packing
		self.embedding = LanguageModelEmbedding(
			config=self.tf_config,
			vocab_size=self.hf_config.vocab_size,
			max_sequence_length=self.hf_config.max_position_embeddings,
			position_embedding_type="rope",
			num_tokentypes=0,
			scatter_to_sequence_parallel=self.scatter_to_sequence_parallel,
			tp_group=self.tp_group,
			)
		if self.pre_process or self.mtp_process:
			with torch.no_grad():
				decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
		else:
			hidden_size = self.tf_config.hidden_size
			total_tokens = input_ids.shape[0] if input_ids.dim() == 1 else input_ids.numel()
			decoder_input = torch.randn(
				total_tokens, hidden_size,
				dtype=self.tf_config.params_dtype,
				device=torch.cuda.current_device()
			)
		
		hidden_states = decoder_input.clone()

		
		if "labels" in micro_batch:
			labels = micro_batch["labels"]
		else:
			labels = micro_batch.get("input_ids")
   
		rotary_percent = getattr(self.tf_config, 'rotary_percent', 1.0)
		rotary_base = getattr(self.tf_config, 'rotary_base', 10000)
		seq_len_interpolation_factor = getattr(self.tf_config, 'seq_len_interpolation_factor', None)
		rope_scaling = getattr(self.tf_config, 'rope_scaling', False)
		rope_scaling_factor = getattr(self.tf_config, 'rope_scaling_factor', 8.0)
			
		self.rotary_pos_emb = RotaryEmbedding(
			kv_channels=self.tf_config.kv_channels,
			rotary_percent=rotary_percent,
			rotary_interleaved=self.tf_config.rotary_interleaved,
			seq_len_interpolation_factor=seq_len_interpolation_factor,
			rotary_base=rotary_base,
			rope_scaling=rope_scaling,
			rope_scaling_factor=rope_scaling_factor,
			use_cpu_initialization=self.tf_config.use_cpu_initialization,
			cp_group=self.pg_collection.cp,
		)
		
		rotary_pos_emb = None
		rotary_seq_len = position_ids.shape[1] if position_ids.dim() > 1 else position_ids.shape[0]
		rotary_pos_emb = self.rotary_pos_emb(
			rotary_seq_len,
			packed_seq=False
		)
			
		
		mtp_in_postprocess = self.mtp_process
		
		extra_block_kwargs = None
		
		return (
			hidden_states,
			input_ids,
			position_ids,
			labels,
			rotary_pos_emb,
			mtp_in_postprocess,
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
		
		# 2. MTP FLOPS - Calculate forward multiply-add operations for each layer in execution order
		if self.mtp_process and self.tf_config.mtp_num_layers is not None and self.tf_config.mtp_num_layers > 0:
			mtp_num_layers = self.tf_config.mtp_num_layers
			ffn_hidden_size = self.tf_config.ffn_hidden_size
			num_attention_heads = self.tf_config.num_attention_heads
			kv_channels = self.tf_config.kv_channels
			# local_seq_len depends on cp splitting already computed
			for _ in range(mtp_num_layers):
				# RMSNorm
				enorm_flops = 3 * hidden_size * total_tokens
				hnorm_flops = 3 * (hidden_size // tp_size) * total_tokens
				final_norm_flops = 3 * (hidden_size // tp_size) * total_tokens
				# eh_proj
				eh_proj_flops = 2 * (2 * hidden_size) * (hidden_size // tp_size) * total_tokens
				# QKV projection
				qkv_flops = 2 * total_tokens * (hidden_size // tp_size) * (4 * hidden_size // tp_size)
				# attention (scaled dot-product + softmax + matmul) -- approximate double multiply cost
				attention_flops = 2 * micro_batch_size * num_attention_heads * local_seq_len * local_seq_len * kv_channels
				attention_flops += 2 * micro_batch_size * num_attention_heads * local_seq_len * local_seq_len * kv_channels
				# output projection
				attn_out_flops = 2 * total_tokens * (hidden_size // tp_size) * (2 * hidden_size // tp_size)
				# MLP
				mlp_fc1_flops = 2 * total_tokens * (hidden_size // tp_size) * (2 * ffn_hidden_size // tp_size)
				mlp_fc2_flops = 2 * total_tokens * (ffn_hidden_size // tp_size) * (hidden_size // tp_size)
				layer_flops = eh_proj_flops + qkv_flops + attention_flops + attn_out_flops + mlp_fc1_flops + mlp_fc2_flops + enorm_flops+hnorm_flops+final_norm_flops
				forward_flops += layer_flops
		
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
		
		# Logits memory
		if self.parallel_output:
			logits_mem = total_tokens * (vocab_size // tp_size) * bytes_per_elem
		else:
			logits_mem = total_tokens * vocab_size * bytes_per_elem
		total_activation_mem += logits_mem

		# MTP activation memory
		if self.mtp_process and self.tf_config.mtp_num_layers is not None and self.tf_config.mtp_num_layers > 0:
			mtp_num_layers = self.tf_config.mtp_num_layers
			ffn_hidden_size = self.tf_config.ffn_hidden_size
			num_attention_heads = self.tf_config.num_attention_heads

			world_tp = tp_size if tp_size and tp_size > 0 else 1
			hidden_shard = hidden_size // world_tp
			ffn_shard = ffn_hidden_size // world_tp 
			num_heads_per_partition = num_attention_heads // world_tp

			# proj
			ehnorm_phase = total_tokens * hidden_size * bytes_per_elem*2
			eh_proj_phase = total_tokens * 2 * hidden_size * bytes_per_elem
			hnorm_phase = total_tokens * hidden_shard * bytes_per_elem
			qkv_phase = total_tokens * (hidden_shard + 4 * hidden_shard) * bytes_per_elem
			proj_phase = ehnorm_phase + eh_proj_phase + hnorm_phase + qkv_phase
			# attention
			attn_scores = micro_batch_size * num_heads_per_partition * local_seq_len * local_seq_len * bytes_per_elem
			attn_output = total_tokens * hidden_shard * bytes_per_elem
			linear_proj_out = total_tokens * hidden_shard * bytes_per_elem
			attn_phase = attn_scores + attn_output + linear_proj_out
			# MLP
			mlp_fc1_phase = total_tokens * 2 * ffn_shard * bytes_per_elem
			mlp_fc2_phase = total_tokens * hidden_shard * bytes_per_elem
			# final norm
			final_norm_phase = total_tokens * hidden_shard * bytes_per_elem
			mlp_attn_phase = mlp_fc1_phase + mlp_fc2_phase + final_norm_phase

			layer_mem = proj_phase + attn_phase + mlp_attn_phase
			total_activation_mem += layer_mem * mtp_num_layers
		
		return {"activations": {"activations": total_activation_mem}}

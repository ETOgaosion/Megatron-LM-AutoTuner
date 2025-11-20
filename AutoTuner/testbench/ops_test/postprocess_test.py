import os
from typing import Dict, Literal, Optional

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
		self.tp_group = parallel_state.get_tensor_model_parallel_group()
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
		mtp_block_spec = None
		if tf_config.mtp_num_layers is not None :
			# 1. 先构造 transformer_layer_spec
			use_te = getattr(tf_config, "transformer_impl", "local") == "transformer_engine"
			if use_te:
				transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
			else:
				transformer_layer_spec = get_gpt_layer_local_spec()
			
			# 2. 用 transformer_layer_spec 构造 mtp_block_spec
			mtp_block_spec = get_gpt_mtp_block_spec(
				config=tf_config,
				spec=transformer_layer_spec,
				use_transformer_engine=use_te,
				vp_stage=self.vp_stage,
			)
		self.mtp_block_spec = mtp_block_spec
		self.embedding = None
		if self.pre_process or tf_config.mtp_num_layers is not None:
			self.embedding = LanguageModelEmbedding(
				config=tf_config,
                vocab_size=hf_config.vocab_size,
                max_sequence_length=hf_config.max_position_embeddings,
                position_embedding_type="rope",
                num_tokentypes=0,
                scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                tp_group=self.tp_group,
			)
		self.mtp_process = mtp_block_spec is not None
		self.mtp = MultiTokenPredictionBlock(
                config=self.tf_config, spec=self.mtp_block_spec, vp_stage=self.vp_stage
            )
		
		# 初始化 RoPE（参考 gpt_model.py）
		self.position_embedding_type = getattr(tf_config, 'position_embedding_type', 'rope')
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

		if profile_mode == ProfileMode.collect_data:
			with MemoryTrackerContext(self.module_name) as memory_tracker_ctx:
				self.op = PostprocessForTest(
					tf_config=self.tf_config,
					share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
					mtp=self.mtp,
					post_process=self.post_process,
					mtp_process=self.pre_process,
					output_layer=self.output_layer,
					cp_group=self.cp_group,
					pg_collectiosn=self.pg_collection,
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
			if self.post_process and not self.share_embeddings_and_output_weights:
				if self.output_layer.weight is not None:
					output_layer_weight = (vocab_size // tp_size) * hidden_size * bytes_per_param
					estimated_weight_mem_bytes += output_layer_weight
			
			if self.mtp_process and mtp_num_layers > 0:
				mtp_layernorm_per_layer = 3 * 2 * hidden_size * bytes_per_param
				mtp_eh_proj_per_layer = (2 * hidden_size) * (hidden_size // tp_size) * bytes_per_param
				transformer_input_ln = 2 * hidden_size * bytes_per_param
				transformer_qkv = 3 * hidden_size * (hidden_size // tp_size) * bytes_per_param
				transformer_attn_proj = (hidden_size // tp_size) * hidden_size * bytes_per_param
				transformer_attention = transformer_qkv + transformer_attn_proj
				transformer_pre_mlp_ln = 2 * hidden_size * bytes_per_param
				transformer_mlp_fc1 = hidden_size * (ffn_hidden_size // tp_size) * bytes_per_param
				transformer_mlp_fc2 = (ffn_hidden_size // tp_size) * hidden_size * bytes_per_param
				transformer_mlp = transformer_mlp_fc1 + transformer_mlp_fc2
    
				transformer_layer_weight = (transformer_input_ln + transformer_attention + transformer_pre_mlp_ln + transformer_mlp)
				mtp_layer_weight = (mtp_layernorm_per_layer + mtp_eh_proj_per_layer + transformer_layer_weight)
    
				mtp_block_weight = mtp_layer_weight * mtp_num_layers
				estimated_weight_mem_bytes += mtp_block_weight
			
			if self.mtp_process and pp_rank != 0 and self.embedding is not None:
				if self.embedding.word_embeddings.weight is not None:
					embedding_replica_weight = (vocab_size // tp_size) * hidden_size * bytes_per_param
					estimated_weight_mem_bytes += embedding_replica_weight
			
			estimated_weight_mem_str = get_memory_str(
				estimated_weight_mem_bytes, human_readable=True
			)
			detailed_mem_report["postprocess_peak_mem_diff"] = estimated_weight_mem_str
			self.memory_db["weights"][self.module_name] = detailed_mem_report
		else:
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
		"""
		Prepare inputs for postprocessing, mimicking the output from decoder.
		准备后处理所需的输入，模拟从 decoder 输出的数据。
		
		参考 gpt_model.py 中的完整 forward 流程：
		1. _preprocess: 使用 embedding 获取 decoder_input
		2. decoder: 处理得到 hidden_states
		3. _postprocess: 使用 hidden_states 生成 logits/loss
		"""
		micro_batch = micro_batch.to(torch.cuda.current_device())
		micro_batch = micro_batch.contiguous()
  
		input_ids, attention_mask, position_ids, packed_seq_params = (
            get_thd_model_input_from_bshd(micro_batch)
        )
  
		if self.pre_process or self.mtp_process:
			with torch.no_grad():
				decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
		
		hidden_states = decoder_input.clone()  # 模拟 decoder 处理
		
		# 从原始 micro_batch 获取 labels（BSHD 格式）
		if "labels" in micro_batch:
			labels = micro_batch["labels"]
		else:
			labels = micro_batch.get("input_ids")
		
		rotary_pos_emb = None
		# 获取 RoPE embeddings（参考 gpt_model.py _preprocess 方法）
		if self.position_embedding_type == 'rope' and self.rotary_pos_emb is not None:
			if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
				rotary_seq_len = input_ids.shape[0]  # total_tokens
			else:
				rotary_seq_len = position_ids.shape[1] if position_ids.dim() > 1 else position_ids.shape[0]
			rotary_pos_emb = self.rotary_pos_emb(
				rotary_seq_len,
				packed_seq=packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
			)
		
		mtp_in_postprocess = self.mtp_process
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
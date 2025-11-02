import logging
import os
# from turtle import shape
from typing import Optional

from AutoTuner.testbench.ops.preprocess import PreprocessForTest
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
# from tests.unit_tests.test_utilities import Utils
import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor
from transformers import PretrainedConfig

from .common import CommonOpsForTest



# class DecoderForTest(CommonOpsForTest):
#     def __init__(
#         self,
#         preprocess: PreprocessForTest,
#         decoder: TransformerBlock,
#         config: TransformerConfig,
        
#     ):
#         super().__init__(
#             hook_activation=False,
#             module_name="Decoder",
#             logging_level=logging.INFO,
#         )
#         self.preprocess = preprocess
#         self.decoder = decoder
#         self.config = config

#     def forward(self, input_ids: Tensor, position_ids: Tensor, attention_mask: Tensor, packed_seq_params: PackedSeqParams) -> Tensor:
#         rotary_pos_emb, preproc_output, attention_mask, packed_seq_params = self.preprocess(input_ids, position_ids, attention_mask, packed_seq_params)
#         (decoder_input, rotary_pos_emb, sequence_len_offset) = (
#             preproc_output[:3]
#         )
#         # seqence_length = packed_seq_params.max_seq_length if packed_seq_params is not None else input_ids.size(1)
#         sequence_length = decoder_input.size(0)
#         attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()
#         # hidden_states = torch.ones((1024, 2, self.config.hidden_size))
#         # hidden_states = hidden_states.to(torch.bfloat16)
#         # hidden_states = hidden_states.cuda()

#         print("Decoder input shape:", decoder_input.shape)
#         # hidden states
#         decoder_output = self.decoder(
#             hidden_states=decoder_input,
#             # hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             inference_context=None,
#             rotary_pos_emb=rotary_pos_emb,
#             packed_seq_params=packed_seq_params,
#             sequence_len_offset=sequence_len_offset,

#         )
#         print("Decoder output shape:", decoder_output.shape)
#         return decoder_output

#     def __call__(self, *args, **kwargs):
#         return self.forward(*args, **kwargs)

class DecoderForTest(TransformerBlock, CommonOpsForTest):
    def __init__(
        self,
        preprocess: PreprocessForTest,
        decoder: TransformerBlock,
        config: TransformerConfig,
        
    ):
        TransformerBlock.__init__(
            self,
            config,
            spec = get_gpt_layer_with_transformer_engine_spec(),
            post_process=False,
        )
        CommonOpsForTest.__init__(
            self,
            hook_activation=False,
            module_name="Decoder",
            logging_level=logging.INFO,
        )
        self.preprocess = preprocess
        self.decoder = decoder
        # self.decoder.config.attention_backend = AttnBackend.flash
        self.config = config
        # os.environ.pop('NVTE_FUSED_ATTN', None)

    def forward(self, input_ids: Tensor, position_ids: Tensor, attention_mask: Tensor, packed_seq_params: PackedSeqParams) -> Tensor:
        rotary_pos_emb, preproc_output, attention_mask, packed_seq_params = self.preprocess(input_ids, position_ids, attention_mask, packed_seq_params)
        (decoder_input, rotary_pos_emb, sequence_len_offset) = (
            preproc_output[:3] 
        )
        # # sequence_length = packed_seq_params.max_seq_length if packed_seq_params is not None else input_ids.size(1)
        # sequence_length = decoder_input.size(0)
        # # # sequence_length = 1024
        # attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()
        # hidden_states = torch.ones((sequence_length, 1, self.config.hidden_size)).cuda()
        # hidden_states = hidden_states.to(torch.bfloat16)

        # hidden_states = hidden_states.cuda()
        
        # extra_block_kwargs: dict = None,
        # print("Decoder input shape:", decoder_input.shape)
        # # hidden states
        # decoder_output = super().forward(
        #     hidden_states=decoder_input,
        #     # rotary_pos_emb=rotary_pos_emb,
        #     # hidden_states=hidden_states,
        #     attention_mask=None,
        #     packed_seq_params=packed_seq_params,
        #     # sequence_len_offset=sequence_len_offset,
        #     # **(extra_block_kwargs or {}),

        # )


        # These inputs are for testing packed sequence attention 
        # and they are coming from test_attention_packed_seq.py
        sequence_length = 32
        micro_batch_size = 1

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.decoder.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None

        packed_seq_params = make_test_packed_seq_params(sequence_length)
        print("backend for attention:")
        if  self.decoder.config.attention_backend == 1:
            print("Using FlashAttention")
        # print(self.decoder.config.attention_backend)
        decoder_output = self.decoder(
            hidden_states=hidden_states,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
        )

        print("Decoder output shape:", decoder_output.shape)
        return decoder_output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    # def _forward(
    #     self,
    #     hidden_states: Tensor,
    #     attention_mask: Tensor = None,
    #     rotary_pos_emb: Tensor = None,
    #     packed_seq_params: PackedSeqParams = None,
    #     sequence_len_offset: Tensor = None,
    #     **kwargs,
    # ) -> Tensor:
    #     return self.decoder._forward(
    #         hidden_states,
    #         attention_mask,
    #         rotary_pos_emb,
    #         packed_seq_params,
    #         sequence_len_offset,
    #         **kwargs,
    #     )


from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
# from tests.unit_tests.test_utilities import Utils


def make_test_packed_seq_params(sequence_length):
    cu_seqlens = torch.IntTensor([0, 6, 19, 22, sequence_length]).cuda()
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = seqlens.max().item()
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
    )
    return packed_seq_params


def make_test_packed_padded_seq_params(sequence_length):
    cu_seqlens = torch.IntTensor([0, 18, 44, 52, 96, 118]).cuda()
    cu_seqlens_padded = torch.IntTensor([0, 20, 48, 56, 100, sequence_length]).cuda()
    seqlens = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    max_seqlen = seqlens.max().item()
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
    )
    return packed_seq_params


# class TestParallelAttentionWithPackedSequence:

#     def setup_method(self, method):
#         Utils.initialize_model_parallel(1, 1)
#         model_parallel_cuda_manual_seed(123)
#         # use BF16 and a large enough hidden size to enable FlashAttention for thd format.
#         self.transformer_config = TransformerConfig(
#             num_layers=2,
#             hidden_size=64,
#             num_attention_heads=4,
#             use_cpu_initialization=True,
#             bf16=True,
#             params_dtype=torch.bfloat16,
#             pipeline_dtype=torch.bfloat16,
#             autocast_dtype=torch.bfloat16,
#         )
#         self.parallel_attention = SelfAttention(
#             self.transformer_config,
#             get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
#             layer_number=1,
#             attn_mask_type=AttnMaskType.causal,
#         )

#     def teardown_method(self, method):
#         Utils.destroy_model_parallel()

#     def test_cpu_forward(self):
#         # we can't currently do this because the global memory buffer is on GPU
#         pass

#     def test_gpu_forward(self):

#         config = self.parallel_attention.config
#         sequence_length = 32
#         micro_batch_size = 1

#         self.parallel_attention.cuda()

#         # [sequence length, batch size, hidden size]
#         hidden_states = torch.ones(
#             (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
#         )
#         hidden_states = hidden_states.cuda().to(torch.bfloat16)

#         attention_mask = None

#         packed_seq_params = make_test_packed_seq_params(sequence_length)
#         output, bias = self.parallel_attention(
#             hidden_states, attention_mask, packed_seq_params=packed_seq_params
#         )

#         assert config.recompute_granularity is None
#         assert output.shape[0] == sequence_length
#         assert output.shape[1] == micro_batch_size
#         assert output.shape[2] == config.hidden_size
#         assert bias.shape[0] == config.hidden_size

    
#     def test_fused_rope_gpu_forward(self):
#         self.parallel_attention.config.apply_rope_fusion = True
#         config = self.parallel_attention.config
#         sequence_length = 32
#         micro_batch_size = 1

#         self.parallel_attention.cuda()

#         # [sequence length, batch size, hidden size]
#         hidden_states = torch.ones(
#             (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
#         )
#         hidden_states = hidden_states.cuda().to(torch.bfloat16)

#         attention_mask = None
#         rotary_pos_emb = torch.ones(
#             sequence_length, 1, 1, self.parallel_attention.config.kv_channels
#         ).cuda()

#         packed_seq_params = make_test_packed_seq_params(sequence_length)
#         output, bias = self.parallel_attention(
#             hidden_states, attention_mask, packed_seq_params=packed_seq_params
#         )

#         assert config.recompute_granularity is None
#         assert output.shape[0] == sequence_length
#         assert output.shape[1] == micro_batch_size
#         assert output.shape[2] == config.hidden_size
#         assert bias.shape[0] == config.hidden_size
#         self.parallel_attention.config.apply_rope_fusion = False
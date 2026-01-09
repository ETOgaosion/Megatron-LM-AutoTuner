from AutoTuner.testbench.ops_test.atten_with_cp_test import TestAttnFuncWithCPAndKVP2P
from AutoTuner.testbench.ops_test.column_parallel_linear_test import (
    TestColumnParallelLinear,
)
from AutoTuner.testbench.ops_test.cpu_embedding_test import (
    TestLanguageModelCPUEmbedding,
)
from AutoTuner.testbench.ops_test.decoder_test_hidden import TestDecoderWithHiddenInputs
from AutoTuner.testbench.ops_test.dot_product_attention_test import (
    TestTEDotProductAttention,
)
from AutoTuner.testbench.ops_test.embedding_test import TestLanguageModelEmbedding
from AutoTuner.testbench.ops_test.gpt_model_test import TestGPTModel
from AutoTuner.testbench.ops_test.layernorm_test import TestLayerNorm
from AutoTuner.testbench.ops_test.mlpdense_test import TestMLPDense
from AutoTuner.testbench.ops_test.moe_layer_test import TestMoELayer
from AutoTuner.testbench.ops_test.postprocess_test import TestPostprocess
from AutoTuner.testbench.ops_test.preprocess_test import TestPreprocess
from AutoTuner.testbench.ops_test.row_parallel_linear_test import (
    TestTERowParallelLinear,
)
from AutoTuner.testbench.ops_test.self_attention_test import TestSelfAttention
from AutoTuner.testbench.ops_test.shared_expert_mlp_test import TestSharedExpertMLP
from AutoTuner.testbench.ops_test.te_grouped_mlp_test import TestTEGroupedMLP
from AutoTuner.testbench.ops_test.transformers_layer_test import TestTransformerLayer

OP_TEST_MAPPING = {
    "Embedding": TestLanguageModelEmbedding,
    "LanguageModelCPUEmbedding": TestLanguageModelCPUEmbedding,
    "Preprocess": TestPreprocess,
    "Decoder": TestDecoderWithHiddenInputs,
    "Postprocess": TestPostprocess,
    "LayerNorm": TestLayerNorm,
    "TransformerLayer": TestTransformerLayer,
    "GPTModel": TestGPTModel,
    "SelfAttention": TestSelfAttention,
    "MLPDense": TestMLPDense,
    "MoELayer": TestMoELayer,
    "SharedExpertMLP": TestSharedExpertMLP,
    "TEDotProductAttention": TestTEDotProductAttention,
    "TEGroupedMLP": TestTEGroupedMLP,
    "TERowParallelLinear": TestTERowParallelLinear,
    "TEAttenWithCP": TestAttnFuncWithCPAndKVP2P,
    "TEColumnParallelLinear": TestColumnParallelLinear,
}

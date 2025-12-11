from AutoTuner.testbench.ops_test.decoder_test_hidden import TestDecoderWithHiddenInputs
from AutoTuner.testbench.ops_test.dot_product_attention_test import (
    TestTEDotProductAttention,
)
from AutoTuner.testbench.ops_test.embedding_test import TestLanguageModelEmbedding
from AutoTuner.testbench.ops_test.gpt_model_test import TestGPTModel
from AutoTuner.testbench.ops_test.layernorm_test import TestLayerNorm
from AutoTuner.testbench.ops_test.moe_layer_test import TestMoELayer
from AutoTuner.testbench.ops_test.postprocess_test import TestPostprocess
from AutoTuner.testbench.ops_test.preprocess_test import TestPreprocess
from AutoTuner.testbench.ops_test.self_attention_test import TestSelfAttention
from AutoTuner.testbench.ops_test.transformers_layer_test import TestTransformerLayer

OP_TEST_MAPPING = {
    "Embedding": TestLanguageModelEmbedding,
    "Preprocess": TestPreprocess,
    "Decoder": TestDecoderWithHiddenInputs,
    "Postprocess": TestPostprocess,
    "LayerNorm": TestLayerNorm,
    "TransformerLayer": TestTransformerLayer,
    "GPTModel": TestGPTModel,
    "SelfAttention": TestSelfAttention,
    "TEDotProductAttention": TestTEDotProductAttention,
    "MoELayer": TestMoELayer,
}

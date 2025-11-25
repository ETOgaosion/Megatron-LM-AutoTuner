from AutoTuner.testbench.ops_test.decoder_test_hidden import TestDecoderWithHiddenInputs
from AutoTuner.testbench.ops_test.embedding_test import TestLanguageModelEmbedding
from AutoTuner.testbench.ops_test.gpt_model_test import TestGPTModel
from AutoTuner.testbench.ops_test.layernorm_test import TestLayerNorm
from AutoTuner.testbench.ops_test.preprocess_test import TestPreprocess
from AutoTuner.testbench.ops_test.self_attention_test import TestSelfAttention

OP_TEST_MAPPING = {
    "Embedding": TestLanguageModelEmbedding,
    "Preprocess": TestPreprocess,
    "Decoder": TestDecoderWithHiddenInputs,
    "LayerNorm": TestLayerNorm,
    "GPTModel": TestGPTModel,
    "SelfAttention": TestSelfAttention,
}

from AutoTuner.testbench.ops_test.decoder_test_hidden import TestDecoderWithHiddenInputs
from AutoTuner.testbench.ops_test.embedding_test import TestLanguageModelEmbedding
from AutoTuner.testbench.ops_test.preprocess_test import TestPreprocess

OP_TEST_MAPPING = {
    "Embedding": TestLanguageModelEmbedding,
    "Preprocess": TestPreprocess,
    "Decoder": TestDecoderWithHiddenInputs,
}

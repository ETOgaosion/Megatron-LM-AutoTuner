import warnings

from AutoTuner.testbench.ops_test.atten_with_cp_test import TestAttnFuncWithCPAndKVP2P
from AutoTuner.testbench.ops_test.column_parallel_linear_test import (
    TestColumnParallelLinear,
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

try:
    from AutoTuner.testbench.ops_test.cpu_embedding_test import (
        TestLanguageModelCPUEmbedding,
    )
    from AutoTuner.testbench.ops_test.gpt_model_enhanced_test import (
        TestGPTModelEnhanced,
    )
    from AutoTuner.testbench.ops_test.gpt_model_module_queue_test import (
        TestGPTModelModuleQueue,
    )
    from AutoTuner.testbench.ops_test.preprocess_enhanced_test import (
        TestPreprocessEnhanced,
    )
    OP_TEST_MAPPING["LanguageModelCPUEmbedding"] = TestLanguageModelCPUEmbedding
    OP_TEST_MAPPING["GPTModelEnhanced"] = TestGPTModelEnhanced
    OP_TEST_MAPPING["GPTModelModuleQueue"] = TestGPTModelModuleQueue
    OP_TEST_MAPPING["PreprocessEnhanced"] = TestPreprocessEnhanced
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        f"Please install Megatron-LM-Enhanced to run Enhanced Ops."
    )

try:
    from AutoTuner.testbench.ops_test.atten_with_cp_enhanced_test import (
        TestAttnFuncWithCPAndKVP2PNVSHMEM,
    )
    OP_TEST_MAPPING["TEAttenWithCPEnhanced"] = TestAttnFuncWithCPAndKVP2PNVSHMEM
except (ImportError, ModuleNotFoundError) as e:
    print(e)
    warnings.warn(
        f"Please install TransformerEngine-Enhanced with NVSHMEM support to run TEAttenWithCPEnhanced."
    )
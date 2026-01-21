# Integrate New CP Attn Operator

CONTEXT: This is an RLHF training engine auto-tuner. You need to integrate and test new Context-Parallel Implementation
TASK:
    - Original CP test operator is in [AutoTuner/testbench/ops/atten_with_cp.py](../../AutoTuner/testbench/ops/atten_with_cp.py)
    - New CP in TE is in [TransformerEngine-Enhanced/transformer_engine/pytorch/attention/dot_product_attention/context_parallel_nvshmem.py](../../TransformerEngine-Enhanced/transformer_engine/pytorch/attention/dot_product_attention/context_parallel_nvshmem.py), Note that if you need to import TransformerEngine-Enhanced package, its name is still TransformerEngine, you can use a try-catch to remind user install TransformerEngine-Enhanced
    - Use new CP to write a attn_with_cp_enhanced.py
    - Add new op to op_mapping.py
EXAMPLES: Original CP test operator is in [AutoTuner/testbench/ops/atten_with_cp.py](../../AutoTuner/testbench/ops/atten_with_cp.py)
OUTPUT: Follow other operators is OK
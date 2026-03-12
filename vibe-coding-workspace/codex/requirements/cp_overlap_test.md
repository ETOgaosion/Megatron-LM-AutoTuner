# Requiremet: CP overlap test

OBJECT: Follow TP overlap in [AutoTuner/Profiler/overlap/tp](../../../AutoTuner/Profiler/overlap/tp) and implement CP overlap
INPUT: Different max_token_lens and seqlen. (seqlen is max sequence length for each sequence)
TASK: 
1. Execute: follow TP, but you should use TEAttenWithCPEnhanced op and use CP=2
2. Analyze: analyze json results like [outputs/2026-03-12_04-56-40/Qwen/Qwen3-0.6B/torch_profiler/jss-Rack-Server_7294.1773291659494299009.pt.trace.json](../../../outputs/2026-03-12_04-56-40/Qwen/Qwen3-0.6B/torch_profiler/jss-Rack-Server_7294.1773291659494299009.pt.trace.json)
    - Compute is `flash::flash_fwd_kernel` in streams with only 1 `flash::flash_fwd_kernel` in it
    - Communication is `Memcpy PtoP (Device -> Device)` in streams with only 1 `Memcpy PtoP (Device -> Device)` in it
    - There are many Compute and Communication stream, you should find them in 1 `TransformerEngine-Enhanced/transformer_engine/pytorch/attention/dot_product_attention/context_parallel_nvshmem.py(479): forward`
    - You should compute overlap ratio like tp_overlap
3. Output like tp_overlap
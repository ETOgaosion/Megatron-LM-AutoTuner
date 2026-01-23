# Tensor Parallel(TP) Overlap Tuner

CONTEXT: This is an RLHF training engine auto-tuner, for better performance. We will implement a fully load-balanced training engine. Now you should write codes and scripts to adjust configurations to enable tp communication/computation overlap. Your codes shall mainly in [AutoTuner/Profiler/overlap](../../AutoTuner/Profiler/overlap/tp_overlap) and [AutoTuner/utils](../../AutoTuner/utils), This needs a workflow.
WORKFLOW:
0. [INPUT] User input a model config
1. Use the config to generate test cases you need to test. You should use binary search to find out the overlappable TP communication/computation overlap config
    - Initally use TP=2 to see whether TP is needed (if cannot overlap, then turn off TP)
    - You can adjust TP size to decrease computation pressure
    - Use a reletively high setting of max_token_len is OK, let the computational intensity get to peak
2. Run the configs, launch TP size GPUs, use torch profile as [AutoTuner/testbench/profile/launcher/torch_profile_launch.py](../../AutoTuner/testbench/profile/launcher/torch_profile_launch.py)
    - There are some specific config of TP overlap, we can config it use a file like [AutoTuner/testbench/profile/configs/tp_comm_overlap_cfg.yaml](../../AutoTuner/testbench/profile/configs/tp_comm_overlap_cfg.yaml)
    - Read [https://github.com/NVIDIA/TransformerEngine/issues/1344#issuecomment-2533487634](https://github.com/NVIDIA/TransformerEngine/issues/1344#issuecomment-2533487634) for detailed infomation of TP overlap config
    - From my point of view, there is no meaning of Pipeline reduce-scatter, you can only use ring exchange to avoid occupying normal computation SM. For bulk overlap, you should judge how many SM is the most efficient configuration
3. Analyze the results in `outputs` directory
    - From torch profile result, you should write code to analyze json file (the sample is in [outputs/sample/jss-Rack-Server_951.1768541469734146129.pt.trace.json](../../outputs/sample/jss-Rack-Server_951.1768541469734146129.pt.trace.json), do not read all of it, only check elements structure)
        - In forward pass of column parallel linear: You should care about 2 kinds of streams set (see [forward_streamsets](./photos/forward_streamsets.png), [backward_streamsets](./photos/backward_streamset.png))
            - first is forward: ProfilerStep#2 with GEMM like `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8::Params)`, ProfilerStep#2 with comm stream `Memcpy PtoP (Device -> Device)` and `kuserbuffers_pushsend(int*, int*, int4*, int4*, int)`
            - backward: GEMM: `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_nn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_nn_align8::Params)`, all-gather: `void userbuffers_fp16_sum_inplace_gpu_rw_ag<2>(int, int, int, int, int, int, int, void**, int, unsigned long)`, reduce-scatter: `void userbuffers_fp16_sum_inplace_gpu_rr_rs<2>(int, int, int, int, int, int, int, void**, int, unsigned long)`
    - According to each comm and computation overlapable or not, generate report
        - if $tp\_new$ is n times of $tp$, the $Time(tp\_new)$ shall be 1/n time of $tp$, or else just use tp as the final config
4. For each operator, generate report of tp usage and execution time of a Linear op
TASK:
0. TP devides all linear operators, so you need to test ColumnParallelLinear and RowParallelLinear with different inputs x outputs (attention/MLP), `["fc1", "fc2", "qkv", "proj"]`
    - For attention, weights: [hidden_size, query_proj_size + 2 * kv_proj_size]
    - For MLP, weights: [hidden_size, ffn_hidden_size] (fc1: ColumnParallelLinear), [ffn_hidden_size, hidden_size] (fc1: RowParallelLinear)
1. You should design APIs to receive model config, you can use the same mechanism in [AutoTuner/testbench/profile](../../AutoTuner/testbench/profile)
2. You should use our implemented single operator [AutoTuner/testbench/ops/column_parallel_linear.py](../../AutoTuner/testbench/ops/column_parallel_linear.py) and [AutoTuner/testbench/ops/row_parallel_linear.py](../../AutoTuner/testbench/ops/row_parallel_linear.py)
3. You should enable TP Communication overlap and use the configs as the run_micro_batch function (Search `tp_comm_overlap`) [AutoTuner/testbench/ops_test/common.py](../../AutoTuner/testbench/ops_test/common.py)
IMPLEMENTATION: There are 2 kinds of implementation, I think you can try the first, the whole implementation may not all in python, you can use shell scripts.
1. directly use our profiler, and we may not need to test so many cases, just use `--run-one-data`
2. write your own profiler
CONSTRAINTS: TP only happens in a single machine (8 GPUs, TP < 8)
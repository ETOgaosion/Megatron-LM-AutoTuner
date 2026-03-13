# Requirements: Tuning Algorithm - Phase 1 Dense Model

## Step 1: Run and Read Overlap Data

- INPUT: user input model config

- CONTEXT: tp and cp overlap analysis program is in [AutoTuner/Profiler/overlap](../../../AutoTuner/Profiler/overlap)

- TASK:
    - Drive and output results
    - Analysis results and find the most overlap ratio task

## Step 2: Configure TP/CP/max_token_len

- INPUT: user input prompt length + resp length, sum to be the seqlen of a single data
- TASK:
    - TP is decided on step 1
    - max_token_len should be CP * (max_token_len in Step 1)
    - CP should be < nprocs_per_node / TP

## Step 3: Decide Activation Handle Strategy

- CONTEXT: 
    - Megatron supports: no_recompute, selective_recompute, full_recompute, for selective recompute, the modules able to select include `["core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe", "shared_experts"]`, the most useful is `mlp` and `moe`, but the computing time is higher.
    - Megatron has recompute_offload strategy: [https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/fine_grained_activation_offloading.html](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/fine_grained_activation_offloading.html)
- GOAL: In [AutoTuner/Profiler/activations](../../../AutoTuner/Profiler/activations), implement an activation handle strategy decider.
    - The decider includes: selective recompute policy and activation offload policy
- TASK:
    - Use the current config (TP/CP/max_token_len) to run a TransformerLayer in [AutoTuner/testbench](../../../AutoTuner/testbench), use no_recompute, capture the activation size and the compute time
    - hook and find out activation size of each part in `["attn_norm", "core_attn", "attn_proj", "mlp_norm", "expert_fc1", "moe_act"]`
    - Compute the Activation offload time using activation size of each config and GPU-CPU transmission PCIe bandwidth [AutoTuner/testbench/functional/cpu_gpu_movements](../../../AutoTuner/testbench/functional/cpu_gpu_movements)
    - Decide Activation Offload strategy, the time for activation offload must be within the TransformerLayer computation time
    - After decide Activation Offload strategy, use activation recompute to handle the left parts, decide activation recompute strategy

## Step 4: Decide PP

- CONTEXT:
    - Now assume that each PP rank are balanced by our enhanced system, and no need to consider ops other than TransformerLayer since they are either in CPU or switch to GPU and use when they are needed
    - In [AutoTuner/runtime/baseline](../../../AutoTuner/runtime/baseline), you can run baseline with the TP/CP/max_token_len/seqlen settings decided
- TASK:
    - Run [AutoTuner/runtime/baseline](../../../AutoTuner/runtime/baseline) for determined TP/CP/max_token_len/seqlen/recompute config
    - Get the model size and activation size of the config for unit module
    - use the model config and the unit size infomation, calculate PP config, holding what number of layers
    - for irregular total number of layers, layer distribution should be `(PP size - 1) * num_layers + num_layers_post_process`, where `num_layers_post_process < num_layers`
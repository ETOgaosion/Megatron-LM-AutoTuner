# Requirements: Tuning Algorithm - Phase 2 per TransformerLayer

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

## Step 3: Decide EP

- INPUT: user input model config

- CONTEXT: EP decrease experts computation and increase communication overhead, you should make sure that the EP communication can be overlapped in whole model pipeline schedule process
    - Overlap Ensurance
        - EP Dispatch forward should be overlapped with attention wgrad process
        - EP Combine backward should be overlapped with MLP forward
        - EP Combine forward should be overlapped with MLP backward
        - EP Dispatch backward should be overlapped with MLP wgard
- TASK:
    - Implement Attention and MLP wgrad calculation in [AutoTuner/testbench](../../../AutoTuner/testbench)
    - Run testbench to get forward/backward/wgrad time of attention and MLP
    - for EP in [1, 2, 4, 8, 16, ...]
        - Calculate EP communication time, use `communication amount / all-to-all bandwidth(intra/inter-node)`
        - find whether EP comm can overlap and the computation time is shortest
    
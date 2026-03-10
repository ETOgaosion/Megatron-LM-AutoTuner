# Requirement: Test EPLB and LPLB

CONTEXT: There are [EPLB](../../../EPLB/) and [LPLB](../../../LPLB/) to handle MoE experts imblance problems, implemented by DeepSeek.

HANDLING_MODELS:
1. Qwen/Qwen1.5-MoE-A2.7B
2. Qwen/Qwen3-30B-A3B-Base
3. Qwen/Qwen3-235B-A22B
4. deepseek-ai/DeepSeek-V3-Base

TASK:
1. Read EPLB and LPLB for there APIs and examples/tests
2. You should add tests on EPLB and LPLB with different models listed in HANDLING_MODELS
    - Only test EPLB and LPLB handle 1 MoE layer
    - Mock the weights data or any data you need
3. You should add scripts to measure running time of a single task
    - A single task means EPLB and LPLB handle 1 MoE layer
4. Draw graph, our system vs EPLB and LPLB, our time consumption is 0
    - output to `outputs` directory
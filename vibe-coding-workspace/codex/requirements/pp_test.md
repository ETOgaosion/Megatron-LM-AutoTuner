# Requirement 2: Add Pipeline Communication test

CONTEXT: This is an RLHF training engine auto-tuner, for better performance. We will implement a fully load-balanced training engine. You need to help me add some tests abount different parallelism
TASK: Your task is to implement a unit test about PP communication and normal calculation in [AutoTuner/testbench/functional](../../../AutoTuner/testbench/functional)
    - You should use a multi-node distributed programming arch like [AutoTuner/runtime/baseline/main.py](../../../AutoTuner/runtime/baseline/main.py) and [tests/functional_test/runtime/runtime_baseline_run.sh](../../../tests/functional_test/runtime/runtime_baseline_run.sh)
    - You should use megatron's implementation about PP communication in [Megatron-LM/megatron/core/pipeline_parallel](../../../Megatron-LM/megatron/core/pipeline_parallel)
    - You can assume that we will test PP comm in diferent machine, we offer different parallel size and 
CONSTRAINTS: Remember your guidance in [guidance](../guidance/), in plan mode first and write your plans
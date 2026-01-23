# Runtime Implementation Requirements

Now please implement a runtime for our project.

CONTEXT: This is an RLHF training engine auto-tuner, for better performance. We will implement a fully load-balanced training engine. You need to develop a runtime in convinience of testing
TASK: Your task is to implement a runtime in [AutoTuner/runtime](../../AutoTuner/runtime)
    - We have a testbench [AutoTuner/testbench](../../AutoTuner/testbench), there is an operator GPTModel, but this one only support tp/cp/etp/ep, which not support all parallelism
    - You need to write a runtime use this operator and support all parallelism
    - In [verl/verl/models/mcore](../../verl/verl/models/mcore), there are examples of using GPTModel, you may need to follow the `get_model` function in [verl/verl/utils/megatron_utils.py](../../verl/verl/utils/megatron_utils.py)
    - Create functional test in [tests/functional_test/runtime] using a script
CONSTRAINTS: Only for last pipeline stage (post process), consider share embedding output layer
EXAMPLES: View the image in [here](../requirements/photos/module_queue.png)
OUTPUT: Should be transparent, Same as not enable this

I've checkout correct branch in [Megatron-LM-Enhanced](../../Megatron-LM-Enhanced).
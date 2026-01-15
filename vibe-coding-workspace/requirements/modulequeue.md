# ModuleQueue Design Requirements

Now please implement another feature: For last PP stage, support a ModuleQueue inherit GPTModel, when a layer calculated, offload its weight to CPU, and load post_process weight from CPU to GPU (initially in CPU), In backward pass reverse this process. calculate post_process weight size, you can devide it into some chunks which can be adjusted to make sure a layer computation can cover its offload and load time

CONTEXT: This is an RLHF training engine auto-tuner, for better performance. We will implement a fully load-balanced training engine.
TASK: Your task is to implement a ModuleQueue class inherit GPTModel, in Megatron-LM-Enhanced, please re-use as much parent code as possible. Note that ModuleQueue is only used to replase post_process GPTModel.
    - Process & Functions: The ModuleQueue is two Queues in GPU and CPU. You should keep the post process weights in CPU during initialization. 
        - In forward pass, whenever a layer calculated, it is offloaded to CPU, and in the meantime, load part of postprocess weights to GPU, util all post process weights ready (And no more normal layers needed offload).
            - Note that the loading times of post process weights should be able to set using arguments & TransformerConfig
        - In backward pass, the post process layer is calculated first, after its calculation, offload it and load the last-offloaded layer to GPU until 1st layer is ready.
    - write a unit test to test ModuleQueue, unit test shall only test post_process training
    - Integrate it into Megatron-LM-Enhanced, when postprocess, use ModuleQueue to build GPTModel
CONSTRAINTS: Only for last pipeline stage (post process), consider share embedding output layer
EXAMPLES: View the image in [here](../requirements/photos/module_queue.png)
OUTPUT: Should be transparent, Same as not enable this

I've checkout correct branch in [Megatron-LM-Enhanced](../../Megatron-LM-Enhanced).
# Convinient testbench for observation

Through the testbench of AutoTuner, we can make straight-forward observation of the operators performance.

There are 3 ways of testbench launching methods: get_data launch, nsys profile launch and torch profiler launch. The first way means we just use our timers and memory sensors to detect an operators performance and return the data for future use. Nsys profile discards our sensors and use NVIDIA nsight system to profile. Torch profiler uses `torch.profiler.profile` API to profile with timeline, stacks and other info.

The design of testbench follows OOP rules.

The structures of testbench:

```sh
AutoTuner
├── testbench
│   ├── functional      # test base capability like CPU-GPU bandwidth
│   ├── __init__.py
│   ├── ops             # override of Megatron operators, mainly for nvtx insertion
│   ├── ops_test        # wrapper of operator testing
│   └── profile         # profile main entrance
│       ├── cases                       # test cases (models and externel configs)
│       ├── configs                     # configs to override huggingface model config and transformer config
│       ├── __init__.py
│       ├── launcher                    # different ways to launch model runner
│       ├── main.py                     # main entrence of get_data launch way
│       ├── nsys_main.py                # main entrence of nsys profile launch way
│       └── op_mapping.py
└── utils               # assistence function, to avoid repetition
```

## Which operators to be added?

- embedding
- TransformerBlock
- TransformerLayer
- pre_layernorm
- self_attn
- rope
- TEDotProductAttention
- linear_proj
- self_attn_bda
- MLP
- Norm​
- fc1_op​
- activation_op​
- fc2_op
- mlp_bda
- final_layernorm
- Post process

## How to add operators

Implement operators in `ops` folder and `ops_test` folder, add op to `AutoTuner/testbench/profile/op_mapping.py`.

For inputs, we may need to checkpoint operators' input. But actually, the attention_mask and position_ids are OK to be meaningful, and the hidden_states may be not that useful, so we can generate random hidden_states for these.

## DeepSeek-V3 Runtime Configuration Requirements

When running tests for DeepSeek-V3, Multi Token Prediction (MTP) must be disabled. This can be configured in AutoTuner/testbench/profile/configs/local/override_model_config.json with the following setting:

```json
{
    "num_nextn_predict_layers": 0
}
```

Additionally, the number of model layers can be modified in the same configuration file:

```json
{
    "num_hidden_layers": 2
}
```

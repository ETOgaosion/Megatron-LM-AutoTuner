# AutoTuner for Megatron + TransformerEngine

## Scenarios

This is a practical auto-tuner on Megatron targeted at post-training frameworks like [verl](https://github.com/volcengine/verl) project.

Our performance tuning target is MFU in MCore training process on both forward-only models and forward-backward-update models, which leads to high performance on training side in RLHF.

Speaking of tuning dimensions, currently include:

|   |   |   |   |   |   |
|---|---|---|---|---|---|
|Dense Layer|TP|CP|DP|PP|VPP|
|MoE Parallel Folding|ETP|EP|EDP|||
|Pipeline layout|||||
|Seqlen|max_token_len|||||
|Recompute|recompute_granularity|recompute_method|recompute_num_layers|recompute_modules||

Target shapes:

- thd (mainly)
- bshd

## Docs

[Doc in Lark](https://acs-frontier.feishu.cn/wiki/JRfAwjGeMiGmwWk3IM7ct26Tn2e)

Go to [docs directory](./docs/)

- [Install](./docs/Install.md)
- [Quick Start](./docs/QuickStart.md)

## Submodules

### Original Open-Source Repos

- Megatron-LM
- TransformerEngine
- verl

### Enhenced Repos

- Megatron-LM-Enhanced
- TransformerEngine-Enhanced
- verl-enganced
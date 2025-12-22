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

### Enhanced Repos

- [Megatron-LM-Enhanced](https://github.com/ETOgaosion/Megatron-LM)
    - @Jetteyc : NetworkEngine
    - @ETOgaosion , @cyn1456492382 , @miceforrat : All basic functions and Auto-Tuner connection
- [TransformerEngine-Enhanced](https://github.com/ETOgaosion/TransformerEngine)
    - @Jetteyc , @Knight-of-Thunder : Context Parallel based on NVSHMEM Async Transport
- [verl-enhanced](https://github.com/ETOgaosion/verl)
    - @ETOgaosion , @cyn1456492382 , @miceforrat : balanced data resharding
    - @LeonardW-sl : Scalable Train-Infer Data & Weights Transport
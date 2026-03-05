# Simulator README

This directory has two scripts:

- `simulate_ep_balance/simulate_ep_balance.py`: rebalance experts for EP and compare before/after.
- `simulate_ep_balance/compare_with_eplb.py`: compare `before` vs `ours` vs `EPLB`.

## Environment

```bash
conda activate megatron-lm-autotuner
```

## Quick Start

### 1) Our EP rebalance only

```bash
python simulator/simulate_ep_balance/simulate_ep_balance.py --ep-sizes 2 4 6 10 12
```

Output:
- `outputs/simulate_ep_balance/<timestamp>/`

Main files:
- `balance_results.json`
- `variance_summary.csv`
- `gpu_token_balance_ep*.png`
- `variance_comparison_across_ep_sizes.png`

### 2) Compare with EPLB (recommended: JSON config)

```bash
python simulator/simulate_ep_balance/compare_with_eplb.py \
  --config simulator/simulate_ep_balance/compare_with_eplb_config.json
```

Run subset EP sizes:

```bash
python simulator/simulate_ep_balance/compare_with_eplb.py \
  --config simulator/simulate_ep_balance/compare_with_eplb_config.json \
  --ep-sizes 4 10
```

Output:
- `outputs/simulate_ep_balance/compare_with_eplb/<timestamp>/`

Main files:
- `compare_with_eplb_results.json`
- `compare_with_eplb_summary.csv`
- `token_balance_compare_ep*.png`
- `memory_compare_all_ep_sizes.png` (stacked bars: weights + activation)
- `variance_summary_compare.png`

## Compare Config

Template:
- `simulator/simulate_ep_balance/compare_with_eplb_config.json`

Minimal format:

```json
{
  "data_path": "simulator/simulate_ep_balance/data/routed_experts_stats.json",
  "model_config": "/data/common/models/Qwen/Qwen1.5-MoE-A2.7B/config.json",
  "activation_multiplier": 1.0,
  "moe_layer_act_m": 180.0,
  "moe_layer_expert_act_m": 66.0,
  "ep_defaults": {
    "eplb_redundant_ratio": 0.2,
    "eplb_min_redundant": 0,
    "num_groups": 1,
    "num_nodes": 1
  },
  "ep_settings": [
    { "ep_size": 4, "num_replicas": 72 },
    { "ep_size": 10, "num_replicas": 80 }
  ]
}
```

Rules:
- `num_replicas >= num_experts`
- `num_replicas % ep_size == 0`
- CLI args override config values.

## EPLB Argument Meanings

For:

```python
phy2log, log2phy, logcnt = eplb.rebalance_experts(
    weight, num_replicas, num_groups, num_nodes, num_gpus
)
```

- `num_replicas`: total physical experts after replication.
- `num_groups`: expert groups used by EPLB grouping.
- `num_nodes`: topology groups for hierarchical packing.
- `num_gpus`: total GPUs participating in balancing.

In this simulator:
- `num_gpus = ep_size`

## Memory Model (compare script)

Per method, max GPU memory is:
- `weight_memory + activation_memory`

- `weight_memory`: from model config (`hidden_size`, `moe_intermediate_size`, dtype), multiplied by MoE layers and physical experts per GPU.
- `weight_memory`: includes
  - routed experts (replicated by EP/EPLB),
  - shared experts (`shared_expert_intermediate_size` from model config).
- `activation_memory`: uses MoELayer profile split
  - total per-layer activation (`moe_layer_act_m`, default 180M elements),
  - expert-dependent part (`moe_layer_expert_act_m`, default 66M elements),
  - scaled by routed-token ratio vs baseline peak.

Note:
- For conservative comparison, EPLB activation is not allowed to be lower than our method.

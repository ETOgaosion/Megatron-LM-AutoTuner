# Runtime Functional Test

This directory provides a minimal runtime check for the baseline GPTModel runtime.

## Usage

1. (Optional) source runtime environment from `.secrets/env.sh`.
2. Activate the recommended conda env and repo-local Python path:

```bash
conda activate megatron-lm-autotuner
export PYTHONPATH=verl:Megatron-LM
```

2. Edit `tests/functional_test/runtime/runtime_baseline_config.json`:
   - JSON root must be an array
   - each entry must be `{ "model_name": "...", "configs": { ... } }`
   - `configs` can include `case` / `parallel` / `runtime` / `paths` / `env`
   - example with full fields: `Qwen/Qwen2.5-0.5B` in `runtime_baseline_config.json`
   - optional sample file: `tests/functional_test/runtime/runtime_baseline_config_sample.json`
   - distributed launch args are from shell env vars: `MASTER_ADDR`, `MASTER_PORT`, `NUM_NODES`, `NODE_RANK`
3. Run:

```bash
bash tests/functional_test/runtime/runtime_baseline_run_qwen_longctx.sh
```

Custom config path:

```bash
bash tests/functional_test/runtime/runtime_baseline_run_qwen_longctx.sh /path/to/config.json
```

Dry run (print command only):

```bash
bash tests/functional_test/runtime/runtime_baseline_run_qwen_longctx.sh \
  tests/functional_test/runtime/runtime_baseline_config.json \
  --dry-run
```

Filter models by substring:

```bash
MODEL_FILTER=Qwen3-8B bash tests/functional_test/runtime/runtime_baseline_run_qwen_longctx.sh
```

The wrapper script calls `runtime_baseline_run_from_config.py`, which generates test case JSON files and launches `torchrun -m AutoTuner.runtime.baseline.main` per model.

## Simulation Example

For a single, copy-pasteable baseline run that also prints simulated PP/DP time from the output summary:

```bash
bash tests/functional_test/runtime/runtime_baseline_run_simulation.sh
```

Useful simulator env vars:

```bash
export AUTOTUNER_RUNTIME_USE_FUSED_KERNELS=1
export AUTOTUNER_BASELINE_DP_ALLREDUCE_BANDWIDTH_GBPS=50
export AUTOTUNER_BASELINE_DP_ALLREDUCE_LATENCY_US=30
```

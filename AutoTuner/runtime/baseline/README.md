# Runtime Baseline

`AutoTuner/runtime/baseline` provides a reproducible runtime baseline for Megatron-Core training-style execution (forward + backward) with synthetic inputs.

It is intended for:
- Comparing throughput/MFU across parallel strategies and input shapes (`bshd` vs `thd`)
- Validating memory usage per rank
- Producing stable baseline JSON reports for later tuning or regression checks

## Entry Points

- `main.py`: CLI parsing, distributed init/teardown, config loading, output directory setup.
- `launcher.py`: model/data construction and per-test-case runtime loop with metric collection.

## How It Runs

1. Load test cases from JSON (`InputTestCase` list).
2. Initialize distributed process groups and Megatron model parallel state.
3. Build HF config + TransformerConfig (with some runtime-friendly defaults enabled).
4. Generate synthetic microbatches (rank 0) and broadcast to all ranks.
5. For each test case, run `max_iterations` iterations of Megatron forward/backward.
6. Record measured iteration metrics, simulate full-model PP/DP runtime from the lightweight launcher result, exclude warmup iterations for averages, and save reports.

Notes:
- Execution uses `forward_only=False`, so this is a train-step style baseline (no optimizer step).
- `run_one_data` limits each test case to one microbatch.

## Quick Start

Run directly:

```bash
torchrun \
  --nproc_per_node 4 \
  --nnodes 1 \
  --master_addr localhost \
  --master_port 6010 \
  -m AutoTuner.runtime.baseline.main \
  --model-name Qwen/Qwen3-0.6B \
  --test-cases-file qwen3_0_6b.json \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-parallel-size 1 \
  --expert-tensor-parallel-size 1
```

Or use the provided scripts:
- `tests/functional_test/runtime/runtime_baseline_run.sh`
- `tests/functional_test/runtime/runtime_baseline_run_qwen_longctx.sh`
- `tests/functional_test/runtime/runtime_baseline_run_simulation.sh`
- `tests/functional_test/runtime/runtime_baseline_run_simulation_example.sh`

## Run And Simulate

Recommended environment:

```bash
conda activate megatron-lm-autotuner
export PYTHONPATH=verl:Megatron-LM
```

Run one baseline job and emit both measured runtime and simulated full-model runtime:

```bash
bash tests/functional_test/runtime/runtime_baseline_run_simulation.sh
```

The example script:
- launches `torchrun -m AutoTuner.runtime.baseline.main`
- keeps fused-kernel post-process enabled by default with `AUTOTUNER_RUNTIME_USE_FUSED_KERNELS=1`
- sets simulator DP knobs with `AUTOTUNER_BASELINE_DP_ALLREDUCE_BANDWIDTH_GBPS` and `AUTOTUNER_BASELINE_DP_ALLREDUCE_LATENCY_US`
- prints the latest measured vs simulated summary after the run

If you want to tune the simulator only, keep the model/test case fixed and change these env vars before running:

```bash
export AUTOTUNER_BASELINE_DP_ALLREDUCE_BANDWIDTH_GBPS=900
export AUTOTUNER_BASELINE_DP_ALLREDUCE_LATENCY_US=8
bash tests/functional_test/runtime/runtime_baseline_run_simulation.sh
```

If you want to disable the actor-style fused forward path:

```bash
export AUTOTUNER_RUNTIME_USE_FUSED_KERNELS=0
bash tests/functional_test/runtime/runtime_baseline_run_simulation.sh
```

## Required Inputs

### 1) Test cases file

Default search path: `AutoTuner/testbench/profile/cases/local/`

Expected structure:

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "cases": [
    {
      "batch_size": 128,
      "micro_batch_size": 2,
      "seqlen": 2048,
      "max_token_len": 8192,
      "shape": "bshd",
      "system": "megatron"
    }
  ]
}
```

### 2) Config files

Default search path: `AutoTuner/testbench/profile/configs/local/`

- `override_model_config.json` (required)
- `override_tf_config.json` (required)
- `tp_comm_overlap_cfg.yaml` (optional; used only when `tp_comm_overlap` is enabled)

## CLI Arguments

Core:
- `--model-name` (required)
- `--test-cases-dir` (default: `AutoTuner/testbench/profile/cases/local/`)
- `--test-cases-file` (required)
- `--config-dir` (default: `AutoTuner/testbench/profile/configs/local/`)
- `--override-model-config-file` (default: `override_model_config.json`)
- `--override-tf-config-file` (default: `override_tf_config.json`)
- `--tp-comm-overlap-cfg` (default: `tp_comm_overlap_cfg.yaml`)
- `--output-dir` (default: `outputs`)

Runtime loop:
- `--num-test-cases` (default: all)
- `--run-one-data` (flag)
- `--max-iterations` (default: `10`)
- `--warmup-iterations` (default: `3`)

Model behavior:
- `--share-embeddings-and-output-weights [true|false]` (default: follow HF `tie_word_embeddings`)
- `--no-ddp` (disable DDP wrapping)

Distributed parallel sizes:
- `--tensor-model-parallel-size` (default: `1`)
- `--pipeline-model-parallel-size` (default: `1`)
- `--virtual-pipeline-model-parallel-size` (default: `None`)
- `--context-parallel-size` (default: `1`)
- `--expert-parallel-size` (default: `1`)
- `--expert-tensor-parallel-size` (default: `1`)

Logging:
- `--log-level` (default: env `AUTOTUNER_LOG_LEVEL` or `INFO`)

## Output Layout

Results are saved to:

`<output-dir>/<timestamp>/<model-name>/runtime_baseline/`

Files:
- `runtime_baseline.json`: full run report with per-iteration metrics for each test case
- `runtime_summary.json`: compact per-test-case summary (warmup-filtered averages)
- `args.json`: resolved CLI args and resolved file paths

To inspect the latest summary quickly:

```bash
python - <<'PY'
import json
from pathlib import Path

paths = sorted(Path("outputs").glob("*/**/runtime_baseline/runtime_summary.json"))
summary_path = paths[-1]
print(summary_path)
data = json.loads(summary_path.read_text())
for item in data:
    print(
        item["test_case_idx"],
        item["time_s"],
        item["simulated_time_s"],
        item["simulated_pp_compute_time_s"],
        item["simulated_dp_allreduce_time_s"],
    )
PY
```

## Metrics

Per iteration:
- `time_s`
- `simulated_time_s`
- `total_tokens`, `total_sequences`
- `throughput_tokens_s`, `throughput_tokens_s_per_gpu`
- `simulated_throughput_tokens_s`, `simulated_throughput_tokens_s_per_gpu`
- `throughput_sequences_s`, `throughput_sequences_s_per_gpu`
- `simulated_throughput_sequences_s`, `simulated_throughput_sequences_s_per_gpu`
- `mfu`, `simulated_mfu`
- `memory_by_rank` (peak allocated/reserved/real-detected + device totals)
- `simulation`:
  - PP stage layer counts for lightweight/full model
  - measured per-rank PP stage forward/backward time from the lightweight run
  - simulated PP compute time from step-wise bottleneck scheduling
  - simulated DP all-reduce time from full stage parameter bytes

Summary per test case:
- Averages computed after dropping `warmup_iterations` (or all iterations if none remain)
- Memory summary keeps max peak values across valid iterations per rank
- `time_s` is the measured lightweight runtime
- `simulated_time_s = simulated_pp_compute_time_s + simulated_dp_allreduce_time_s`

## Important Behavior and Caveats

- Lightweight baseline layer construction:
  - Runtime baseline builds `pp * vpp` decoder layers in total (one decoder layer per virtual chunk).
  - This means each PP rank builds `vpp` decoder layers, instead of all original model layers on that rank.
  - Each built layer is remapped to the first layer index of its theoretical chunk from the full model.
  - Embedding and post-process/output behavior is unchanged: only first PP stage has embedding (`pre_process=True`), only last PP stage has post-process/output (`post_process=True`).
- Full-model simulator:
  - PP compute is reconstructed from measured per-rank PP stage forward/backward time in the lightweight run.
  - Each PP rank time is scaled by its theoretical full-model layer count.
  - PP step bottlenecks are simulated from a training-style 1F1B pipeline schedule, using the slowest active PP stage at each scheduled step.
  - DP time is simulated as a per-stage gradient all-reduce over the full-model parameter bytes on that stage, and the slowest stage all-reduce is added to the iteration time.
- Loss/post-process path:
  - Baseline uses actor-style next-token log-prob post-processing instead of a raw logits mean reduction.
  - When enabled, baseline patches the same fused MCore forward path used by `verl` Megatron actor code.
- `vpp` constraint: when `pipeline-model-parallel-size <= 1`, `virtual-pipeline-model-parallel-size` must be `None`.
- TP overlap user buffers are initialized only when:
  - tensor parallel world size > 1
  - `tp_comm_overlap` is enabled in TransformerConfig
  - input shape is compatible (`bshd` path with fixed token count)
- For `thd`/variable sequence length path, TP overlap user-buffer mode is skipped by design.
- Synthetic input masks can be reused from `tmp/` to keep compute amount stable across runs.

## Environment Variables

- `AUTOTUNER_LOG_LEVEL`: default log level when `--log-level` is not passed.
- `AUTOTUNER_LOG_ALL_RANKS`: if truthy (`1/true/yes/on`), microbatch logs are emitted from all ranks.
- `AUTOTUNER_RUNTIME_USE_FUSED_KERNELS`: enables actor-style fused MCore forward patching in baseline (default: `1`).
- `AUTOTUNER_BASELINE_DP_ALLREDUCE_BANDWIDTH_GBPS`: simulator DP all-reduce bandwidth override (default: `50`).
- `AUTOTUNER_BASELINE_DP_ALLREDUCE_LATENCY_US`: simulator DP all-reduce per-hop latency override in microseconds (default: `30`).

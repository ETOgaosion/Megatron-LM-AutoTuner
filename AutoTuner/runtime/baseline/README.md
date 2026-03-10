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
6. Record iteration metrics, exclude warmup iterations for averages, and save reports.

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

## Metrics

Per iteration:
- `time_s`
- `total_tokens`, `total_sequences`
- `throughput_tokens_s`, `throughput_tokens_s_per_gpu`
- `throughput_sequences_s`, `throughput_sequences_s_per_gpu`
- `mfu`
- `memory_by_rank` (peak allocated/reserved/real-detected + device totals)

Summary per test case:
- Averages computed after dropping `warmup_iterations` (or all iterations if none remain)
- Memory summary keeps max peak values across valid iterations per rank

## Important Behavior and Caveats

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


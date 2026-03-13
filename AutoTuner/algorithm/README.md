# AutoTuner Algorithm

This directory contains the top-level tuning algorithms that convert a model config and sequence-length target into recommended parallelism settings.

There are three entry points:

- `main.py`: dispatches to dense or MoE tuning.
- `dense_tuning.py`: 4-step algorithm for dense models.
- `moe_tuning.py`: 3-step algorithm for MoE models.

Shared step-1 and step-2 logic lives in `common.py`.

## What The Algorithm Produces

For a given model, prompt length, response length, and `nprocs_per_node`, the algorithm produces:

- `TP`: tensor parallel size
- `CP`: context parallel size
- `max_token_len`
- Dense only: activation handling strategy and `PP`
- MoE only: expert parallel strategy

The algorithm is intended to sit above the lower-level profilers in:

- `AutoTuner.Profiler.overlap.tp`
- `AutoTuner.Profiler.overlap.cp`
- `AutoTuner.Profiler.activations`
- `AutoTuner.Profiler.expert_parallel`
- `AutoTuner.runtime.baseline`

## Dispatch Modes

Run the auto dispatcher:

```bash
python -m AutoTuner.algorithm.main \
  --model-name Qwen/Qwen3-0.6B \
  --prompt-len 4096 \
  --response-len 1024 \
  --nprocs-per-node 8
```

Force a specific algorithm:

```bash
python -m AutoTuner.algorithm.main \
  --algorithm dense \
  --model-name Qwen/Qwen3-0.6B \
  --prompt-len 4096 \
  --response-len 1024 \
  --nprocs-per-node 8
```

```bash
python -m AutoTuner.algorithm.main \
  --algorithm moe \
  --model-name Qwen/Qwen3-30B-A3B \
  --prompt-len 4096 \
  --response-len 1024 \
  --nprocs-per-node 16
```

Auto mode uses HuggingFace config metadata to decide whether the model is dense or MoE.

It treats a model as MoE when one of these conditions is true:

- `model_type` contains `moe`, `mixtral`, `deepseekv3`, or `deepseek_v3`
- any value in `architectures` contains one of those keywords
- the config exposes an expert-count field such as `num_moe_experts` or `num_experts` with value greater than `1`

`--model-name` can be:

- a HuggingFace repo id
- a local HuggingFace model directory
- a local `config.json` path

If `HUGGINGFACE_MODEL_DIR` is set, the dispatcher resolves `--model-name` relative to that directory first.

## JSON Config Input

`main.py` supports `--config <path.json>`. The JSON file can be:

- an object whose keys are expanded into CLI flags
- an array of raw CLI arguments

Object form example:

```json
{
  "model_name": "Qwen/Qwen3-0.6B",
  "prompt_len": 4096,
  "response_len": 1024,
  "nprocs_per_node": 8,
  "output_dir": "outputs/algorithm/functional_auto",
  "max_tp_size": 8,
  "tp_max_token_len": 8192
}
```

Expansion rules:

- `snake_case` keys become `--kebab-case`
- `true` booleans become a flag
- `false` and `null` are omitted
- arrays are emitted as repeated argument values after one flag
- extra raw arguments can be appended through `argv` or `extra_args`

Reference configs:

- `tests/functional_test/algorithm/algorithm_config_auto.json`
- `tests/functional_test/algorithm/algorithm_config_dense.json`
- `tests/functional_test/algorithm/algorithm_config_moe.json`

## Shared Step 1 And Step 2

Both dense and MoE tuning begin with the same first two steps.

### Step 1: Read Or Generate TP/CP Overlap Reports

Step 1 consumes:

- TP overlap report: `tp_overlap/tuning_report.json`
- CP overlap report: `cp_overlap/cp_overlap_report.json`

If `--skip-tp-profiling` or `--skip-cp-profiling` is not set, the algorithm runs the corresponding profiler and writes those reports first.

Selection rules:

- Best TP task is the analysis with the largest `total_overlap_ratio`
- TP tie-breakers prefer lower `operator_e2e_time_us`, then larger `tp_size`, then lexicographically larger `config_id`
- Selected TP size comes from `tp_scaling.optimal_tp_size` when available, otherwise from the best TP task
- Best CP task is the successful case with the largest `analysis.overlap_ratio`
- CP tie-breakers prefer larger `max_token_len`, then larger `seqlen`, then lexicographically larger `case_id`

Step 1 returns:

- selected TP size
- best TP task summary
- best CP task summary
- best overall task summary

### Step 2: Decide TP/CP/max_token_len

Inputs:

- `seqlen = prompt_len + response_len`
- selected TP size from step 1
- step-1 CP `max_token_len`
- `nprocs_per_node`
- optional `cp_candidates`

Rules:

- `required_cp = ceil(seqlen / step1_max_token_len)`
- valid `CP` must satisfy `CP < nprocs_per_node / TP`
- if `cp_candidates` is omitted, candidates default to all positive integers up to the maximum valid CP
- the selected `CP` is the smallest candidate that is both valid and at least `required_cp`
- `max_token_len = CP * step1_max_token_len`

This makes step 2 conservative: it chooses the smallest CP that can cover the requested sequence length under the TP constraint.

## Dense Tuning Flow

Dense tuning runs four steps.

### Step 3: Activation Strategy

Dense step 3 calls `AutoTuner.Profiler.activations` with the step-2 `TP`, `CP`, `seqlen`, and `max_token_len`.

It produces:

- `activation_profile.json`
- `activation_strategy_report.json`

The algorithm records:

- measured TransformerLayer compute time
- activation bytes by part
- modules selected for activation offload
- recompute mode: `none`, `selective`, or `full`
- recompute modules and reason

### Step 4: Pipeline Parallel Size

Dense step 4 runs `AutoTuner.runtime.baseline` with:

- `TP` and `CP` from step 2
- `PP=1`
- the recompute decision from step 3

It then reads the generated `runtime_baseline.json` and derives:

- `pipeline_model_parallel_size`
- per-stage layer capacity
- layer distribution
- per-layer parameter and activation memory estimates

Current assumptions in code:

- the runtime baseline must correspond to `PP=1`
- decoder parameter bytes and activation bytes are scaled linearly per layer
- non-decoder parameter overhead is assigned to the final post-process stage
- `runtime_memory_target_fraction` reserves memory headroom before choosing `PP`

Dense outputs:

- `dense_tuning_report.json`
- `summary.txt`

Typical dense output tree:

```text
<output_dir>/
  dense_tuning_report.json
  summary.txt
  tp_overlap/
    tuning_report.json
  cp_overlap/
    cp_overlap_report.json
  activation_strategy/
    activation_profile.json
    activation_strategy_report.json
  runtime_baseline/
    .../runtime_baseline.json
```

## MoE Tuning Flow

MoE tuning runs three steps.

### Step 3: Expert Parallel Strategy

MoE step 3 calls `AutoTuner.Profiler.expert_parallel` with the step-2 `TP`, `CP`, `seqlen`, and `max_token_len`.

It produces:

- `expert_parallel_profile.json`
- `expert_parallel_strategy_report.json`

The algorithm records:

- selected `EP`
- local expert count
- chosen bandwidth tier
- whether dispatch/combine communication can be fully overlapped
- exposed communication time and objective time
- all candidate evaluations

Current assumptions in code:

- overlap windows come from profiled attention and TEGroupedMLP timings
- the decision prefers the lowest objective among candidates whose communication can be fully overlapped

MoE outputs:

- `moe_tuning_report.json`
- `summary.txt`

Typical MoE output tree:

```text
<output_dir>/
  moe_tuning_report.json
  summary.txt
  tp_overlap/
    tuning_report.json
  cp_overlap/
    cp_overlap_report.json
  expert_parallel/
    expert_parallel_profile.json
    expert_parallel_strategy_report.json
```

## Reusing Existing Reports

Each stage can reuse an existing profiler output instead of rerunning profiling.

Common reuse flags:

- `--skip-tp-profiling`
- `--skip-cp-profiling`

Dense-specific reuse flags:

- `--skip-activation-profiling`
- `--skip-runtime-profiling`

MoE-specific reuse flags:

- `--skip-ep-profiling`

When a skip flag is set, the corresponding report must already exist under the configured output directory, otherwise the algorithm raises `FileNotFoundError`.

This pattern is used by the unit tests in:

- `tests/unit_test/algorithm/test_dense_tuning.py`
- `tests/unit_test/algorithm/test_moe_tuning.py`

## CLI Surfaces

Dense CLI:

```bash
python -m AutoTuner.algorithm.dense_tuning --help
```

MoE CLI:

```bash
python -m AutoTuner.algorithm.moe_tuning --help
```

Auto dispatcher help:

```bash
python -m AutoTuner.algorithm.main --help
```

The functional smoke script is:

```bash
tests/functional_test/algorithm/algorithm_run.sh
```

## Programmatic Use

The package exports lazy imports through `AutoTuner.algorithm`:

```python
from AutoTuner.algorithm import (
    DenseTuningAlgorithm,
    DenseTuningConfig,
    MoETuningAlgorithm,
    MoETuningConfig,
)
```

Example:

```python
from AutoTuner.algorithm import DenseTuningAlgorithm, DenseTuningConfig

config = DenseTuningConfig(
    model_name="Qwen/Qwen3-0.6B",
    prompt_len=4096,
    response_len=1024,
    nprocs_per_node=8,
)

report = DenseTuningAlgorithm(config).run()
print(report.step2.tp_size, report.step2.cp_size, report.step4.pipeline_model_parallel_size)
```

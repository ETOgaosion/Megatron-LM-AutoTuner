# TP Overlap Tuner

Auto-tuning framework for TP (Tensor Parallel) communication/computation overlap configurations in RLHF training with Megatron-LM.

## Overview

The TP Overlap Tuner automatically finds optimal configurations for overlapping GEMM computations with TP collective communications (all-gather, reduce-scatter). It profiles different overlap methods and SM allocations to maximize throughput.

### Workflow

The tuner follows this workflow:

```
0. [INPUT] User input a model config
1. Generate test cases using binary search strategy
2. Run the configs with torch profiler
3. Analyze torch profiler JSON traces
4. Generate report for each operator with optimal configurations
```

### Supported Operators

| Operator | Linear Type | Communication |
|----------|-------------|---------------|
| `qkv` | ColumnParallelLinear | All-Gather |
| `proj` | RowParallelLinear | Reduce-Scatter |
| `fc1` | ColumnParallelLinear | All-Gather |
| `fc2` | RowParallelLinear | Reduce-Scatter |

### Overlap Methods

- **ring_exchange**: Ring-based all-gather/reduce-scatter. Avoids occupying normal computation SM. Best for forward pass (fprop) and some backward passes (dgrad).
- **bulk**: Bulk collective operations with configurable SM count. Binary search finds optimal num_sm. Best for backward passes (dgrad, wgrad).

## Quick Start

### Using Shell Script (Recommended)

```bash
# 1. Copy and configure test environment
cp tests/functional_test/overlap/test_env_sample.sh tests/functional_test/overlap/test_env.sh
# Edit test_env.sh to set MODEL_NAME

# 2. Run the complete tuning workflow
bash tests/functional_test/overlap/tp_overlap_tuner.sh
```

### Using Python CLI

```bash
# Run complete tuning workflow (model params auto-fetched from HuggingFace)
python -m AutoTuner.Profiler.overlap.main --model-name Qwen/Qwen3-0.6B

# Tune with specific TP size limit
python -m AutoTuner.Profiler.overlap.main --model-name meta-llama/Llama-2-7b --max-tp-size 4

# Tune specific operators only
python -m AutoTuner.Profiler.overlap.main --model-name Qwen/Qwen3-0.6B --operators qkv proj
```

### Using Python API

```python
from AutoTuner.Profiler.overlap import TPOverlapTuner, TPOverlapTunerConfig

# Configure the tuner (model params auto-fetched from model_name)
config = TPOverlapTunerConfig(
    model_name="Qwen/Qwen3-0.6B",  # hidden_size, ffn_hidden_size, etc. auto-fetched
    max_tp_size=8,
    max_token_len=8192,
    operators=["fc1", "fc2", "qkv", "proj"],
    output_dir="outputs/tp_overlap_tuner/my_run",
)

# Model parameters are automatically fetched
print(f"Hidden Size: {config.hidden_size}")
print(f"FFN Hidden Size: {config.ffn_hidden_size}")

# Run the complete tuning workflow
tuner = TPOverlapTuner(config, overlap_threshold=0.5)
report = tuner.run()

# Access results
for rec in report.recommendations:
    print(rec)
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | (required) | Model name from HuggingFace. Model parameters are automatically fetched. |
| `--max-tp-size` | `8` | Maximum TP size to test (tests 2, 4, 8 up to this value) |
| `--max-token-len` | `8192` | Maximum token length. Use high value for peak computational intensity. |
| `--operators` | `fc1 fc2 qkv proj` | Operators to tune |
| `--min-num-sm` | `1` | Minimum num_sm for bulk method binary search |
| `--max-num-sm` | `16` | Maximum num_sm for bulk method binary search |
| `--overlap-threshold` | `0.5` | Minimum overlap ratio to consider effective |
| `--output-dir` | auto-generated | Output directory for results |

## Output Files

After tuning completes, the following files are generated:

```
outputs/tp_overlap_tuner/<timestamp>/
├── traces/                              # Profiler traces per config
│   ├── tp2_fc1_fprop_ring_agg0/
│   │   ├── tp_comm_overlap_cfg.yaml
│   │   └── *.pt.trace.json
│   ├── tp2_fc1_dgrad_bulk_sm2/
│   └── ...
├── tuning_report.json                   # Full analysis in JSON format
├── summary.txt                          # Human-readable summary
├── optimal_tp_comm_overlap_cfg.yaml     # Best config (smallest TP)
├── optimal_tp_comm_overlap_cfg_tp2.yaml # Best config for TP=2
├── optimal_tp_comm_overlap_cfg_tp4.yaml # Best config for TP=4
└── optimal_tp_comm_overlap_cfg_tp8.yaml # Best config for TP=8
```

### Output YAML Format

The generated `optimal_tp_comm_overlap_cfg.yaml` can be used directly with Megatron-LM:

```yaml
qkv_fprop:
  method: ring_exchange
  aggregate: 1
proj_fprop:
  method: ring_exchange
  aggregate: 1
fc1_dgrad:
  method: bulk
  num_sm: 4
  set_sm_margin: 0
# ... more configs
```

To use the optimal config:
```bash
cp outputs/tp_overlap_tuner/<timestamp>/optimal_tp_comm_overlap_cfg.yaml \
   AutoTuner/testbench/profile/configs/local/tp_comm_overlap_cfg.yaml
```

### JSON Report Format

The `tuning_report.json` includes detailed metrics for each configuration:

```json
{
  "config_id": "tp2_fc1_fprop_ring_agg0",
  "operator": "fc1",
  "phase": "fprop",
  "tp_size": 2,
  "overlap_method": "ring_exchange",
  "forward_gemm_time_us": 1234.5,
  "forward_comm_time_us": 567.8,
  "forward_overlap_time_us": 400.0,
  "forward_overlap_ratio": 0.70,
  "forward_e2e_time_us": 1402.3,
  "backward_gemm_time_us": 2345.6,
  "backward_comm_time_us": 890.1,
  "backward_overlap_time_us": 700.0,
  "backward_overlap_ratio": 0.78,
  "backward_e2e_time_us": 2535.7,
  "operator_e2e_time_us": 3938.0,
  "total_overlap_ratio": 0.75,
  "num_gemm_events": 4,
  "num_comm_events": 2
}
```

Key metrics:
- `*_e2e_time_us`: End-to-end execution time (wall clock from first to last event)
- `operator_e2e_time_us`: Total Linear operator execution time
- `*_overlap_ratio`: Ratio of overlap time to min(GEMM, comm) time

The JSON report also includes TP scaling efficiency analysis with TP=1 as baseline:

```json
{
  "tp_scaling": {
    "optimal_tp_size": 4,
    "tolerance": 0.2,
    "results": {
      "fc1": {
        "optimal_tp_size": 4,
        "scaling_efficient": {"1": true, "2": true, "4": true, "8": false},
        "scaling_ratios": {"1": 1.0, "2": 1.1, "4": 1.2, "8": 1.6},
        "e2e_times": {"1": 2000.0, "2": 1100.0, "4": 600.0, "8": 400.0},
        "reason": "Baseline: TP=1 (no TP) @ 2000.0us. TP=2: EFFICIENT; TP=4: EFFICIENT; TP=8: NOT efficient"
      }
    }
  }
}
```

## Workflow Details

### Step 0: [INPUT] Model Configuration

Model parameters are automatically fetched from HuggingFace:
- `hidden_size`
- `ffn_hidden_size` (intermediate_size)
- `num_attention_heads`
- `num_kv_heads` (num_key_value_heads)

### Step 1: Generate Test Cases

Uses binary search strategy for `num_sm` parameter:
- Tests TP=2 first to see if overlap is beneficial
- Adjusts TP size (2, 4, 8) to vary computational pressure
- For ring_exchange: tests aggregate=0 and aggregate=1
- For bulk: tests num_sm in {1, 2, 4, 8, 16}

Total ~288 configurations (3 TP sizes × 4 operators × ~24 configs/operator)

### Step 2: Run Profiling

For each configuration:
- Generates `tp_comm_overlap_cfg.yaml`
- Runs torch profiler with `--run-one-data` flag
- Uses high `max_token_len` for peak computational intensity

### Step 3: Analyze Traces

Parses torch profiler JSON to detect:
- GEMM kernels: `cutlass`, `cublas`, `sm*_xmma`, etc.
- Communication kernels: `userbuffers`, `Memcpy PtoP`, `kuserbuffers`, `nccl`

Calculates overlap ratio: `overlap_time / min(gemm_time, comm_time)`

**TP Scaling Efficiency Check:**

The tuner uses TP=1 (no tensor parallelism) as the baseline for comparison:
- TP=1 represents pure computation without communication overhead
- Rule: If using TP=n, then `Time(TP=n)` should be ≈ `Time(TP=1) / n`
- Tolerance: 20% (configurable)
- If scaling is not efficient, the smaller TP size (or no TP) is recommended

Example:
- TP=1 (baseline): 2000us (pure computation, no comm)
- TP=2: Expected 1000us (1/2 of TP=1), Actual 1100us → ratio=1.1 → EFFICIENT
- TP=4: Expected 500us (1/4 of TP=1), Actual 600us → ratio=1.2 → EFFICIENT (within 20%)
- TP=8: Expected 250us (1/8 of TP=1), Actual 400us → ratio=1.6 → NOT efficient (>20%)
- Result: Use TP=4 as optimal (TP=8 has too much communication overhead)

### Step 4: Generate Report

Produces:
- Best config per operator/phase (highest overlap ratio)
- Recommendations for effective overlap (>50% ratio)
- **Optimal TP size based on scaling efficiency analysis**
- Optimal YAML configs per TP size

## Architecture

```
AutoTuner/Profiler/overlap/
├── __init__.py           # Package exports
├── main.py               # CLI entry point (complete workflow)
├── tuner.py              # Main orchestrator (TPOverlapTuner)
├── config_generator.py   # Test configuration generation
├── trace_analyzer.py     # Torch profiler JSON parsing
├── overlap_detector.py   # Overlap calculation
├── report_generator.py   # Report & YAML generation
└── test_trace_analyzer.py # Unit tests
```

## Requirements

- Python >= 3.10
- PyTorch with CUDA support
- TransformerEngine with userbuffers support
- NCCL
- HuggingFace transformers (for auto-fetching model config)

### Environment Variables

Set automatically by the shell script:

```bash
export UB_SKIPMC=1          # Enable userbuffers
export NVTE_FLASH_ATTN=1    # Use flash attention
export NVTE_FUSED_ATTN=0    # Disable fused attention
```

## Constraints

- TP only happens in a single machine (8 GPUs max, TP < 8)
- Pipeline method not used for reduce-scatter (only ring_exchange and bulk)

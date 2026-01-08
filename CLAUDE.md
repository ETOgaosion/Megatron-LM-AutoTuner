# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Megatron-LM-AutoTuner is an automated performance tuning framework for LLM pre-training and post-training, targeted at frameworks like verl. It focuses on optimizing MFU (Model FLOPs Utilization) in Megatron-Core training processes for both forward-only and forward-backward-update models, leading to high performance in RLHF scenarios.

## Project Structure

This is a multi-repository project with submodules:

- **AutoTuner/**: Main auto-tuner implementation
  - `testbench/`: Profiling and testing infrastructure
    - `ops/`: Operator implementations (attention, linear layers, MLP, MoE, etc.)
    - `ops_test/`: Test classes for operators with theoretical calculation support
    - `profile/`: Profiling launcher and configuration system
    - `functional/`: CPU-GPU movement tests and interference analysis
  - `utils/`: Utilities for distributed training, logging, memory tracking, timing, NVTX
  - `Profiler/`: Profiler components for communicators, operators, and overlap analysis

- **Megatron-LM/**: Enhanced Megatron-LM submodule with AutoTuner integration and NetworkEngine
- **TransformerEngine/**: Enhanced TE submodule with NVSHMEM-based async Context Parallel
- **verl/**: Enhanced verl submodule with balanced data resharding and scalable train-infer transport

- **tests/functional_test/**: Functional test scripts
- **runtime/megatron/**: Runtime configurations
- **scripts/**: Installation, Docker, and utility scripts
- **docs/**: Documentation including install guides, quick start, design docs

## Environment Setup

### Installation Commands

```bash
# Clone with submodules
git clone git@github.com:ETOgaosion/Megatron-LM-AutoTuner.git --recurse-submodules

# Or update existing clone
git submodule update --init --recursive

# Install in development mode (after setting up conda environment)
pip install -e .
```

### Environment Configuration

Before running tests or profiling:

1. Copy `.secrets/env_sample.sh` to `.secrets/env.sh` and configure your environment
2. Copy `tests/functional_test/test_env_sample.sh` to `tests/functional_test/test_env.sh` for testing config
3. Create local config files in `AutoTuner/testbench/profile/configs/local/`:
   - `override_model_config.json`
   - `override_tf_config.json`
   - `tp_comm_overlap_cfg.yaml` (if using TP communication overlap)

## Common Development Commands

### Profiling and Data Collection

```bash
# Collect profiling data (profile mode 0)
bash tests/functional_test/testbench_collect_data.sh

# Run torch profiler
bash tests/functional_test/testbench_torch_profile.sh

# Visualize torch profiler results
pip install torch_tb_profiler
tensorboard --logdir=outputs/[TIMESTAMP]/[MODEL]/torch_profiler --host 0.0.0.0
# Open browser to http://[ip]:6006/#pytorch_profiler

# Run nsys profiler (requires Docker with root privileges)
bash tests/functional_test/testbench_nsys_profile.sh

# Run memory snapshot
bash tests/functional_test/testbench_torch_memory_snapshot.sh
```

### Docker Environment

```bash
# Pull Docker image (with robust retry mechanism)
bash scripts/docker_robust_pull.sh whatcanyousee/megatron-autotuner-env:mcore0.15.1_te2.7

# Start Docker container
bash scripts/start_docker.sh

# Install inside Docker
bash scripts/install_in_docker.sh
```

### Code Formatting

```bash
# Format code with black and isort
bash scripts/format.sh
```

## Architecture and Key Concepts

### Tuning Dimensions

The AutoTuner optimizes across multiple parallelization and configuration dimensions:

- **Dense Layer Parallelism**: TP (tensor parallel), CP (context parallel), DP (data parallel), PP (pipeline parallel), VPP (virtual pipeline parallel)
- **MoE Parallelism**: ETP (expert tensor parallel), EP (expert parallel), EDP (expert data parallel)
- **Pipeline Layout**: Configuration of pipeline stages
- **Sequence Length**: `max_token_len` parameter
- **Recompute**: `recompute_granularity`, `recompute_method`, `recompute_num_layers`, `recompute_modules`

### Operator Testing Framework

The testbench uses a three-tier architecture:

1. **Op Classes**: Contain operator logic (e.g., `Decoder`, `SelfAttention`, `MLPDense`)
2. **TestOp Classes**: Inherit from `TestWithHiddenInputs` and implement `prepare_input()` to generate test inputs
3. **Launcher Classes**: Execute tests by calling TestOp methods and running forward passes

Key base class: `TestWithHiddenInputs` automatically provides `HiddenStatus` inputs for testing operators.

### Profiling Modes

- **Mode 0**: Data collection mode - generates theoretical and real profiling data
- Theoretical metrics: weights, activations, FLOPS (optional)
- Output format: JSON files with `{"real": "...", "estimated": "..."}` structure

### Data Collection Options

```bash
# Enable/disable theoretical estimations
--theoretical-flops true          # Enable FLOPS estimation (default: disabled)
--theoretical-activations false   # Disable activation estimation (default: enabled)
# Weights estimation is always enabled
```

### Profiling Output Structure

```
outputs/
  └── [TIMESTAMP]/
      └── [MODEL_NAME]/
          ├── collect_data/
          │   ├── rank_0/
          │   ├── rank_1/
          │   └── ...
          ├── torch_profiler/
          │   └── *.pt.trace.json
          ├── nsys_profile/
          │   └── nsight_report.nsys-rep
          └── memory_snapshot/
```

## Important Implementation Details

### Parallel Configuration

Test environment variables in `test_env.sh`:
- `TP_SIZE`: Tensor parallel size
- `CP_SIZE`: Context parallel size
- `EP_SIZE`: Expert parallel size
- `ETP_SIZE`: Expert tensor parallel size
- `TEST_OPS_LIST`: Array of operators to test (e.g., `("Decoder" "SelfAttention")`)
- `TEST_CASE_IDXES`: Specific test case indices to run
- `TP_COMM_OVERLAP`: Enable TP communication overlap

### TP Communication Overlap

When enabling TP overlap:
```bash
export UB_SKIPMC=1  # Set automatically by test scripts
```
Must also enable in `AutoTuner/testbench/profile/configs/override_tf_config.json`

### Flash Attention Configuration

```bash
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
```

### Multi-Rank Profiling

Use `tools/merge_profiler_traces.py` to merge PyTorch profiler traces from multiple ranks:

```bash
python3 tools/merge_profiler_traces.py -i outputs/[TIMESTAMP]/[MODEL]/torch_profiler
# Upload merged JSON to https://ui.perfetto.dev/ for visualization
```

## Testing Configuration

The profiling system uses JSON configuration files for model and test case definitions:
- Test cases defined in `AutoTuner/testbench/profile/configs/[model].json`
- Model overrides in `local/override_model_config.json`
- Transformer config overrides in `local/override_tf_config.json`

## Dependencies and Versions

Core dependencies (see `requirements_now.txt` and `requirements_dev.txt`):
- Python >= 3.10
- PyTorch 2.8.0+cu128 (for CUDA 12.8 / RTX 5090 support)
- flash_attn 2.8.1+ (local) / 2.7.4 (Docker image)
- Megatron-LM (submodule, dev branch)
- TransformerEngine (submodule, dev branch, with flash_attention patch)
- verl (submodule, dev branch)
- apex

## Build System

Uses both `pyproject.toml` (primary) and `setup.py` (fallback):
- Package name: `Megatron-LM-AutoTuner`
- Includes: `AutoTuner` and `AutoTuner.*` packages
- Package data: `testbench/profile/configs/*.json`

## Git Submodules

The project tracks enhanced versions of upstream repositories. Always ensure submodules are up to date:
```bash
git submodule update --remote
```

Modified/enhanced by project contributors:
- **Megatron-LM-Enhanced**: NetworkEngine, AutoTuner integration
- **TransformerEngine-Enhanced**: NVSHMEM async CP transport
- **verl-enhanced**: Balanced data resharding, scalable train-infer transport

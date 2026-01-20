#!/bin/bash
#
# Profile a single TP overlap configuration
#
# This script profiles a single TP overlap configuration using torch profiler.
# It is called by the Python tuner for each configuration to test.
#
# Usage:
#   bash scripts/tp_overlap_tuner/profile_single_config.sh
#
# Required environment variables:
#   TP_SIZE           - Tensor parallel size (2, 4, or 8)
#   OPERATOR          - Operator name (fc1, fc2, qkv, proj)
#   TEST_CLASS        - Test class name (TEColumnParallelLinear or TERowParallelLinear)
#   OUTPUT_DIR        - Output directory for trace files
#   YAML_CONFIG       - Path to tp_comm_overlap_cfg.yaml
#   MODEL_NAME        - Model name from HuggingFace
#   TEST_CASES_DIR    - Directory containing test cases file
#   TEST_CASES_FILE   - Test cases JSON file name
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source environment if available
if [ -f "$PROJECT_ROOT/.secrets/env.sh" ]; then
    source "$PROJECT_ROOT/.secrets/env.sh"
fi

# Validate required environment variables
if [ -z "$TP_SIZE" ]; then
    echo "Error: TP_SIZE is required"
    exit 1
fi

if [ -z "$OPERATOR" ]; then
    echo "Error: OPERATOR is required"
    exit 1
fi

if [ -z "$TEST_CLASS" ]; then
    echo "Error: TEST_CLASS is required"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: OUTPUT_DIR is required"
    exit 1
fi

if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME="Qwen/Qwen3-0.6B"
fi

if [ -z "$TEST_CASES_DIR" ]; then
    echo "Error: TEST_CASES_DIR is required"
    exit 1
fi

if [ -z "$TEST_CASES_FILE" ]; then
    echo "Error: TEST_CASES_FILE is required"
    exit 1
fi

# Set required environment variables for TP overlap
export UB_SKIPMC=1
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export NVTE_NVTX_ENABLED=1

# Copy YAML config to expected location if provided
if [ -n "$YAML_CONFIG" ] && [ -f "$YAML_CONFIG" ]; then
    CONFIG_DIR="AutoTuner/testbench/profile/configs/local"
    mkdir -p "$CONFIG_DIR"
    cp "$YAML_CONFIG" "$CONFIG_DIR/tp_comm_overlap_cfg.yaml"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Profiling configuration:"
echo "  TP Size:    $TP_SIZE"
echo "  Operator:   $OPERATOR"
echo "  Test Class: $TEST_CLASS"
echo "  Model:      $MODEL_NAME"
echo "  Output:     $OUTPUT_DIR"

# Run profiling with torchrun
cd "$PROJECT_ROOT"

torchrun --nproc_per_node="${TP_SIZE}" \
    -m AutoTuner.testbench.profile.main \
    --model-name "${MODEL_NAME}" \
    --test-cases-dir "${TEST_CASES_DIR}" \
    --test-cases-file "${TEST_CASES_FILE}" \
    --profile-mode 2 \
    --test-ops-list "${TEST_CLASS}" \
    --tp-comm-buffer-name "${OPERATOR}" \
    --run-one-data \
    --tensor-model-parallel-size "${TP_SIZE}" \
    --output-dir "${OUTPUT_DIR}"

# Check if trace file was generated
TRACE_FILES=$(find "$OUTPUT_DIR" -name "*.pt.trace.json" 2>/dev/null | head -1)
if [ -n "$TRACE_FILES" ]; then
    echo "Profiling completed successfully!"
    echo "Trace file: $TRACE_FILES"
else
    echo "Warning: No trace file generated"
    exit 1
fi

#!/bin/bash
#
# TP Overlap Tuner - Complete Tuning Workflow
#
# This script runs the complete TP overlap tuning workflow:
#   0. [INPUT] Load model configuration
#   1. Generate test cases using binary search strategy
#   2. Run configs with torch profiler
#   3. Analyze torch profiler JSON traces
#   4. Generate report for each operator
#
# Usage:
#   bash tests/functional_test/overlap/tp_overlap_tuner.sh
#
# Before running:
#   1. Copy test_env_sample.sh to test_env.sh
#   2. Modify test_env.sh for your model name
#

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Source environment
if [ -f "$PROJECT_ROOT/.secrets/env.sh" ]; then
    source "$PROJECT_ROOT/.secrets/env.sh"
fi

# Source test environment
if [ -f "$SCRIPT_DIR/test_env.sh" ]; then
    source "$SCRIPT_DIR/test_env.sh"
else
    echo "Warning: test_env.sh not found. Using default settings."
    echo "Copy test_env_sample.sh to test_env.sh and modify for your setup."
    echo ""
    MODEL_NAME="Qwen/Qwen3-0.6B"
    MAX_TP_SIZE=8
    MAX_TOKEN_LEN=8192
    OPERATORS="fc1 fc2 qkv proj"
    OVERLAP_THRESHOLD=0.5
    MIN_NUM_SM=1
    MAX_NUM_SM=16
    OUTPUT_DIR=""
fi

# Generate output directory with timestamp if not provided
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="outputs/tp_overlap_tuner/${TIMESTAMP}"
fi

# Set environment variables for TP overlap
export UB_SKIPMC=1
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Print header
echo "============================================================"
echo "TP Overlap Tuner - Complete Workflow"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model:              ${MODEL_NAME}"
echo "  Max TP Size:        ${MAX_TP_SIZE}"
echo "  Max Token Len:      ${MAX_TOKEN_LEN}"
echo "  Operators:          ${OPERATORS}"
echo "  Overlap Threshold:  ${OVERLAP_THRESHOLD}"
echo "  Num SM Range:       ${MIN_NUM_SM} - ${MAX_NUM_SM}"
echo "  Output Dir:         ${OUTPUT_DIR}"
echo ""
echo "Workflow:"
echo "  0. [INPUT] Load model config from HuggingFace"
echo "  1. Generate test cases (binary search for num_sm)"
echo "  2. Run configs with torch profiler"
echo "  3. Analyze torch profiler JSON traces"
echo "  4. Generate report for each operator"
echo "============================================================"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Run the complete tuning workflow
python -m AutoTuner.Profiler.overlap.main \
    --model-name "${MODEL_NAME}" \
    --max-tp-size "${MAX_TP_SIZE}" \
    --max-token-len "${MAX_TOKEN_LEN}" \
    --operators ${OPERATORS} \
    --overlap-threshold "${OVERLAP_THRESHOLD}" \
    --min-num-sm "${MIN_NUM_SM}" \
    --max-num-sm "${MAX_NUM_SM}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "Tuning completed!"
echo "============================================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
echo "  - ${OUTPUT_DIR}/tuning_report.json"
echo "  - ${OUTPUT_DIR}/summary.txt"
echo "  - ${OUTPUT_DIR}/optimal_tp_comm_overlap_cfg.yaml"
echo ""
echo "To use the optimal config, copy it to:"
echo "  AutoTuner/testbench/profile/configs/local/tp_comm_overlap_cfg.yaml"
echo "============================================================"

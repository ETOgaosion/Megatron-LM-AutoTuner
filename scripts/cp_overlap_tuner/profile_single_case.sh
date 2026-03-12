#!/bin/bash
#
# Profile a single CP overlap case with TEAttenWithCPEnhanced and CP=2.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$PROJECT_ROOT/.secrets/env.sh" ]; then
    source "$PROJECT_ROOT/.secrets/env.sh"
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: OUTPUT_DIR is required"
    exit 1
fi

if [ -z "$TEST_CASES_DIR" ]; then
    echo "Error: TEST_CASES_DIR is required"
    exit 1
fi

if [ -z "$TEST_CASES_FILE" ]; then
    echo "Error: TEST_CASES_FILE is required"
    exit 1
fi

if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME="Qwen/Qwen3-0.6B"
fi

if [ -z "$CP_SIZE" ]; then
    CP_SIZE="2"
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export NVTE_NVTX_ENABLED=1

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"

torchrun --nproc_per_node="${CP_SIZE}" \
    -m AutoTuner.testbench.profile.main \
    --model-name "${MODEL_NAME}" \
    --test-cases-dir "${TEST_CASES_DIR}" \
    --test-cases-file "${TEST_CASES_FILE}" \
    --profile-mode 2 \
    --test-ops-list "TEAttenWithCPEnhanced" \
    --run-one-data \
    --tensor-model-parallel-size "1" \
    --context-parallel-size "${CP_SIZE}" \
    --output-dir "${OUTPUT_DIR}"

TRACE_FILES=$(find "$OUTPUT_DIR" -name "*.pt.trace.json" 2>/dev/null | head -1)
if [ -n "$TRACE_FILES" ]; then
    echo "Profiling completed successfully!"
    echo "Trace file: $TRACE_FILES"
else
    echo "Warning: No trace file generated"
    exit 1
fi

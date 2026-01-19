#!/bin/bash
#
# TP Overlap Tuner - Main orchestration script
#
# This script runs the TP Overlap Tuner to find optimal communication/computation
# overlap configurations for RLHF training.
#
# Usage:
#   bash scripts/tp_overlap_tuner/run_tuner.sh
#
# Environment variables (optional):
#   MODEL_NAME      - Model name from HuggingFace (default: Qwen/Qwen3-0.6B)
#   MAX_TP_SIZE     - Maximum TP size to test (default: 8)
#   OPERATORS       - Space-separated list of operators (default: "fc1 fc2 qkv proj")
#   OUTPUT_DIR      - Output directory (default: auto-generated with timestamp)
#   SKIP_PROFILING  - Set to "true" to skip profiling and analyze existing traces
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source environment if available
if [ -f "$PROJECT_ROOT/.secrets/env.sh" ]; then
    source "$PROJECT_ROOT/.secrets/env.sh"
fi

# Default configurations
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MAX_TP_SIZE="${MAX_TP_SIZE:-8}"
OPERATORS="${OPERATORS:-fc1 fc2 qkv proj}"
OVERLAP_THRESHOLD="${OVERLAP_THRESHOLD:-0.5}"

# Generate output directory with timestamp if not provided
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="outputs/tp_overlap_tuner/${TIMESTAMP}"
fi

# Check if we should skip profiling
SKIP_FLAG=""
if [ "${SKIP_PROFILING}" = "true" ]; then
    SKIP_FLAG="--skip-profiling"
fi

echo "============================================================"
echo "TP Overlap Tuner"
echo "============================================================"
echo "Model:          ${MODEL_NAME}"
echo "Max TP Size:    ${MAX_TP_SIZE}"
echo "Operators:      ${OPERATORS}"
echo "Output Dir:     ${OUTPUT_DIR}"
echo "Skip Profiling: ${SKIP_PROFILING:-false}"
echo "============================================================"
echo ""

# Run the tuner
cd "$PROJECT_ROOT"

python -m AutoTuner.Profiler.overlap.tuner \
    --model-name "${MODEL_NAME}" \
    --max-tp-size "${MAX_TP_SIZE}" \
    --operators ${OPERATORS} \
    --overlap-threshold "${OVERLAP_THRESHOLD}" \
    --output-dir "${OUTPUT_DIR}" \
    ${SKIP_FLAG}

echo ""
echo "============================================================"
echo "Tuning completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"

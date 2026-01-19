#!/bin/bash
#
# Analyze existing TP overlap profiling results
#
# This script analyzes previously collected torch profiler traces
# without re-running the profiling step.
#
# Usage:
#   bash scripts/tp_overlap_tuner/analyze_results.sh [OUTPUT_DIR]
#
# Arguments:
#   OUTPUT_DIR - Directory containing the traces (default: latest in outputs/tp_overlap_tuner/)
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source environment if available
if [ -f "$PROJECT_ROOT/.secrets/env.sh" ]; then
    source "$PROJECT_ROOT/.secrets/env.sh"
fi

# Determine output directory
if [ -n "$1" ]; then
    OUTPUT_DIR="$1"
else
    # Find latest output directory
    TUNER_OUTPUT="$PROJECT_ROOT/outputs/tp_overlap_tuner"
    if [ -d "$TUNER_OUTPUT" ]; then
        OUTPUT_DIR=$(ls -dt "$TUNER_OUTPUT"/*/ 2>/dev/null | head -1)
        if [ -z "$OUTPUT_DIR" ]; then
            echo "Error: No output directories found in $TUNER_OUTPUT"
            exit 1
        fi
        OUTPUT_DIR="${OUTPUT_DIR%/}"  # Remove trailing slash
    else
        echo "Error: No previous tuning results found"
        exit 1
    fi
fi

# Verify output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory does not exist: $OUTPUT_DIR"
    exit 1
fi

echo "============================================================"
echo "Analyzing TP Overlap Results"
echo "============================================================"
echo "Output Dir: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# Run analysis with skip-profiling flag
cd "$PROJECT_ROOT"

python -m AutoTuner.Profiler.overlap.tuner \
    --output-dir "${OUTPUT_DIR}" \
    --skip-profiling

echo ""
echo "============================================================"
echo "Analysis completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
echo ""
echo "Generated files:"
echo "  - ${OUTPUT_DIR}/tuning_report.json"
echo "  - ${OUTPUT_DIR}/summary.txt"
echo "  - ${OUTPUT_DIR}/optimal_tp_comm_overlap_cfg.yaml"

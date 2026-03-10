#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "megatron-lm-autotuner" ]]; then
    echo "Warning: current conda env is '${CONDA_DEFAULT_ENV:-<unset>}'"
    echo "Recommended: conda activate megatron-lm-autotuner"
fi

if [ -f .secrets/env.sh ]; then
    source .secrets/env.sh
fi

if [ -f tests/functional_test/runtime/test_env.sh ]; then
    source tests/functional_test/runtime/test_env.sh
else
    MODEL_NAME="Qwen/Qwen3-0.6B"
    TEST_CASES_FILE="qwen3_0_6b.json"
    NUM_TEST_CASES=1
    MAX_ITERATIONS=4
    WARMUP_ITERATIONS=1
    SHARE_EMB=None
    TP_SIZE=1
    CP_SIZE=1
    EP_SIZE=1
    ETP_SIZE=1
    PP_SIZE=2
    VPP_SIZE=2
fi

GPUS_PER_NODE=$(($TP_SIZE * $CP_SIZE * $EP_SIZE * $ETP_SIZE * $PP_SIZE))

OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-outputs/runtime_simulation}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6010}"
NUM_NODES="${NUM_NODES:-1}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export UB_SKIPMC="${UB_SKIPMC:-1}"
export AUTOTUNER_RUNTIME_USE_FUSED_KERNELS="${AUTOTUNER_RUNTIME_USE_FUSED_KERNELS:-1}"
export AUTOTUNER_BASELINE_DP_ALLREDUCE_BANDWIDTH_GBPS="${AUTOTUNER_BASELINE_DP_ALLREDUCE_BANDWIDTH_GBPS:-50}"
export AUTOTUNER_BASELINE_DP_ALLREDUCE_LATENCY_US="${AUTOTUNER_BASELINE_DP_ALLREDUCE_LATENCY_US:-30}"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
    --context-parallel-size "$CP_SIZE"
    --expert-parallel-size "$EP_SIZE"
    --expert-tensor-parallel-size "$ETP_SIZE"
)

if [[ "${VPP_SIZE}" != "None" ]]; then
    PARALLEL_ARGS+=(--virtual-pipeline-model-parallel-size "$VPP_SIZE")
fi

RUNTIME_ARGS=(
    --model-name "$MODEL_NAME"
    --test-cases-file "$TEST_CASES_FILE"
    --output-dir "$OUTPUT_BASE_DIR"
    --num-test-cases "${NUM_TEST_CASES:-1}"
    --max-iterations "${MAX_ITERATIONS:-4}"
    --warmup-iterations "${WARMUP_ITERATIONS:-1}"
)

if [[ "${SHARE_EMB}" != "None" ]]; then
    RUNTIME_ARGS+=(--share-embeddings-and-output-weights "$SHARE_EMB")
fi

echo "Running runtime baseline with simulator enabled"
echo "  model: $MODEL_NAME"
echo "  parallel: tp=$TP_SIZE cp=$CP_SIZE ep=$EP_SIZE etp=$ETP_SIZE pp=$PP_SIZE vpp=$VPP_SIZE"
echo "  simulator: dp_bw=${AUTOTUNER_BASELINE_DP_ALLREDUCE_BANDWIDTH_GBPS}GB/s dp_latency=${AUTOTUNER_BASELINE_DP_ALLREDUCE_LATENCY_US}us"
echo "  python: $PYTHON_BIN"
echo "  output_dir: $OUTPUT_BASE_DIR"

"$PYTHON_BIN" -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" -m AutoTuner.runtime.baseline.main \
    "${RUNTIME_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}"

LATEST_SUMMARY=$(find "$OUTPUT_BASE_DIR" -path "*/runtime_baseline/runtime_summary.json" | sort | tail -n 1)
if [[ -z "${LATEST_SUMMARY:-}" ]]; then
    echo "No runtime_summary.json found under $OUTPUT_BASE_DIR"
    exit 1
fi

echo
echo "Latest summary: $LATEST_SUMMARY"
"$PYTHON_BIN" - "$LATEST_SUMMARY" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, "r") as fp:
    data = json.load(fp)

for item in data:
    print(
        "test_case_idx={idx} measured_time_s={measured:.4f} simulated_time_s={sim:.4f} "
        "simulated_pp_s={sim_pp:.4f} simulated_dp_s={sim_dp:.4f} "
        "measured_toks={measured_tps:.2f} simulated_toks={sim_tps:.2f}".format(
            idx=item["test_case_idx"],
            measured=item["time_s"],
            sim=item["simulated_time_s"],
            sim_pp=item["simulated_pp_compute_time_s"],
            sim_dp=item["simulated_dp_allreduce_time_s"],
            measured_tps=item["throughput_tokens_s"],
            sim_tps=item["simulated_throughput_tokens_s"],
        )
    )
PY

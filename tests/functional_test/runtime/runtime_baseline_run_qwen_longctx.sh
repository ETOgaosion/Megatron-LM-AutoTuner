#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

if [ -f .secrets/env.sh ]; then
    source .secrets/env.sh
else
    echo "Warning: .secrets/env.sh not found. Continuing with current environment."
fi

MODELS=(
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-72B"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-0.6B-Base"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-1.7B-Base"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-4B-Base"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-8B-Base"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-14B-Base"
    "Qwen/Qwen1.5-MoE-A2.7B"
)

SEQLEN="${SEQLEN:-20480}"
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-40960}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"

TP_SIZE="${TP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-4}"
EP_SIZE="${EP_SIZE:-1}"
ETP_SIZE="${ETP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-2}"
VPP_SIZE="${VPP_SIZE:-2}"

NUM_TEST_CASES="${NUM_TEST_CASES:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-3}"
SHARE_EMB="${SHARE_EMB:-None}"

TEST_CASES_DIR="${TEST_CASES_DIR:-tests/functional_test/runtime/generated_cases/qwen_longctx}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

MODEL_FILTER="${MODEL_FILTER:-}"

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6010}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

GPUS_PER_NODE=$((TP_SIZE * CP_SIZE * EP_SIZE * ETP_SIZE * PP_SIZE))

mkdir -p "${TEST_CASES_DIR}"

to_case_filename() {
    local model_name="$1"
    local normalized
    normalized=$(echo "${model_name}" | tr '[:upper:]' '[:lower:]' | tr '/.-' '_')
    echo "${normalized}_s${SEQLEN}_mtl${MAX_TOKEN_LEN}.json"
}

write_case_file() {
    local model_name="$1"
    local case_path="$2"
    cat > "${case_path}" <<EOF
{
  "model": "${model_name}",
  "cases": [
    {
      "batch_size": ${BATCH_SIZE},
      "micro_batch_size": ${MICRO_BATCH_SIZE},
      "seqlen": ${SEQLEN},
      "max_token_len": ${MAX_TOKEN_LEN},
      "shape": "bshd",
      "system": "megatron"
    },
    {
      "batch_size": ${BATCH_SIZE},
      "micro_batch_size": ${MICRO_BATCH_SIZE},
      "seqlen": ${SEQLEN},
      "max_token_len": ${MAX_TOKEN_LEN},
      "shape": "thd",
      "system": "megatron"
    }
  ]
}
EOF
}

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export UB_SKIPMC="${UB_SKIPMC:-1}"

for MODEL_NAME in "${MODELS[@]}"; do
    if [[ -n "${MODEL_FILTER}" && "${MODEL_NAME}" != *"${MODEL_FILTER}"* ]]; then
        continue
    fi

    TEST_CASES_FILE=$(to_case_filename "${MODEL_NAME}")
    TEST_CASE_PATH="${TEST_CASES_DIR}/${TEST_CASES_FILE}"
    write_case_file "${MODEL_NAME}" "${TEST_CASE_PATH}"

    DISTRIBUTED_ARGS=(
        --nproc_per_node "${GPUS_PER_NODE}"
        --nnodes "${NUM_NODES}"
        --node_rank "${NODE_RANK}"
        --master_addr "${MASTER_ADDR}"
        --master_port "${MASTER_PORT}"
    )

    PARALLEL_ARGS=(
        --tensor-model-parallel-size "${TP_SIZE}"
        --pipeline-model-parallel-size "${PP_SIZE}"
        --context-parallel-size "${CP_SIZE}"
        --expert-parallel-size "${EP_SIZE}"
        --expert-tensor-parallel-size "${ETP_SIZE}"
    )

    if [[ "${VPP_SIZE}" != "None" ]]; then
        PARALLEL_ARGS+=(--virtual-pipeline-model-parallel-size "${VPP_SIZE}")
    fi

    RUNTIME_ARGS=(
        --model-name "${MODEL_NAME}"
        --test-cases-dir "${TEST_CASES_DIR}"
        --test-cases-file "${TEST_CASES_FILE}"
        --num-test-cases "${NUM_TEST_CASES}"
        --max-iterations "${MAX_ITERATIONS}"
        --warmup-iterations "${WARMUP_ITERATIONS}"
        --output-dir "${OUTPUT_DIR}"
    )

    if [[ "${SHARE_EMB}" != "None" ]]; then
        RUNTIME_ARGS+=(--share-embeddings-and-output-weights "${SHARE_EMB}")
    fi

    echo "==================================================================="
    echo "Running runtime baseline for ${MODEL_NAME}"
    echo "case=${TEST_CASE_PATH}"
    echo "tp=${TP_SIZE} cp=${CP_SIZE} ep=${EP_SIZE} etp=${ETP_SIZE} pp=${PP_SIZE} vpp=${VPP_SIZE}"
    echo "seqlen=${SEQLEN} max_token_len=${MAX_TOKEN_LEN} batch_size=${BATCH_SIZE} micro_batch_size=${MICRO_BATCH_SIZE}"
    echo "==================================================================="

    torchrun "${DISTRIBUTED_ARGS[@]}" -m AutoTuner.runtime.baseline.main \
        "${RUNTIME_ARGS[@]}" \
        "${PARALLEL_ARGS[@]}"
done

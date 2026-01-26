#!/bin/bash

set -euo pipefail

source .secrets/env.sh

if [ -f tests/functional_test/runtime/test_env.sh ]; then
    source tests/functional_test/runtime/test_env.sh
else
    echo "Warning: tests/functional_test/runtime/test_env.sh not found. Using defaults."
    MODEL_NAME="Qwen/Qwen3-0.6B"
    TEST_CASES_FILE="qwen3_0_6b.json"
    NUM_TEST_CASES=1
    MAX_ITERATIONS=10
    WARMUP_ITERATIONS=3

    SHARE_EMB=None

    TP_SIZE=1
    CP_SIZE=1
    EP_SIZE=1
    ETP_SIZE=1
    PP_SIZE=1
    VPP_SIZE=None
fi

GPUS_PER_NODE=$(($TP_SIZE*$CP_SIZE*$EP_SIZE*$ETP_SIZE*$PP_SIZE))

MASTER_ADDR=localhost
MASTER_PORT=6010
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --context-parallel-size $CP_SIZE
    --expert-parallel-size $EP_SIZE
    --expert-tensor-parallel-size $ETP_SIZE
)

if [[ "${VPP_SIZE}" != "None" ]]; then
    PARALLEL_ARGS+=(--virtual-pipeline-model-parallel-size $VPP_SIZE)
fi

RUNTIME_ARGS=(
    --model-name $MODEL_NAME
    --test-cases-file $TEST_CASES_FILE
)

if [[ "${SHARE_EMB}" != "None" ]]; then
    RUNTIME_ARGS+=(--share-embeddings-and-output-weights $SHARE_EMB)
fi

if [[ -n "${NUM_TEST_CASES:-}" ]]; then
    RUNTIME_ARGS+=(--num-test-cases $NUM_TEST_CASES)
fi

if [[ -n "${MAX_ITERATIONS:-}" ]]; then
    RUNTIME_ARGS+=(--max-iterations $MAX_ITERATIONS)
fi

if [[ -n "${WARMUP_ITERATIONS:-}" ]]; then
    RUNTIME_ARGS+=(--warmup-iterations $WARMUP_ITERATIONS)
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1

export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0

torchrun ${DISTRIBUTED_ARGS[@]} -m AutoTuner.runtime.baseline.main \
    ${RUNTIME_ARGS[@]} \
    ${PARALLEL_ARGS[@]}

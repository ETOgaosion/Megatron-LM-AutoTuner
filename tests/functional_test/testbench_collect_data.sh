#!/bin/bash

source .secrets/env.sh

MEGATRON_LM_HASH=$(git -C "Megatron-LM" rev-parse --short=6 HEAD)
TRANSFORMER_ENGINE_HASH=$(git -C "TransformerEngine" rev-parse --short=6 HEAD)
VERL_HASH=$(git -C "verl" rev-parse --short=6 HEAD)

MODEL_NAME="Qwen/Qwen3-0.6B"
TEST_CASES_FILE="qwen3_0_6b.json"

TEST_OPS_LIST=None
TEST_CASE_IDXES=None

TIMESTAMP_VAR=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR=outputs/${TIMESTAMP_VAR}

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
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
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    # --virtual-pipeline-model-parallel-size None
    --context-parallel-size 1
    --expert-parallel-size 1
    --expert-tensor-parallel-size 1
)

PROFILE_ARGS=(
    --model-name $MODEL_NAME
    --test-cases-file $TEST_CASES_FILE
    --output-dir $OUTPUT_DIR
    --profile-mode 0
    # --theoretical-flops true
    # --theoretical-activations false
)

OPTIONAL_PROFILE_ARGS=()
if [[ "${TEST_OPS_LIST}" != "None" ]]; then
    OPTIONAL_PROFILE_ARGS+=(--test-ops-list ${TEST_OPS_LIST})
fi
if [[ "${TEST_CASE_IDXES}" != "None" ]]; then
    OPTIONAL_PROFILE_ARGS+=(--test-case-idxes ${TEST_CASE_IDXES})
fi

export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
torchrun ${DISTRIBUTED_ARGS[@]} -m AutoTuner.testbench.profile.main \
    ${PROFILE_ARGS[@]} \
    ${OPTIONAL_PROFILE_ARGS[@]} \
    ${PARALLEL_ARGS[@]}
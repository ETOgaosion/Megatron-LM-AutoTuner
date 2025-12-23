#!/bin/bash

source .secrets/env.sh

MEGATRON_LM_HASH=$(git -C "Megatron-LM" rev-parse --short=6 HEAD)
TRANSFORMER_ENGINE_HASH=$(git -C "TransformerEngine" rev-parse --short=6 HEAD)
VERL_HASH=$(git -C "verl" rev-parse --short=6 HEAD)

# Use the test environment settings if available
if [ -f tests/functional_test/test_env.sh ]; then
    source tests/functional_test/test_env.sh
else
    echo "Warning: tests/functional_test/test_env.sh not found. Using default settings."
    MODEL_NAME="Qwen/Qwen3-0.6B"
    TEST_CASES_FILE="local/qwen3_0_6b.json"

    TEST_OPS_LIST=None
    TEST_CASE_IDXES=None
    TP_COMM_OVERLAP=False
    
    TP_SIZE=1
    CP_SIZE=1
    EP_SIZE=1
    ETP_SIZE=1
    
    NSYS_BIN=nsys
fi

GPUS_PER_NODE=$(($TP_SIZE*$CP_SIZE*$EP_SIZE*$ETP_SIZE))

TIMESTAMP_VAR=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR=outputs/${TIMESTAMP_VAR}
SINGLE_NODES=${1:-False}

mkdir -p "${OUTPUT_DIR}/${MODEL_NAME}/nsys_profile"

export NVTE_NVTX_ENABLED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

PROFILE_ARGS=(
    --model-name $MODEL_NAME
    --test-cases-file $TEST_CASES_FILE
    --output-dir $OUTPUT_DIR
    --profile-mode 1
    --fix-compute-amount
)

OPTIONAL_PROFILE_ARGS=()
if [[ "${TEST_OPS_LIST}" != "None" ]]; then
    OPTIONAL_PROFILE_ARGS+=(--test-ops-list ${TEST_OPS_LIST[@]})
fi
if [[ "${TEST_CASE_IDXES}" != "None" ]]; then
    OPTIONAL_PROFILE_ARGS+=(--test-case-idxes ${TEST_CASE_IDXES[@]})
fi

PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    --expert-parallel-size $EP_SIZE
    --expert-tensor-parallel-size $ETP_SIZE
)

# Run the command and capture stderr/stdout
NSYS_OUTPUT=$(nsys profile --gpu-metrics-devices=help 2>&1)

# Check if output contains "Insufficient privilege"
if echo "$NSYS_OUTPUT" | grep -q "Insufficient privilege"; then
    GPU_METRICS_USABLE=0
else
    GPU_METRICS_USABLE=1
fi

NSYS_ARGS=(
    # --run-as root
    -w true
    -o "${OUTPUT_DIR}/${MODEL_NAME}/nsys_profile/nsight_report"
    -f true
    -x true
    -t cuda,nvtx,cudnn,cublas,python-gil
    --capture-range=cudaProfilerApi
    --capture-range-end=repeat
    --cudabacktrace=all
    --cuda-memory-usage=true
    --python-backtrace=cuda
    --enable network_interface
    --python-sampling=true
    --nic-metrics=true
)

if [ $GPU_METRICS_USABLE -eq 1 ]; then
    NSYS_ARGS=("${NSYS_ARGS[@]}"
                --gpu-metrics-devices=all
                --cuda-event-trace=false
              )
else
    echo "Warning: GPU metrics are not usable due to insufficient privileges. Proceeding without GPU metrics."
fi

if [[ "${TP_COMM_OVERLAP}" == "True" ]]; then
    export UB_SKIPMC=1
    echo "Notice that the overlap can only be enabled if you enable the config field in AutoTuner/testbench/profile/configs/override_tf_config.json"
fi

export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
if [ "$SINGLE_NODES" = "True" ]; then
    python3 -m AutoTuner.testbench.profile.nsys_main \
        ${PROFILE_ARGS[@]}
    exit $?
else
    $NSYS_BIN profile "${NSYS_ARGS[@]}" \
        torchrun ${DISTRIBUTED_ARGS[@]} -m AutoTuner.testbench.profile.main \
            ${PROFILE_ARGS[@]} \
            ${OPTIONAL_PROFILE_ARGS[@]} \
            ${PARALLEL_ARGS[@]}
    exit $?
fi
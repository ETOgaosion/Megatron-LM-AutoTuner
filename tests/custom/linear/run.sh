#!/bin/bash

KERNEL_NAME=${1:-linear}
PROFILE_MODE=${2:-all}
export CUDA_VISIBLE_DEVICES=1

echo "Profiling mode: ${PROFILE_MODE}"

mkdir -p outputs/test/custom/${KERNEL_NAME}

if [[ "$PROFILE_MODE" == "all" || "$PROFILE_MODE" == "direct" ]]; then
    echo "Running direct execution..."
    # 1. Directly run
    python -m tests.custom.${KERNEL_NAME}.${KERNEL_NAME} --num_iters 20 --draw > outputs/test/custom/${KERNEL_NAME}/data.txt
fi

if echo "$NSYS_OUTPUT" | grep -q "Insufficient privilege"; then
    GPU_METRICS_USABLE=0
else
    GPU_METRICS_USABLE=1
fi

if [ "$PROFILE_MODE" == "all" ] || [ "$PROFILE_MODE" == "nsys" ]; then
    # 2. Run with nsys profiling
    # Check if output contains "Insufficient privilege"

    NSYS_ARGS=(
        # --run-as root
        -w true
        -o "outputs/test/custom/${KERNEL_NAME}/nsight_report"
        -f true
        -x true
        -t nvtx,cudnn,cublas
        # --cudabacktrace=all
        # --cuda-memory-usage=true
        # --python-backtrace=cuda
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

    nsys profile "${NSYS_ARGS[@]}" python tests/custom/${KERNEL_NAME}/${KERNEL_NAME}.py --num_iters 1
fi

# 3. Run with ncu again to capture more accurate metrics
if [ "$PROFILE_MODE" == "all" ] || [ "$PROFILE_MODE" == "ncu" ]; then
    NCU_ARGS=(
        --set full
        --nvtx
        -f
        # --call-stack true
        --call-stack-type python
        # --target-processes all
        --export "outputs/test/custom/${KERNEL_NAME}/ncu_report.ncu-rep"
    )

    ncu ${NCU_ARGS[@]} python tests/custom/${KERNEL_NAME}/${KERNEL_NAME}.py --num_iters 1
fi

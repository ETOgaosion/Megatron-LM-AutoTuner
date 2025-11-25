#!/bin/bash

KERNEL_NAME=${1:-flash_attn}

mkdir -p outputs/test/custom/${KERNEL_NAME}

python -c "from megatron.core.utils import nvtx_range_pop"

# 1. Directly run
python -m tests.custom.${KERNEL_NAME}.${KERNEL_NAME} --iterations 20 --draw > outputs/test/custom/${KERNEL_NAME}/data.txt


# 2. Run with nsys profiling
# Check if output contains "Insufficient privilege"
if echo "$NSYS_OUTPUT" | grep -q "Insufficient privilege"; then
    GPU_METRICS_USABLE=0
else
    GPU_METRICS_USABLE=1
fi

NSYS_ARGS=(
    # --run-as root
    -w true
    -o "outputs/test/custom/${KERNEL_NAME}/nsight_report"
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

nsys profile "${NSYS_ARGS[@]}" python -m tests.custom.${KERNEL_NAME}.${KERNEL_NAME} --iterations 1


# 3. Run with ncu again to capture more accurate metrics
NCU_ARGS=(
    --set full
    --nvtx enable
    --call-stack true
    --target-processes all
    --export "outputs/test/custom/${KERNEL_NAME}/ncu_report.ncu-rep"
)

ncu ${NCU_ARGS[@]} python -m tests.custom.${KERNEL_NAME}.${KERNEL_NAME} --iterations 1
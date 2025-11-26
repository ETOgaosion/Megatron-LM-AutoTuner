#!/bin/bash

KERNEL_NAME=${1:-flash_attention}

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
    -t nvtx,cudnn,cublas,python-gil
    # --capture-range=cudaProfilerApi
    # --cudabacktrace=all
    # --cuda-memory-usage=true
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

nsys profile "${NSYS_ARGS[@]}" python tests/custom/${KERNEL_NAME}/${KERNEL_NAME}.py --iterations 2


# 3. Run with ncu again to capture more accurate metrics
# NCU_ARGS=(
#     --set full
#     --nvtx
#     # --call-stack true
#     --call-stack-type python
#     # --target-processes all
#     --export "outputs/test/custom/${KERNEL_NAME}/ncu_report.ncu-rep"
# )

# ncu ${NCU_ARGS[@]} python tests/custom/${KERNEL_NAME}/${KERNEL_NAME}.py --iterations 2
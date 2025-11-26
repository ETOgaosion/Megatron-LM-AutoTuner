#!/bin/bash

KERNEL_NAME=${1:-gemm}
PROFILE_MODE=${2:-all}

echo "Profiling mode: ${PROFILE_MODE}"

mkdir -p outputs/test/custom/${KERNEL_NAME}

if [[ "$PROFILE_MODE" == "all" || "$PROFILE_MODE" == "direct" ]]; then
    echo "Running direct execution..."
    # 1. Directly run
    python -m tests.custom.${KERNEL_NAME}.${KERNEL_NAME} --num_runs 20 --draw > outputs/test/custom/${KERNEL_NAME}/data.txt
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

    nsys profile "${NSYS_ARGS[@]}" python tests/custom/${KERNEL_NAME}/${KERNEL_NAME}.py --num_runs 1
fi

# 3. Run with ncu again to capture more accurate metrics
if [ "$PROFILE_MODE" == "all" ] || [ "$PROFILE_MODE" == "ncu" ]; then
    NCU_ARGS=(
        --set full
        --nvtx
        # -f
        # --call-stack true
        --call-stack-type python
        # --target-processes all
        --export "outputs/test/custom/${KERNEL_NAME}/ncu_report.ncu-rep"
    )

    ncu ${NCU_ARGS[@]} python tests/custom/${KERNEL_NAME}/${KERNEL_NAME}.py --num_runs 1
fi

# 4. test gemm nsys profile
if [ "$PROFILE_MODE" == "cuda_nsys" ]; then
    BLOCK_SIZES=("16 32 64 128")
    GRID_SIZES=("16 32 64 128")

    echo "Running GEMM tests with varying BLOCK_SIZE and GRID_SIZE..." > outputs/test/custom/${KERNEL_NAME}/gemm_cuda_tests.txt

    for BLOCK_SIZE in ${BLOCK_SIZES[@]}; do
        for GRID_SIZE in ${GRID_SIZES[@]}; do
            echo "Testing GEMM with BLOCK_SIZE=${BLOCK_SIZE} and GRID_SIZE=${GRID_SIZE}"
            for i in {1..3}; do
                tests/custom/${KERNEL_NAME}/${KERNEL_NAME} 1024 1024 1024 ${BLOCK_SIZE} ${BLOCK_SIZE} ${GRID_SIZE} ${GRID_SIZE}
            done
            time tests/custom/${KERNEL_NAME}/${KERNEL_NAME} 1024 1024 1024 ${BLOCK_SIZE} ${BLOCK_SIZE} ${GRID_SIZE} ${GRID_SIZE} >> outputs/test/custom/${KERNEL_NAME}/gemm_cuda_tests.txt
            NSYS_ARGS=(
                # --run-as root
                -w true
                -o "outputs/test/custom/${KERNEL_NAME}/nsight_report_cuda_BLOCK${BLOCK_SIZE}_GRID${GRID_SIZE}"
                -f true
                -x true
                -t nvtx,cudnn,cublas,python-gil
                # --capture-range=cudaProfilerApi
                # --capture-range-end=repeat
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

            # printf("Usage: ./matmul M N K block_x block_y grid_x grid_y\n");
            # printf("Example: ./matmul 1024 1024 1024 16 16 64 64\n");
            nsys profile "${NSYS_ARGS[@]}" tests/custom/${KERNEL_NAME}/${KERNEL_NAME} 1024 1024 1024 ${BLOCK_SIZE} ${BLOCK_SIZE} ${GRID_SIZE} ${GRID_SIZE}
        done
    done
fi
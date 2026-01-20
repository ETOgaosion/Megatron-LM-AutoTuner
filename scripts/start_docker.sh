#!/bin/bash

MARK=${1:-"0"}

docker create --rm -it --gpus all --shm-size=25GB --name megatron_autotuner -v $(pwd):/workspace/Megatron-LM-AutoTuner -v /data:/data --network=host --cap-add SYS_ADMIN whatcanyousee/megatron-autotuner-env:mcore0.15.1_te2.7
docker start megatron_autotuner
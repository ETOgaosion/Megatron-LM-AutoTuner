# Environment for Claude vibe-coding

## About host machine

Remenber that you are currently work on an Linux Host Server with 8 5090 GPUs. You can either use conda environment to do tests without privilege auth, or use docker image `whatcanyousee/megatron-autotuner-env:mcore0.15.1_te2.7`

Actually there are 2 machines 5090-1, 5090-2, read .secrets for IP and username, our project in each machine is in the same location.

## About Conda

We use `conda` in server machine, execute `conda activate megatron-lm-autotuner` to activate virtual environment, execute in it. If there are any environment issue, please make sure the packages is the same with `requirements_dev.txt`.

If you need to test your codes or I ask you to test, try below way to access the test environment.

These packages are always fixed for this project:

- CUDA: 12.8
- CUDNN: 9.10.2 (not strictly needed)
- nvidia-cudnn-cu12: 9.10.2.21
- Flash Attention: 2.7.4.post1
- torch: 2.8.0+cu128
- TransformerEngine: 2.7.0
- Megatron Core: core_v0.15.1

## About docker

In some case, we use `docker` to run our codes.

There are 2 kinds of docker: for baseline and for enhanced implementations

### Baseline docker

The container has been ready: name is `megatron_autotuner`, simply run `docker exec -it megatron_autotuner bash` to enter it. If there is error with `nvidia-smi`, you should restart it or recreate docker container:

```sh
docker create --rm -it --gpus all --shm-size=25GB --name megatron_autotuner -v $(pwd):/workspace/Megatron-LM-AutoTuner -v /data:/data --network=host --cap-add SYS_ADMIN whatcanyousee/megatron-autotuner-env:mcore0.15.1_te2.7
```

### Enhanced docker

The container has been ready: name is `megatron_autotuner_enhanced`, simply run `docker exec -it megatron_autotuner_enhanced bash` to enter it. If there is error with `nvidia-smi`, you should restart it or recreate docker container:

```sh
docker create --rm -it --gpus all --shm-size=25GB --name megatron_autotuner_enhanced -v $(pwd):/workspace/Megatron-LM-AutoTuner -v /data:/data --network=host --cap-add SYS_ADMIN whatcanyousee/megatron-autotuner-env:mcore0.15.1_te2.7
```

In this docker, you shall manually install [Megatron-Enhanced](../../../Megatron-Enhanced) and [TransformerEngine-Enhanced](../../../TransformerEngine-Enhanced) in this repo
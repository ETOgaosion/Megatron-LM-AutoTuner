# Environment for Claude vibe-coding

## About host machine

Remenber that you are currently work on an Apple Sillicon MacBook Pro, since our sever network cannot reach out Claude Server, please connect into environment with Terminal app in host machine.

Command to connect to ssh machine is:

```sh
ssh 5090-1
ssh 5090-2
```

I recommand `5090-1`. When in target machine, go to `~/projects/Megatron-LM-AutoTuner`, this is our repo.

## About remote server machine

If you need to test your codes or I ask you to test, try below way to access the test environment.

These packages are always fixed for this project:

- CUDA: 12.8
- CUDNN: 9.10.2 (not strictly needed)
- nvidia-cudnn-cu12: 9.10.2.21
- Flash Attention: 2.7.4.post1
- torch: 2.8.0+cu128
- TransformerEngine: 2.7.0
- Megatron Core: core_v0.15.1

We use `conda` in target remote machine, execute `conda activate megatron-lm-autotuner` to activate virtual environment, execute in it. If there are any environment issue, please make sure the packages is the same with `requirements_dev.txt`.

## In remote server

In some case, we use `docker` to run our codes, the container has been ready: name is `megatron_autotuner_new`, simply run `docker exec -it megatron_autotuner_new bash` to enter it. If there is error with `nvidia-smi`, you should restart it or recreate docker container:

```sh
docker create --rm -it --gpus all --shm-size=25GB --name megatron_autotuner_new -v $(pwd):/workspace/Megatron-LM-AutoTuner -v /data:/data --network=host --cap-add SYS_ADMIN whatcanyousee/megatron-autotuner-env:mcore0.15.1_te2.7
```
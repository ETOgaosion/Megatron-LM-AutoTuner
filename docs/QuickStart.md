# Quick Start

## Pre-Requirements

Followed the [install guides](./Install.md)

Followed [.secrets/env_sample.sh](../.secrets/env_sample.sh), create an `env.sh` in the same directory to hold your config.

Followed [tests/functional_test/test_env_sample.sh](../tests/functional_test/test_env_sample.sh), create an `test_env.sh` in the same directory to hold your testing config.

## To Try Collect Data

Run [tests/functional_test/testbench_collect_data.sh](../tests/functional_test/testbench_collect_data.sh), modify the configs to follow your environment.

The outputs like:

```sh
(megatron-lm-autotuner) ➜  Megatron-LM-AutoTuner git:(main) ls outputs/2025-10-15_17-48-12/Qwen/Qwen3-0.6B/collect_data                  
rank_0  rank_1  rank_2  rank_3
```

## To Try Torch Profiler

Run [tests/functional_test/testbench_torch_profile.sh](tests/functional_test/testbench_torch_profile.sh), modify the configs to follow your environment.

After finish, you can check output dir:

```sh
(megatron-lm-autotuner) ➜  Megatron-LM-AutoTuner git:(main) ✗ ls outputs/2025-10-17_16-22-50/Qwen/Qwen3-0.6B/torch_profiler                                                        
jss-Rack-Server_1528039.1760689383431851581.pt.trace.json  jss-Rack-Server_1528040.1760689383437475633.pt.trace.json  jss-Rack-Server_1528041.1760689383439235537.pt.trace.json  jss-Rack-Server_1528042.1760689383440949434.pt.trace.json
```

Then run:

```
pip install torch_tb_profiler

tensorboard --logdir=outputs/2025-10-17_16-22-50/Qwen/Qwen3-0.6B/torch_profiler --host 0.0.0.0
```

Open your browser and go to `http://[ip]:6006/#pytorch_profiler`, you will see the traces in `Views` tabs, use `WASD` to check the traces.

![torch_profiler](./figs/QuickStart/torch_profiler.png)

## To Try Nsys Profiler

**NOTE: This function requires to use docker: `verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2`, since nsys can only work in root mode.**

1. Launch docker:

```bash
docker create --rm -it --gpus all --shm-size=25GB --name megatron_autotuner -v $(pwd):/workspace/Megatron-LM-AutoTuner --network=host verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2

docker start megatron_autotuner

docker exec -it megatron_autotuner bash
```

2. In docker container:

```bash
cd Megatron-LM-AutoTuner

cd verl && pip install --no-deps -e . && cd ..

bash tests/functional_test/testbench_nsys_profile.sh
```

3. In your local Mac/PC, download [nsight compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)

Open profiled file in path: `outputs/[TIMESTAMP]/Qwen/Qwen3-0.6B/nsys_profile/nsight_report.nsys-rep`

Result:

![nsys_profile_sample](./figs/QuickStart/nsys_result_sample.png)
# About Testing

You should test your implementation after finishing. Read [Env.md](./Env.md) first, remember that:

- Your remote machine is `5090-1`
- In the remote machine, our project path is `~/projects/Megatron-LM-AutoTuner`
- In the remote machine, you should use conda environment `megatron-lm-autotuner-enhanced` to test
- In the remote machine, there is a docker image with proper packages: `whatcanyousee/megatron-autotuner-env:mcore0.15.1_te2.7`, there has already been a container named `megatron_autotuner_enhanced`.

The tests are mainly involved with each submodules unit tests, our [functional tests](../../tests/functional_test/).

Notice that in remote machine, there is a test_env.sh in [that folder](../../tests/functional_test/), you should change the operator you implemented, or the configurations involved.

Then run these scripts:

- [testbench_collect_data.sh](../../tests/functional_test/testbench_collect_data.sh): can test in host machine
- [testbench_torch_profile.sh](../../tests/functional_test/testbench_torch_profile.sh): can test in host machine
- [testbench_nsys_profile.sh](../../tests/functional_test/testbench_nsys_profile.sh): can only test in docker container `megatron_autotuner_enhanced`


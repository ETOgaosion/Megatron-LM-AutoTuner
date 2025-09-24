import os

import torch
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


def init_distributed_single_node():
    """Initialize distributed environment"""
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(0)


def init_distributed_multi_nodes(
    tp: int = 1,
    cp: int = 1,
    ep: int = 1,
    etp: int | None = None,
    pp: int = 1,
    vpp: int | None = None,
) -> None:
    """Initialize distributed environment"""
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(torch.device(int(os.environ["LOCAL_RANK"])))
    if pp <= 1:
        # check megatron arguments.py
        assert vpp is None, "vpp must be None when pp <= 1"
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
    )
    model_parallel_cuda_manual_seed(0)


def destroy_distributed():
    """Destroy distributed environment"""
    torch.distributed.destroy_process_group()
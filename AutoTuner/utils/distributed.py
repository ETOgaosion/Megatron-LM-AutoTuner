import os

import torch
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


def _get_local_device_index() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _init_nccl_process_group() -> None:
    """Initialize NCCL process group with explicit device binding when supported."""
    local_rank = _get_local_device_index()
    torch.cuda.set_device(torch.device(local_rank))

    try:
        torch.distributed.init_process_group("nccl", device_id=torch.device(local_rank))
    except TypeError:
        torch.distributed.init_process_group("nccl")


def init_distributed_single_node():
    """Initialize distributed environment"""
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    _init_nccl_process_group()
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
    _init_nccl_process_group()
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

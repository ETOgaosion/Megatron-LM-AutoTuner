import datetime
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchcomms import new_comm

TORCH_PROFILER_OUTPUT_DIR = (
    "outputs/functional/cpu_gpu_movements/interference_with_comm/torch_profiler"
)

prof = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler(TORCH_PROFILER_OUTPUT_DIR),
    record_shapes=False,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    with_modules=True,
)


def init_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12354"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    # dist.init_process_group(
    #     backend=backend,
    #     rank=rank,
    #     world_size=world_size,
    # )
    torch.cuda.set_device(rank)
    device = torch.device(torch.cuda.current_device())
    torchcomm = new_comm(
        "ncclx", device, name="main_comm", timeout=datetime.timedelta(seconds=300)
    )
    print(
        f"Initialized process {rank} of {world_size} on backend {backend}, torchcomm: {torchcomm.get_rank()}"
    )
    return torchcomm


def rank0_process(rank: int, world_size: int, tensor_size: tuple = (16, 8192, 8, 2048)):
    """
    Rank 0: One stream for GPU->CPU offload with pinned memory,
    another stream for send operations.
    """
    torchcomm = init_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Create streams for overlap
    offload_stream = torch.cuda.Stream(device=device)
    send_stream = torch.cuda.Stream(device=device)

    # Create GPU tensor
    gpu_tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)

    # Create pinned CPU tensor for async transfer
    cpu_tensor = torch.empty(tensor_size, dtype=torch.float32).pin_memory()

    # Tensor for communication
    send_tensor = torch.ones(tensor_size, dtype=torch.float32, device=device)

    print(f"Rank {rank}: Starting overlapped operations")
    start_time = time.time()

    prof.start()

    # Operation 2: Send on different stream (overlaps with offload)
    with torch.cuda.stream(send_stream):
        for _ in range(5):  # Send multiple times to increase communication load
            torchcomm.send(send_tensor, dst=1)

    # Operation 1: Async GPU->CPU offload with pinned memory
    with torch.cuda.stream(offload_stream):
        for _ in range(5):  # Offload multiple times to increase load
            cpu_tensor.copy_(gpu_tensor, non_blocking=True)

    # Synchronize streams
    offload_stream.synchronize()
    send_stream.synchronize()

    elapsed = time.time() - start_time
    print(f"Rank {rank}: Completed in {elapsed:.4f}s")
    print(f"Rank {rank}: CPU tensor sample: {cpu_tensor[:5]}")

    prof.stop()

    # dist.destroy_process_group()


def rank1_process(rank: int, world_size: int, tensor_size: tuple = (16, 8192, 8, 2048)):
    """
    Rank 1: One stream for CPU load (computation on CPU data),
    another stream for recv operations.
    """
    torchcomm = init_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Create streams for overlap
    load_stream = torch.cuda.Stream(device=device)
    recv_stream = torch.cuda.Stream(device=device)

    # Create pinned CPU tensor for CPU operations
    cpu_data = torch.randn(tensor_size, dtype=torch.float32).pin_memory()

    # Tensor for communication
    recv_tensor = torch.empty(tensor_size, dtype=torch.float32, device=device)

    print(f"Rank {rank}: Starting overlapped operations")
    start_time = time.time()

    prof.start()

    # Operation 2: Recv on different stream (overlaps with load)
    with torch.cuda.stream(recv_stream):
        for _ in range(5):  # Receive multiple times to increase communication load
            torchcomm.recv(recv_tensor, src=0)

    # Operation 1: CPU load - transfer pinned memory to GPU for computation
    with torch.cuda.stream(load_stream):
        for _ in range(5):  # Load multiple times to increase load
            gpu_data = cpu_data.to(device, non_blocking=True)

    # Synchronize streams
    load_stream.synchronize()
    recv_stream.synchronize()

    prof.stop()

    elapsed = time.time() - start_time
    print(f"Rank {rank}: Completed in {elapsed:.4f}s")
    print(f"Rank {rank}: Received tensor sample: {recv_tensor[:5]}")

    # dist.destroy_process_group()


def main():
    """Main entry point for multi-process execution."""
    world_size = 2
    tensor_size = (16, 8192, 8, 2048)

    processes = []

    # Rank 0 process
    p0 = mp.Process(target=rank0_process, args=(0, world_size, tensor_size))
    p0.start()
    processes.append(p0)

    # Rank 1 process
    p1 = mp.Process(target=rank1_process, args=(1, world_size, tensor_size))
    p1.start()
    processes.append(p1)

    # Wait for all processes
    for p in processes:
        p.join()

    print("All processes completed")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

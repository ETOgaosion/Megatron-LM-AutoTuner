#!/usr/bin/env python3
# example.py
import torch
from torchcomms import ReduceOp, new_comm


def main():
    # Initialize TorchComm with NCCLX backend
    device = torch.device("cuda")
    torchcomm = new_comm("ncclx", device, name="main_comm")

    # Get rank and world size
    rank = torchcomm.get_rank()
    world_size = torchcomm.get_size()

    # Calculate device ID
    num_devices = torch.cuda.device_count()
    device_id = rank % num_devices
    target_device = torch.device(f"cuda:{device_id}")

    print(f"Rank {rank}/{world_size}: Running on device {device_id}")

    # Create a tensor with rank-specific data
    tensor = torch.full(
        (1024,), float(rank + 1), dtype=torch.float32, device=target_device
    )

    print(f"Rank {rank}: Before AllReduce: {tensor[0].item()}")

    # Perform synchronous AllReduce (sum across all ranks)
    torchcomm.all_reduce(tensor, ReduceOp.SUM, async_op=False)

    # Synchronize CUDA stream
    torch.cuda.current_stream().synchronize()

    print(f"Rank {rank}: After AllReduce: {tensor[0].item()}")

    # Cleanup
    torchcomm.finalize()


if __name__ == "__main__":
    main()

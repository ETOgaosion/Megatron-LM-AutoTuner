import logging
import os
import statistics
import subprocess as sp
from typing import Optional

from ..utils.logging import log_rank0


def get_gpu_memory() -> float:
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return statistics.mean(memory_free_values)


def get_memory_str(mem: int, human_readable: bool = True) -> str:
    if human_readable:
        if mem < 1024:
            return f"{mem} B"
        elif mem < 1024**2:
            return f"{mem / 1024:.2f} KB"
        elif mem < 1024**3:
            return f"{mem / (1024 ** 2):.2f} MB"
        else:
            return f"{mem / (1024 ** 3):.2f} GB"
    else:
        return str(mem)


def reset_peak_memory_stats(
    device: Optional[int] = None, synchronize: bool = True
) -> None:
    import torch

    if device is None:
        device = torch.cuda.current_device()
    if synchronize:
        torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)


def get_rank_peak_memory_stats(
    device: Optional[int] = None, synchronize: bool = True
) -> dict:
    import torch

    if device is None:
        device = torch.cuda.current_device()
    if synchronize:
        torch.cuda.synchronize(device)

    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    real_detected_bytes = int(total_bytes - free_bytes)

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = int(os.getenv("RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))

    return {
        "rank": rank,
        "world_size": world_size,
        "device_index": int(device),
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_allocated": get_memory_str(peak_allocated_bytes),
        "peak_reserved_bytes": peak_reserved_bytes,
        "peak_reserved": get_memory_str(peak_reserved_bytes),
        "real_detected_bytes": real_detected_bytes,
        "real_detected": get_memory_str(real_detected_bytes),
        "total_device_bytes": int(total_bytes),
        "total_device_memory": get_memory_str(int(total_bytes)),
    }


def get_all_rank_peak_memory_stats(
    device: Optional[int] = None, synchronize: bool = True
) -> list[dict]:
    import torch

    local_stats = get_rank_peak_memory_stats(device=device, synchronize=synchronize)
    if not torch.distributed.is_initialized():
        return [local_stats]

    local_tensor = torch.tensor(
        [
            local_stats["device_index"],
            local_stats["peak_allocated_bytes"],
            local_stats["peak_reserved_bytes"],
            local_stats["real_detected_bytes"],
            local_stats["total_device_bytes"],
        ],
        dtype=torch.int64,
        device=torch.cuda.current_device(),
    )
    gathered = [torch.empty_like(local_tensor) for _ in range(local_stats["world_size"])]
    torch.distributed.all_gather(gathered, local_tensor)

    all_rank_stats = []
    for rank, rank_tensor in enumerate(gathered):
        device_index = int(rank_tensor[0].item())
        peak_allocated_bytes = int(rank_tensor[1].item())
        peak_reserved_bytes = int(rank_tensor[2].item())
        real_detected_bytes = int(rank_tensor[3].item())
        total_device_bytes = int(rank_tensor[4].item())
        all_rank_stats.append(
            {
                "rank": rank,
                "world_size": local_stats["world_size"],
                "device_index": device_index,
                "peak_allocated_bytes": peak_allocated_bytes,
                "peak_allocated": get_memory_str(peak_allocated_bytes),
                "peak_reserved_bytes": peak_reserved_bytes,
                "peak_reserved": get_memory_str(peak_reserved_bytes),
                "real_detected_bytes": real_detected_bytes,
                "real_detected": get_memory_str(real_detected_bytes),
                "total_device_bytes": total_device_bytes,
                "total_device_memory": get_memory_str(total_device_bytes),
            }
        )
    return all_rank_stats


class MemoryTrackerContext:
    def __init__(
        self, name: str = "", log_level: int = logging.INFO, human_readable: bool = True
    ):
        self.name = name
        self.log_level = log_level
        self.human_readable = human_readable

    def __enter__(self) -> "MemoryTrackerContext":
        import torch

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self.start_mem = torch.cuda.memory_allocated()
        self.start_peak_mem = torch.cuda.max_memory_allocated()
        self.start_real_mem = get_gpu_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import torch

        torch.cuda.synchronize()
        self.end_mem = torch.cuda.memory_allocated()
        self.end_peak_mem = torch.cuda.max_memory_allocated()
        self.end_real_mem = get_gpu_memory()
        self.peak_mem_diff = self.end_peak_mem - self.start_peak_mem
        self.mem_diff = self.end_mem - self.start_mem
        self.real_mem_diff = self.end_real_mem - self.start_real_mem
        self.result = {
            "start_mem": get_memory_str(self.start_mem, self.human_readable),
            "end_mem": get_memory_str(self.end_mem, self.human_readable),
            "start_peak_mem": get_memory_str(self.start_peak_mem, self.human_readable),
            "end_peak_mem": get_memory_str(self.end_peak_mem, self.human_readable),
            "start_real_mem": get_memory_str(self.start_real_mem, self.human_readable),
            "end_real_mem": get_memory_str(self.end_real_mem, self.human_readable),
            "mem_diff": get_memory_str(self.mem_diff, self.human_readable),
            "peak_mem_diff": get_memory_str(self.peak_mem_diff, self.human_readable),
            "real_mem_diff": get_memory_str(self.real_mem_diff, self.human_readable),
        }
        reserved_mem = torch.cuda.memory_reserved()
        log_rank0(
            f"[MemoryTracker] {self.name} | "
            f"Start Mem: {get_memory_str(self.start_mem, self.human_readable)} | "
            f"End Mem: {get_memory_str(self.end_mem, self.human_readable)} | "
            f"Start Peak Mem: {get_memory_str(self.start_peak_mem, self.human_readable)} | "
            f"End Peak Mem: {get_memory_str(self.end_peak_mem, self.human_readable)} | "
            f"Start Real Mem: {get_memory_str(self.start_real_mem, self.human_readable)} | "
            f"End Real Mem: {get_memory_str(self.end_real_mem, self.human_readable)} | "
            f"Memory Diff: {get_memory_str(self.mem_diff, self.human_readable)} | "
            f"Peak Memory Diff: {get_memory_str(self.peak_mem_diff, self.human_readable)} | "
            f"Real Memory Diff: {get_memory_str(self.real_mem_diff, self.human_readable)} | "
            f"Reserved Memory: {get_memory_str(reserved_mem, self.human_readable)}",
            level=self.log_level,
        )

    def get_result(self) -> dict:
        return self.result


class MemoryTracker:
    @staticmethod
    def track_function(func: callable, *args, **kwargs) -> tuple:
        with MemoryTrackerContext(name=func.__name__) as tracker:
            result = func(*args, **kwargs)
        return result, {
            "start_mem": tracker.start_mem,
            "end_mem": tracker.end_mem,
            "start_peak_mem": tracker.start_peak_mem,
            "end_peak_mem": tracker.end_peak_mem,
            "start_real_mem": tracker.start_real_mem,
            "end_real_mem": tracker.end_real_mem,
            "mem_diff": tracker.mem_diff,
            "peak_mem_diff": tracker.peak_mem_diff,
            "real_mem_diff": tracker.real_mem_diff,
        }

    @staticmethod
    def track_decorator(preffix: str = "") -> callable:
        def decorator(func: callable) -> callable:
            def wrapper(*args, **kwargs):
                with MemoryTrackerContext(name=f"{preffix} {func.__name__}") as tracker:
                    result = func(*args, **kwargs)
                return result, {
                    "start_mem": tracker.start_mem,
                    "end_mem": tracker.end_mem,
                    "start_peak_mem": tracker.start_peak_mem,
                    "end_peak_mem": tracker.end_peak_mem,
                    "start_real_mem": tracker.start_real_mem,
                    "end_real_mem": tracker.end_real_mem,
                    "mem_diff": tracker.mem_diff,
                    "peak_mem_diff": tracker.peak_mem_diff,
                    "real_mem_diff": tracker.real_mem_diff,
                }

            return wrapper

        return decorator


class ActivationHook:
    def __init__(
        self,
        enable: bool = True,
        module_name: str = "",
        logging_level: int = logging.INFO,
        online: bool = False,
    ):
        self.activation_tensors = []
        self.enable = enable
        self.module_name = module_name
        self.logging_level = logging_level
        self.online_mem_res = 0
        self.online = online

    def save_hook(self, x) -> object:
        if self.enable:
            log_rank0(
                f"[{self.module_name} save] tensor shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}",
                level=self.logging_level,
            )
            self.activation_tensors.append(x)
            if self.online:
                self.online_mem_res += x.numel() * x.element_size()
        return x  # Must return x, otherwise the computation graph will error

    def load_hook(self, x) -> object:
        if self.enable:
            log_rank0(
                f"[{self.module_name} load] tensor shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}",
                level=self.logging_level,
            )
        return x

    def clear(self) -> None:
        self.activation_tensors = []
        self.online_mem_res = 0

    def get_activation_memory(self) -> int:
        if self.online:
            return self.online_mem_res
        mem = 0

        for tensor in self.activation_tensors:
            mem += tensor.numel() * tensor.element_size()
        return mem

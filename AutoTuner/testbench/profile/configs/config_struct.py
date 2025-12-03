from dataclasses import dataclass


class ProfileMode:
    collect_data: int = 0
    nsys_profile: int = 1
    torch_profiler: int = 2
    torch_memory_snapshot: int = 3


PROFILE_MODEL_MAP = {
    ProfileMode.collect_data: "collect_data",
    ProfileMode.nsys_profile: "nsys_profile",
    ProfileMode.torch_profiler: "torch_profiler",
    ProfileMode.torch_memory_snapshot: "torch_memory_snapshot",
}


@dataclass
class ProfileConfig:
    profile_mode: int = (
        ProfileMode.collect_data
    )  # 0: collect data, 1: nsys profile, 2: torch profiler
    warmup_iters: int = 2  # warmup_iters operator executions
    theoretical_flops: bool = False
    theoretical_activations: bool = False


@dataclass
class TorchProfilerConfig:
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True
    with_flops: bool = True
    with_modules: bool = True
    activities: list = (
        None  # if None, will be set to [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    )
    schedule: object = (
        None  # if None, will be set to torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2)
    )
    # on_trace_ready: object = None  # if None, will be set to torch.profiler.tensorboard_trace_handler('./log_dir')
    output_dir: str = (
        "log_dir"  # will be in `outputs/timestamp/hf_model_name/torch_profiler/log_dir`
    )


@dataclass
class MemorySnapshotConfig:
    snapshot_interval: int = (
        1  # interval between two snapshots in number of operator runs
    )
    output_dir: str = (
        "memory_snapshots"  # will be in `outputs/timestamp/hf_model_name/memory_snapshots/`
    )

import torch
import os

def d2h(x):
    x_cpu = torch.empty_like(x.data, device=torch.device('cpu'), pin_memory=True)
    x_cpu.copy_(x.data, non_blocking=True)
    x.cpu_data = x_cpu
    x.data = torch.empty((1,), device=torch.device('cuda'))

def h2d(x):
    x_gpu = torch.empty_like(x.cpu_data, device=torch.device('cuda'))
    x_gpu.copy_(x.cpu_data, non_blocking=True)
    x.data = x_gpu
    x.cpu_data = None

if __name__ == '__main__':
    xs = [torch.randn(16000, 8192, device='cuda', dtype=torch.bfloat16) for _ in range(10)]
    os.makedirs(f"./outpus/functional/cpu_gpu_movements", exist_ok=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./outputs/functional/cpu_gpu_movements/profile'),
        record_shapes=True,
        with_stack=True,
        with_modules=True,
    ) as prof:
        for _ in range(10):
            for x in xs:
                d2h(x)
            for x in xs:
                h2d(x)
            prof.step()
import argparse
import torch
import triton
import triton.language as tl
import time
from typing import Tuple
from torch.cuda import cudart, check_error
from megatron.core.utils import (
    configure_nvtx_profiling,
    nvtx_decorator,
    nvtx_range_pop,
    nvtx_range_push,
)

# ================================================================
# Customizable LayerNorm Kernel (Forward + Backward)
# ================================================================

@triton.jit
def layernorm_fwd_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    mean_ptr, rstd_ptr,
    M, N,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    x = tl.load(x_ptr + offsets * N, mask=mask, other=0.0)
    # Load full rows
    row_ptrs = x_ptr + offsets[:, None] * N + tl.arange(0, N)[None, :]
    row = tl.load(row_ptrs, mask=mask[:, None])

    mean = tl.mean(row, axis=1)
    var = tl.var(row, axis=1)
    rstd = 1.0 / tl.sqrt(var + eps)

    # Store mean and rstd for backward
    tl.store(mean_ptr + offsets, mean, mask=mask)
    tl.store(rstd_ptr + offsets, rstd, mask=mask)

    # Normalize and apply affine transform
    weight = tl.load(weight_ptr + tl.arange(0, N))
    bias = tl.load(bias_ptr + tl.arange(0, N))

    x_norm = (row - mean[:, None]) * rstd[:, None]
    y = x_norm * weight[None, :] + bias[None, :]

    tl.store(y_ptr + offsets[:, None] * N + tl.arange(0, N)[None, :], y, mask=mask[:, None])


@triton.jit
def layernorm_bwd_kernel(
    dy_ptr, x_ptr, weight_ptr,
    mean_ptr, rstd_ptr,
    dx_ptr, dweight_ptr, dbias_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    # Load dy and x rows
    dy_row = tl.load(dy_ptr + offsets[:, None] * N + tl.arange(0, N)[None, :], mask=mask[:, None])
    x_row = tl.load(x_ptr + offsets[:, None] * N + tl.arange(0, N)[None, :], mask=mask[:, None])
    mean = tl.load(mean_ptr + offsets, mask=mask)
    rstd = tl.load(rstd_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + tl.arange(0, N))

    # Center
    x_hat = (x_row - mean[:, None]) * rstd[:, None]

    # Compute gradients
    dweight_local = tl.sum(dy_row * x_hat, axis=0)
    dbias_local = tl.sum(dy_row, axis=0)

    # dx = (dy * weight) * rstd * (1 - x_hat^2 * var_factor) - mean adjustments
    N_float = N
    dy_weight = dy_row * weight[None, :]
    dx = dy_weight * rstd[:, None]

    # Subtract mean components
    mean_dy = tl.sum(dy_weight, axis=1) / N_float
    mean_dy_xhat = tl.sum(dy_weight * x_hat, axis=1) / N_float
    dx = dx - mean_dy[:, None] - x_hat * mean_dy_xhat[:, None]

    # Accumulate dweight and dbias
    tl.atomic_add(dweight_ptr + tl.arange(0, N), dweight_local)
    tl.atomic_add(dbias_ptr + tl.arange(0, N), dbias_local)

    tl.store(dx_ptr + offsets[:, None] * N + tl.arange(0, N)[None, :], dx, mask=mask[:, None])


class TritonLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        M, N = x.shape
        y = torch.empty_like(x)
        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        BLOCK_SIZE = triton.next_power_of_2(min(M, 1024))
        GRID = (M + BLOCK_SIZE - 1) // BLOCK_SIZE

        layernorm_fwd_kernel[GRID](
            x, weight, bias, y, mean, rstd,
            M, N, eps,
            BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.eps = eps
        ctx.MN = (M, N)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, rstd = ctx.saved_tensors
        M, N = ctx.MN
        dx = torch.empty_like(x)
        dweight = torch.zeros_like(weight)
        dbias = torch.zeros_like(weight)

        GRID = (M + ctx.BLOCK_SIZE - 1) // ctx.BLOCK_SIZE

        layernorm_bwd_kernel[GRID](
            dy, x, weight, mean, rstd,
            dx, dweight, dbias,
            M, N,
            BLOCK_SIZE=ctx.BLOCK_SIZE
        )

        return dx, dweight, dbias, None


# ================================================================
# Benchmarking Script: Test Different BLOCK_SIZE and GRID configs
# ================================================================

def benchmark_triton_layernorm(
    M: int = 4096,
    N: int = 2048,
    num_warmup: int = 50,
    num_iters: int = 200,
    block_sizes: Tuple[int, ...] = (128, 256, 512, 1024, 2048),
):
    torch.manual_seed(42)
    device = torch.device("cuda")

    x = torch.randn(M, N, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)

    # Reference PyTorch LayerNorm
    torch_ln = torch.nn.LayerNorm(N, eps=1e-5, elementwise_affine=True).to(device)
    torch_ln.weight.data.copy_(weight)
    torch_ln.bias.data.copy_(bias)

    print(f"Benchmarking LayerNorm: M={M}, N={N}")
    print(f"{'Mode':<10} {'BLOCK':<8} {'GRID':<8} {'Time (ms)':<12} {'TFLOPS':<8} {'Speedup vs Torch'}")

    # Baseline: PyTorch
    torch.cuda.synchronize()
    for _ in range(num_warmup):
        y = torch_ln(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        y = torch_ln(x)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / num_iters * 1000
    tflops_torch = 2 * M * N * N / 1e12 / (torch_time / 1000)

    print(f"{'Torch':<10} {'-':<8} {'-':<8} {torch_time:<12.3f} {tflops_torch:<8.2f} {'1.00x'}")

    results = []

    for BLOCK_SIZE in block_sizes:
        if BLOCK_SIZE > M:
            continue

        GRID = (M + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Custom Triton LayerNorm with fixed block size
        def triton_fwd():
            return TritonLayerNorm.apply(x, weight, bias)

        # Warmup
        for _ in range(num_warmup):
            y = triton_fwd()
            y.backward(torch.randn_like(y))
        torch.cuda.synchronize()

        # Forward benchmark
        check_error(cudart().cudaProfilerStart())
        start = time.time()
        for _ in range(num_iters):
            nvtx_range_push("LayerNorm Forward")
            y = triton_fwd()
            nvtx_range_pop("LayerNorm Forward")
        torch.cuda.synchronize()
        check_error(cudart().cudaProfilerStop())
        fwd_time = (time.time() - start) / num_iters * 1000

        # Backward benchmark
        y = triton_fwd()
        dy = torch.randn_like(y)
        torch.cuda.synchronize()
        check_error(cudart().cudaProfilerStart())
        start = time.time()
        for _ in range(num_iters):
            nvtx_range_push("LayerNorm Backward")
            y.backward(dy, retain_graph=True)
            nvtx_range_pop("LayerNorm Backward")
        torch.cuda.synchronize()
        check_error(cudart().cudaProfilerStop())
        bwd_time = (time.time() - start) / num_iters * 1000

        total_time = fwd_time + bwd_time
        tflops = 4 * M * N * N / 1e12 / (total_time / 1000)  # rough fwd+bwd estimate

        speedup = torch_time / total_time

        print(f"{'FWD':<10} {BLOCK_SIZE:<8} {GRID:<8} {fwd_time:<12.3f} {tflops:<8.2f} {speedup:>6.2f}x")
        print(f"{'BWD':<10} {BLOCK_SIZE:<8} {GRID:<8} {bwd_time:<12.3f} {'':<8} {'':<6}")
        print(f"{'FWD+BWD':<10} {BLOCK_SIZE:<8} {GRID:<8} {total_time:<12.3f} {'':<8} {speedup:>6.2f}x")

        results.append({
            'block': BLOCK_SIZE,
            'grid': GRID,
            'fwd_ms': fwd_time,
            'bwd_ms': bwd_time,
            'total_ms': total_time,
            'speedup': speedup
        })

    return results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=65536, help="Number of rows")
    parser.add_argument("--N", type=int, default=1024, help="Number of columns")
    parser.add_argument("block_sizes", nargs='+', type=int, default=[128, 256, 512, 1024, 2048], help="List of block sizes to benchmark")

    parser.add_argument("--num_warmups", type=int, default=5)
    parser.add_argument("--num_runs", type=int, default=1)
    
    parser.add_argument("--output-dir", type=str, default="outputs/test/custom/gemm", help="Output directory")
    
    parser.add_argument("--draw", action="store_true", help="Draw performance figure")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    results = benchmark_triton_layernorm(
        M=args.M,
        N=args.N,
        num_warmup=args.num_warmups,
        num_iters=args.num_runs,
        block_sizes=tuple(args.block_sizes)
    )
    if args.draw:
        import matplotlib.pyplot as plt
        blocks = [r['block'] for r in results]
        fwd_times = [r['fwd_ms'] for r in results]
        bwd_times = [r['bwd_ms'] for r in results]

        plt.figure(figsize=(10, 6))
        plt.plot(blocks, fwd_times, marker='o', label='Forward Time (ms)')
        plt.plot(blocks, bwd_times, marker='s', label='Backward Time (ms)')
        plt.xlabel('BLOCK_SIZE')
        plt.ylabel('Time (ms)')
        plt.title('Triton LayerNorm Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{args.output_dir}/layernorm_performance.png", dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {args.output_dir}/layernorm_performance.png")
        plt.show()
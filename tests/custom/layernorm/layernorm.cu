#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) \
    do { cudaError_t e = (x); if (e != cudaSuccess) { \
        printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    }} while (0)


// Simple LayerNorm kernel
// Input shape: [N, H]
// Each block handles multiple rows depending on grid size
__global__ void layernorm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, int H)
{
    int row = blockIdx.x;
    if (row >= N) return;

    // Compute mean
    float mean = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        mean += input[row * H + i];

    // block-wide reduction
    __shared__ float sm_mean;
    __shared__ float sm_var;

    // reduce mean
    __shared__ float buf[1024];
    buf[threadIdx.x] = mean;

    __syncthreads();

    // reduction tree
    int stride = blockDim.x / 2;
    while (stride > 0) {
        if (threadIdx.x < stride)
            buf[threadIdx.x] += buf[threadIdx.x + stride];
        __syncthreads();
        stride /= 2;
    }

    if (threadIdx.x == 0)
        sm_mean = buf[0] / H;

    __syncthreads();

    // compute variance
    float var = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float x = input[row * H + i];
        var += (x - sm_mean) * (x - sm_mean);
    }

    // reduction for var
    buf[threadIdx.x] = var;
    __syncthreads();

    stride = blockDim.x / 2;
    while (stride > 0) {
        if (threadIdx.x < stride)
            buf[threadIdx.x] += buf[threadIdx.x + stride];
        __syncthreads();
        stride /= 2;
    }

    if (threadIdx.x == 0)
        sm_var = buf[0] / H;

    __syncthreads();

    // normalize
    float inv_std = rsqrtf(sm_var + 1e-5f);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float x = input[row * H + i];
        float norm = (x - sm_mean) * inv_std;
        output[row * H + i] = gamma[i] * norm + beta[i];
    }
}


int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <block_size> <grid_size>\n", argv[0]);
        return 0;
    }

    int block_size = atoi(argv[1]);   // threads per block
    int grid_size  = atoi(argv[2]);   // number of rows processed

    // Input shape (内部指定)
    int N = grid_size;
    int H = 1024;   // hidden dimension

    printf("Running LayerNorm with N=%d, H=%d, block_size=%d, grid_size=%d\n",
        N, H, block_size, grid_size);

    size_t bytes = N * H * sizeof(float);
    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    float *h_gamma = (float*)malloc(H * sizeof(float));
    float *h_beta  = (float*)malloc(H * sizeof(float));

    // init
    for (int i = 0; i < N * H; i++) h_in[i] = (rand() % 1000) * 0.001f;
    for (int i = 0; i < H; i++) h_gamma[i] = 1.0f, h_beta[i] = 0.0f;

    float *d_in, *d_out, *d_gamma, *d_beta;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMalloc(&d_gamma, H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_beta,  H * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma, H*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta,  h_beta,  H*sizeof(float), cudaMemcpyHostToDevice));

    layernorm_kernel<<<grid_size, block_size>>>(d_in, d_out, d_gamma, d_beta, N, H);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    printf("LayerNorm done.\n");

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    free(h_in);
    free(h_out);
    free(h_gamma);
    free(h_beta);

    return 0;
}
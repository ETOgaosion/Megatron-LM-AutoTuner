#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) \
    do { cudaError_t err = x; if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); }} while(0)

// ======================================
// Simple matrix multiply C = A * B
// ======================================
__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}

int main(int argc, char** argv)
{
    if (argc < 7) {
        printf("Usage: ./matmul M N K block_x block_y grid_x grid_y\n");
        printf("Example: ./matmul 1024 1024 1024 16 16 64 64\n");
        return 0;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int block_x = atoi(argv[4]);
    int block_y = atoi(argv[5]);
    int grid_x  = atoi(argv[6]);
    int grid_y  = atoi(argv[7]);

    printf("M=%d N=%d K=%d\n", M, N, K);
    printf("block = (%d, %d)\n", block_x, block_y);
    printf("grid  = (%d, %d)\n", grid_x, grid_y);

    dim3 block(block_x, block_y);
    dim3 grid(grid_x, grid_y);

    // Allocate memory
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    float *A, *B, *C;
    CHECK_CUDA(cudaMalloc(&A, bytesA));
    CHECK_CUDA(cudaMalloc(&B, bytesB));
    CHECK_CUDA(cudaMalloc(&C, bytesC));

    // Init A/B
    float* hA = (float*)malloc(bytesA);
    float* hB = (float*)malloc(bytesB);

    for (int i = 0; i < M * K; i++) hA[i] = (float)(rand() % 5);
    for (int i = 0; i < K * N; i++) hB[i] = (float)(rand() % 5);

    CHECK_CUDA(cudaMemcpy(A, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B, hB, bytesB, cudaMemcpyHostToDevice));

    // Launch kernel
    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Matmul finished.\n");

    // Cleanup
    free(hA);
    free(hB);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
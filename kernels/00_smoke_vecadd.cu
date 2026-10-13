// 冒烟测试：验证 本地写代码 -> 同步 -> Slurm 编译运行 -> 拉回结果 整条链路
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while (0)

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const int n = 1 << 24;
    size_t bytes = n * sizeof(float);

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    float *hx = (float*)malloc(bytes), *hy = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) { hx[i] = 1.0f; hy[i] = 2.0f; }

    float *dx, *dy;
    CK(cudaMalloc(&dx, bytes)); CK(cudaMalloc(&dy, bytes));
    CK(cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dy, hy, bytes, cudaMemcpyHostToDevice));

    int threads = 256, blocks = (n + threads - 1) / threads;
    for (int it = 0; it < 5; ++it) saxpy<<<blocks, threads>>>(n, 2.0f, dx, dy);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    CK(cudaMemcpy(hy, dy, bytes, cudaMemcpyDeviceToHost));
    printf("y[0] = %.1f (expected 12.0) -> %s\n", hy[0], hy[0] == 12.0f ? "OK" : "FAIL");

    cudaFree(dx); cudaFree(dy); free(hx); free(hy);
    return 0;
}

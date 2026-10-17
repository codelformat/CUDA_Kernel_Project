// matmul v1：最朴素的矩阵乘 —— 一个 thread 负责 C 里的一个元素。
// 目标：先写对，再测量。性能是后面各版本要改进的基线。
#include "common.cuh"

// 线程块形状（可用 nvcc -DBX=.. -DBY=.. 覆盖）
#ifndef BX
#define BX 32
#endif
#ifndef BY
#define BY 8
#endif
// threadIdx.x 对应 C 的行还是列？0：x 对应列（相邻 thread 算同一行里相邻的列）；1：x 对应行
#ifndef X_IS_ROW
#define X_IS_ROW 0
#endif

// ============================== 你要写的部分 ==============================
// 行主序：A[i][k] 在 A[i * K + k]，B[k][j] 在 B[k * N + j]，C[i][j] 在 C[i * N + j]。
__global__ void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO 1: 算出这个 thread 负责 C 的哪一行 row、哪一列 col。
    //         用 blockIdx / blockDim / threadIdx 的 x、y 分量；
    //         X_IS_ROW == 0 时 x 方向对应列、y 方向对应行；X_IS_ROW == 1 时反过来。
    int row = 0, col = 0;
    if (X_IS_ROW) {
        row = blockIdx.x * blockDim.x + threadIdx.x;
        col = blockIdx.y * blockDim.y + threadIdx.y;
    } else {
        row = blockIdx.y * blockDim.y + threadIdx.y;
        col = blockIdx.x * blockDim.x + threadIdx.x;
    }

    // TODO 2: 越界保护（grid 是向上取整算出来的，M、N 不一定是 BX/BY 的整数倍）。
    if (row >= M || col >= N) return;

    // TODO 3: 沿 K 方向累加 sum += A[row][k] * B[k][col]，最后写 C[row][col]。
    float sum = 0.0f;
    for (int k = 0; k< K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
// ========================================================================

void launch(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(BX, BY);
    // grid 的 x 方向要覆盖 threadIdx.x 对应的那一维，y 同理
    dim3 grid(CDIV(X_IS_ROW ? M : N, BX), CDIV(X_IS_ROW ? N : M, BY));
    gemm_naive<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) {
    return run_gemm_benchmark("01_naive", launch, argc, argv);
}

// 张量核心阶段（阶段 B）的共用测试框架：
//   A、B 是 fp16（half），C 是 fp32；乘法在张量核心上做，累加用 fp32。
//   参考答案：cuBLAS cublasGemmEx（fp16 输入、fp32 累加、允许张量核心）。
//   另外顺便测一次同样数据的 cuBLAS SGEMM（fp32，CUDA core），用来直观对比"张量核心 vs CUDA core"。
//
// 约定（行主序 row-major）：A 是 M×K，B 是 K×N，C 是 M×N；C = A · B。
// 命令行:  <程序> [M N K]，默认 4096 4096 4096
// 环境变量: WARMUP(3) REPS(20) TOL 相对误差阈值（默认 1e-3：输入只有 fp16 精度，累加顺序也和 cuBLAS 不同）
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CKB(call) do { cublasStatus_t s_ = (call); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)
#define CDIV(a, b) (((a) + (b) - 1) / (b))

typedef void (*GemmTcFn)(const half* A, const half* B, float* C, int M, int N, int K);

static int env_int(const char* k, int d) { const char* v = getenv(k); return v ? atoi(v) : d; }
static double env_dbl(const char* k, double d) { const char* v = getenv(k); return v ? atof(v) : d; }

static void fill_random(std::vector<float>& v, uint32_t seed) {
    uint32_t s = seed;
    for (auto& x : v) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; x = (float)(s >> 8) * (2.0f / 16777216.0f) - 1.0f; }
}

template <class F>
static float bench_median_ms(F&& f, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) f();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<float> t(reps);
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a)); f(); CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&t[i], a, b));
    }
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    std::sort(t.begin(), t.end());
    return t[reps / 2];
}

static int run_gemm_tc_benchmark(const char* name, GemmTcFn fn, int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    else if (argc == 2) { M = N = K = atoi(argv[1]); }
    const double tol = env_dbl("TOL", 1e-3);
    const int warmup = env_int("WARMUP", 3), reps = env_int("REPS", 20);

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("== %s ==\n", name);
    if (const char* mid = getenv("MACHINE_ID")) printf("Machine: %s\n", mid);
    printf("Device: %s (sm_%d%d, %d SMs)   M=%d N=%d K=%d (A,B fp16 / C fp32, fp32 累加)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, M, N, K);

    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
    std::vector<float> fA(nA), fB(nB), hC(nC), hRef(nC);
    fill_random(fA, 1u); fill_random(fB, 2u);
    std::vector<half> hA(nA), hB(nB);
    for (size_t i = 0; i < nA; ++i) { hA[i] = __float2half(fA[i]); fA[i] = __half2float(hA[i]); }  // fp32 副本也用 fp16 舍入后的值，
    for (size_t i = 0; i < nB; ++i) { hB[i] = __float2half(fB[i]); fB[i] = __half2float(hB[i]); }  // 保证 SGEMM 对比的是同一组数

    half *dA, *dB; float *dC, *dRef, *dAf, *dBf, *dCf;
    CK(cudaMalloc(&dA, nA * sizeof(half))); CK(cudaMalloc(&dB, nB * sizeof(half)));
    CK(cudaMalloc(&dC, nC * sizeof(float))); CK(cudaMalloc(&dRef, nC * sizeof(float)));
    CK(cudaMalloc(&dAf, nA * sizeof(float))); CK(cudaMalloc(&dBf, nB * sizeof(float))); CK(cudaMalloc(&dCf, nC * sizeof(float)));
    CK(cudaMemcpy(dA, hA.data(), nA * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), nB * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dAf, fA.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBf, fB.data(), nB * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS 是列主序：行主序 C = A·B 等价于列主序 C^T = B^T·A^T，交换 A、B 传入即可（同 common.cuh）
    cublasHandle_t h; CKB(cublasCreate(&h));
    const float alpha = 1.f, beta = 0.f;
    auto run_ref = [&]() {
        CKB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, K,
                         &beta, dRef, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    };
    auto run_sgemm = [&]() {
        CKB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dBf, N, dAf, K, &beta, dCf, N));
    };
    run_ref(); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hRef.data(), dRef, nC * sizeof(float), cudaMemcpyDeviceToHost));

    CK(cudaMemset(dC, 0xFF, nC * sizeof(float)));   // 先填 NaN，没写到的位置会露馅
    fn(dA, dB, dC, M, N, K);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hC.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0; size_t worst = 0, n_nan = 0;
    for (size_t i = 0; i < nC; ++i) {
        max_ref = std::max(max_ref, (double)fabsf(hRef[i]));
        if (std::isnan(hC[i])) { ++n_nan; continue; }
        double d = fabs((double)hC[i] - (double)hRef[i]);
        if (d > max_err) { max_err = d; worst = i; }
    }
    const double rel = max_err / max_ref;
    const bool pass = (n_nan == 0) && (rel < tol);
    printf("max_abs_err = %.3e (row %zu, col %zu)   max|ref| = %.3f   rel = %.3e   nan = %zu   -> %s (tol %.0e)\n",
           max_err, worst / N, worst % N, max_ref, rel, n_nan, pass ? "PASS" : "FAIL", tol);

    const double flops = 2.0 * M * N * K;
    const float ms_ref = bench_median_ms(run_ref, warmup, reps);
    const float ms_sg = bench_median_ms(run_sgemm, warmup, reps);
    const float ms_me = bench_median_ms([&]() { fn(dA, dB, dC, M, N, K); }, warmup, reps);
    auto tf = [&](float ms) { return flops / (ms * 1e-3) / 1e12; };
    printf("cuBLAS SGEMM (fp32, CUDA core) : median %9.3f ms   %7.2f TFLOPS\n", ms_sg, tf(ms_sg));
    printf("cuBLAS GemmEx (fp16, 张量核心)  : median %9.3f ms   %7.2f TFLOPS   (%.2f× SGEMM)\n", ms_ref, tf(ms_ref), ms_sg / ms_ref);
    printf("%-30s : median %9.3f ms   %7.2f TFLOPS   (%5.1f%% of cuBLAS GemmEx)\n", name, ms_me, tf(ms_me), 100.0 * tf(ms_me) / tf(ms_ref));

    CKB(cublasDestroy(h));
    CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC)); CK(cudaFree(dRef));
    CK(cudaFree(dAf)); CK(cudaFree(dBf)); CK(cudaFree(dCf));
    return pass ? 0 : 1;
}

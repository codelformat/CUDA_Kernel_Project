// matmul 各版本共用的测试框架（host 代码）：
//   随机生成 A、B -> 用 cuBLAS 算参考答案 -> 调用你的 kernel 并校验 -> 热身 -> 计时 -> 报告 TFLOPS
// 每个 kernel 文件只需提供一个 launch 函数，再在 main 里调用 run_gemm_benchmark。
//
// 约定（行主序 row-major）：A 是 M×K，B 是 K×N，C 是 M×N；C = A · B。
// 命令行:  <程序> [M N K]   只给一个数就是 M=N=K=该数；默认 4096 4096 4096
// 环境变量: PEAK_TFLOPS 理论峰值(默认 120，来自 RTX PRO 6000 Blackwell Server 数据手册的 FP32 峰值)
//           WARMUP 热身次数(默认 3)   REPS 正式计时次数(默认 20)   TOL 相对误差阈值(默认 2e-5)
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)

#define CKB(call) do { cublasStatus_t s_ = (call); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__, __LINE__, (int)s_); \
    exit(1); } } while (0)

#define CDIV(a, b) (((a) + (b) - 1) / (b))

// launch 函数：在里面决定 grid / block 配置并启动 kernel。A、B、C 都是设备指针。
typedef void (*GemmFn)(const float* A, const float* B, float* C, int M, int N, int K);

static int env_int(const char* key, int def) {
    const char* v = getenv(key);
    return v ? atoi(v) : def;
}
static double env_dbl(const char* key, double def) {
    const char* v = getenv(key);
    return v ? atof(v) : def;
}

// xorshift32 生成 [-1, 1) 的均匀随机数，固定种子，结果可复现
static void fill_random(std::vector<float>& v, uint32_t seed) {
    uint32_t s = seed;
    for (auto& x : v) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        x = (float)(s >> 8) * (2.0f / 16777216.0f) - 1.0f;
    }
}

// 对 f() 计时：每次只用 event 围住一次调用，返回 reps 次的中位数（ms）
template <class F>
static float bench_median_ms(F&& f, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) f();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<float> t(reps);
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a));
        f();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&t[i], a, b));
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    std::sort(t.begin(), t.end());
    return t[reps / 2];
}

static int run_gemm_benchmark(const char* name, GemmFn fn, int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    else if (argc == 2) { M = N = K = atoi(argv[1]); }
    if (M <= 0 || N <= 0 || K <= 0) { fprintf(stderr, "M N K 必须是正整数\n"); return 1; }
    const double peak = env_dbl("PEAK_TFLOPS", 120.0);
    const double tol = env_dbl("TOL", 2e-5);
    const int warmup = env_int("WARMUP", 3), reps = env_int("REPS", 20);

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("== %s ==\n", name);
    printf("Device: %s (sm_%d%d, %d SMs)   M=%d N=%d K=%d (fp32)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, M, N, K);

    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
    std::vector<float> hA(nA), hB(nB), hC(nC), hRef(nC);
    fill_random(hA, 1u);
    fill_random(hB, 2u);

    float *dA, *dB, *dC, *dRef;
    CK(cudaMalloc(&dA, nA * sizeof(float)));
    CK(cudaMalloc(&dB, nB * sizeof(float)));
    CK(cudaMalloc(&dC, nC * sizeof(float)));
    CK(cudaMalloc(&dRef, nC * sizeof(float)));
    CK(cudaMemcpy(dA, hA.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), nB * sizeof(float), cudaMemcpyHostToDevice));

    // 参考答案：cuBLAS SGEMM（默认数学模式：fp32，不使用 TF32 张量核心）。
    // cuBLAS 是列主序。行主序的 C = A·B 等价于列主序的 C^T = B^T · A^T，
    // 把 B、A 当作列主序矩阵（B 看成 N×K，A 看成 K×M）交换顺序传入即可，结果 dRef 仍是行主序的 C。
    cublasHandle_t handle;
    CKB(cublasCreate(&handle));
    CKB(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    const float alpha = 1.f, beta = 0.f;
    auto run_cublas = [&]() {
        CKB(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dRef, N));
    };
    run_cublas();
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hRef.data(), dRef, nC * sizeof(float), cudaMemcpyDeviceToHost));

    // 你的 kernel：先把 C 填成 NaN（0xFF），没写到的位置会因此露馅
    CK(cudaMemset(dC, 0xFF, nC * sizeof(float)));
    fn(dA, dB, dC, M, N, K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hC.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));

    double max_err = 0.0, max_ref = 0.0;
    size_t worst = 0, n_nan = 0;
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

    // 计时
    const double flops = 2.0 * M * N * K;
    const float ms_ref = bench_median_ms(run_cublas, warmup, reps);
    const float ms_me = bench_median_ms([&]() { fn(dA, dB, dC, M, N, K); }, warmup, reps);
    const double tf_ref = flops / (ms_ref * 1e-3) / 1e12;
    const double tf_me = flops / (ms_me * 1e-3) / 1e12;
    printf("cuBLAS SGEMM : median %9.3f ms   %7.2f TFLOPS   (%5.1f%% of %.0f TFLOPS peak)\n",
           ms_ref, tf_ref, 100.0 * tf_ref / peak, peak);
    printf("%-13s: median %9.3f ms   %7.2f TFLOPS   (%5.1f%% of peak, %5.1f%% of cuBLAS)\n",
           name, ms_me, tf_me, 100.0 * tf_me / peak, 100.0 * tf_me / tf_ref);

    CKB(cublasDestroy(handle));
    CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC)); CK(cudaFree(dRef));
    return pass ? 0 : 1;
}

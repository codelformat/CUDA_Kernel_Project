// 所有版本共用的测试框架（host 代码）：
//   读数据 -> 拷到 GPU -> 第 0 次启动 + 正确性校验 -> 热身 10 次 -> 正式计时 100 次 -> 报告耗时与有效带宽
// 每个 kernel 文件只需要提供一个 launch 函数，再在 main 里调用 run_benchmark 即可。
// 命令行:  <程序> [数据目录, 默认 data] [threads_per_block, 默认 256]
// 环境变量: PEAK_GBPS 理论显存带宽(GB/s)，默认 1597（RTX PRO 6000 Blackwell Server 标称值）
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)

// launch 函数：在里面决定 grid / block 配置并启动 kernel。threads 来自命令行。
typedef void (*LaunchFn)(const float* x, const float* gamma, const float* beta, float* y,
                         int N, int H, float eps, int threads);

static std::vector<float> read_bin(const std::string& path, size_t count) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "打不开 %s\n  请先运行 python ref/gen_data.py 生成 data/\n", path.c_str());
        exit(1);
    }
    std::vector<float> v(count);
    size_t got = fread(v.data(), sizeof(float), count, f);
    fclose(f);
    if (got != count) {
        fprintf(stderr, "%s: 期望 %zu 个 float，实际读到 %zu 个\n", path.c_str(), count, got);
        exit(1);
    }
    return v;
}

static int run_benchmark(const char* name, LaunchFn launch, int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "data";
    const int threads = argc > 2 ? atoi(argv[2]) : 256;
    const char* pk = getenv("PEAK_GBPS");
    const double peak_gbps = pk ? atof(pk) : 1597.0;

    int N = 0, H = 0;
    float eps = 0.f;
    {
        FILE* f = fopen((dir + "/meta.txt").c_str(), "r");
        if (!f || fscanf(f, "%d %d %f", &N, &H, &eps) != 3) {
            fprintf(stderr, "读不到 %s/meta.txt\n", dir.c_str());
            return 1;
        }
        fclose(f);
    }
    const size_t nx = (size_t)N * H;

    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    printf("== %s ==\n", name);
    printf("Device: %s (sm_%d%d, %d SMs)   N=%d H=%d eps=%g   threads/block=%d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, N, H, eps, threads);
    int l2_bytes = 0;
    CK(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, dev));
    printf("L2 cache = %.0f MB\n", l2_bytes / 1048576.0);

    std::vector<float> hx = read_bin(dir + "/x.bin", nx);
    std::vector<float> hgamma = read_bin(dir + "/gamma.bin", H);
    std::vector<float> hbeta = read_bin(dir + "/beta.bin", H);
    std::vector<float> href = read_bin(dir + "/y_ref.bin", nx);
    std::vector<float> hy(nx, 0.f);

    float *d_x, *d_gamma, *d_beta, *d_y;
    CK(cudaMalloc(&d_x, nx * sizeof(float)));
    CK(cudaMalloc(&d_gamma, H * sizeof(float)));
    CK(cudaMalloc(&d_beta, H * sizeof(float)));
    CK(cudaMalloc(&d_y, nx * sizeof(float)));
    CK(cudaMemcpy(d_x, hx.data(), nx * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_gamma, hgamma.data(), H * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_beta, hbeta.data(), H * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_y, 0, nx * sizeof(float)));  // 清零：kernel 没写到的位置会因此露馅

    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));

    // ---- 第 0 次启动：既测冷启动耗时，也是用来校验正确性的那一次 ----
    CK(cudaEventRecord(start));
    launch(d_x, d_gamma, d_beta, d_y, N, H, eps, threads);
    CK(cudaEventRecord(stop));
    CK(cudaGetLastError());          // 启动参数是否合法
    CK(cudaEventSynchronize(stop));  // 等 kernel 跑完，并捕获运行期错误
    float first_ms = 0.f;
    CK(cudaEventElapsedTime(&first_ms, start, stop));
    printf("first launch = %.3f ms (含冷启动开销)\n", first_ms);

    CK(cudaMemcpy(hy.data(), d_y, nx * sizeof(float), cudaMemcpyDeviceToHost));
    float max_abs = 0.f;
    size_t worst = 0, n_nan = 0;
    for (size_t i = 0; i < nx; ++i) {
        if (std::isnan(hy[i])) { ++n_nan; continue; }
        float d = fabsf(hy[i] - href[i]);
        if (d > max_abs) { max_abs = d; worst = i; }
    }
    const float tol = 1e-5f;
    const bool pass = (n_nan == 0) && (max_abs < tol);
    printf("max_abs_err = %.3e (row %zu, col %zu)   nan = %zu   -> %s (tol %.0e)\n",
           max_abs, worst / H, worst % H, n_nan, pass ? "PASS" : "FAIL", tol);

    // ---- 热身 10 次，不计时 ----
    for (int i = 0; i < 10; ++i) launch(d_x, d_gamma, d_beta, d_y, N, H, eps, threads);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    // ---- 正式测量 100 次，每次单独计时（只围住 kernel 启动）----
    std::vector<float> times(100);
    for (int i = 0; i < 100; ++i) {
        CK(cudaEventRecord(start));
        launch(d_x, d_gamma, d_beta, d_y, N, H, eps, threads);
        CK(cudaEventRecord(stop));
        CK(cudaEventSynchronize(stop));
        CK(cudaEventElapsedTime(&times[i], start, stop));
    }
    CK(cudaGetLastError());
    std::sort(times.begin(), times.end());

    // 有效带宽 = 理想最少搬运字节数 / 耗时。最少字节 = x 读一次 + y 写一次（gamma/beta 太小，忽略）
    const double bytes = 2.0 * (double)N * (double)H * sizeof(float);
    auto gbps = [&](double ms) { return bytes / (ms * 1e6); };
    printf("median = %.3f ms  (%.1f GB/s, %.1f%% of %.0f GB/s peak)\n",
           times[50], gbps(times[50]), 100.0 * gbps(times[50]) / peak_gbps, peak_gbps);
    printf("min    = %.3f ms  (%.1f GB/s, %.1f%% of %.0f GB/s peak)\n",
           times[0], gbps(times[0]), 100.0 * gbps(times[0]) / peak_gbps, peak_gbps);

    CK(cudaEventDestroy(start));
    CK(cudaEventDestroy(stop));
    cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_y);
    return pass ? 0 : 1;
}

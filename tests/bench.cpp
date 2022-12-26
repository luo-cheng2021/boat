#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <gflags/gflags.h>
#include <boat.h>

using namespace boat;
// inner product sample:
// --fix-times-per-prb=10000 --mode=p --matmul --reset --allow-enum-tags-only=0 --engine=cpu --dir=FWD_B  --cfg=f32
//     --stag=ab --wtag=ab --dtag=ab  --attr-scratchpad=user mb15ic512oc37
DEFINE_int32(fix_times_per_prb, 1, "running times");
DEFINE_bool(matmul, true, "inner product testing");

using Ms = std::chrono::duration<double, std::ratio<1, 1000>>;

void test_ip(const char* param) {
    int M, N, K;
    if (sscanf(param, "mb%dic%doc%d", &M, &K, &N) != 3) {
        std::cout << "param format is wrong: " << param << "\n";
        return;
    }
    matmul gemm;
    GemmDynMStaticParam gemmParam = {
        dnnl_f32, dnnl_f32, dnnl_f32,
        N, K, K * 4, N * 4, N * 4
    };
    if (!gemm.init(gemmParam)) {
        std::cout << "init ip failed with:" << param << "\n";
        return;
    }
    std::vector<float> a(M * K, 2), b(K * N, 1), c(M * N);
    std::iota(a.begin(), a.end(), 1.0f);
    std::iota(b.begin(), b.end(), 2.0f);
    GemmDynMRuntimeParam rtParam = {
        M, a.data(), b.data(), c.data()
    };

    gemm(rtParam);
    double total = 0, min_time = 1000000.0f;
    for (int i = 0; i < FLAGS_fix_times_per_prb; i++) {
        auto start = std::chrono::steady_clock::now();
        gemm(rtParam);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<Ms>(end - start).count();
        total += duration;
        if (duration < min_time)
            min_time = duration;
    }
    printf("%s:\ntotal perf: min(ms): %.4f avg(ms): %.4f\n", param, min_time, total / FLAGS_fix_times_per_prb);
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_matmul) {
        if (argc < 2) {
            std::cout << "command line format is wrong\n";
            return -1;
        }
        test_ip(argv[argc - 1]);
    }

    return 0;
}
#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iostream>
#include "gtest/gtest.h"
#include "boat.h"

using namespace std;

class GemmTest : public ::testing::Test {

protected:

  virtual void SetUp() {
  };

  virtual void TearDown() {
  };

  virtual void verify(int index) {
    EXPECT_EQ(index + 1, 1);
  }
};

TEST_F(GemmTest, Case1) {
  verify(0);
}

static void matmul_ref(float* a, float* b, float* c, int M, int N, int K, int lda, int ldb, int ldc, float* ops = nullptr) {
#define A(i, j) a[(j) + (i) * lda]
#define B(i, j) b[(j) + (i) * ldb]
#define C(i, j) c[(j) + (i) * ldc]

    int i, j, p;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            C(i, j) = ops == nullptr ? 0 : ops[j];//j; // post ops, per-channel
            for (p = 0; p < K; p++) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
}

void test_stride_func(int M, int N, int K) {
    // postops, perchannel
    std::vector<float> d(N, 0);
    std::iota(d.begin(), d.end(), 10000000.0f);
    d[1] += d[0];
    d[2] += d[1];

    std::vector<float> a(M * K, 2), b(K * N, 1), c(M * N), c_ref(M * N);
    std::iota(a.begin(), a.end(), 1.0f);
    std::iota(b.begin(), b.end(), 2.0f);

    Jit::PostOpStaticParams post_ops;
    post_ops.num = 2;
    post_ops.ops[0].alg_type = Jit::AlgType::Abs;
    post_ops.ops[1].alg_type = Jit::AlgType::Add;
    post_ops.ops[1].binary_param.layout = Jit::BinaryDataLayout::PerChannel;
    auto f = Jit::kernel<16>::make_gemm_stride(N, K, K * 4, N * 4, N * 4, &post_ops);
    Jit::PostOpRuntimeParams ops;
    ops.params[1].right_addr = d.data();

    f(M, a.data(), b.data(), c.data(), &ops);
    matmul_ref(a.data(), b.data(), c_ref.data(), M, N, K, K, N, N, d.data());
    if (c == c_ref) {
        printf("M %d, N %d, K %d: correct \n", M, N, K);
    }
    else {
        bool error = false;
        for (int i = 0; i < (int)c.size(); i++) {
            if (std::abs(c[i] - c_ref[i]) > 0.00001f * std::abs(c[i])) {
                error = true;
                printf("first error at %d, cur %f ref %f\n", i, c[i], c_ref[i]);
                break;
            }
        }
        if (error)
            printf("M %d, N %d, K %d: wrong result\n", M, N, K);
        else
            printf("M %d, N %d, K %d: correct with minor error\n", M, N, K);
    }
    Jit::delete_func(Jit::cast(f));
}

void test_stride_func_dyn_m(int M, int N, int K) {
    Jit::PostOpStaticParams post_ops;
    post_ops.num = 0;
    post_ops.ops[0].alg_type = Jit::AlgType::Abs;
    post_ops.ops[1].alg_type = Jit::AlgType::Add;
    post_ops.ops[1].binary_param.layout = Jit::BinaryDataLayout::PerChannel;
    auto f = Jit::kernel<16>::make_gemm_stride(N, K, K * 4, N * 4, N * 4, &post_ops);
    Jit::PostOpRuntimeParams ops;

    // postops, perchannel
    std::vector<float> d(N, 0);
    std::iota(d.begin(), d.end(), 0.0f);

    std::vector<float> b(K * N, 1);
    std::iota(b.begin(), b.end(), 2.0f);
    ops.params[1].right_addr = d.data();

    for (int m = M; m < 3 * M; m += M) {
        std::vector<float> a(m * K, 2), c(m * N), c_ref(m * N);
        std::iota(a.begin(), a.end(), 1.0f);

        f(m, a.data(), b.data(), c.data(), &ops);
        matmul_ref(a.data(), b.data(), c_ref.data(), m, N, K, K, N, N, d.data());
        if (c == c_ref) {
            printf("M %d, N %d, K %d: correct \n", m, N, K);
        }
        else {
            bool error = false;
            for (int i = 0; i < (int)c.size(); i++) {
                if (std::abs(c[i] - c_ref[i]) > 0.00001f * std::abs(c[i])) {
                    error = true;
                    printf("first error at %d, cur %f ref %f\n", i, c[i], c_ref[i]);
                    break;
                }
            }
            if (error)
                printf("M %d, N %d, K %d: wrong result\n", m, N, K);
            else
                printf("M %d, N %d, K %d: correct with minor error\n", m, N, K);
        }
    }
    Jit::delete_func(Jit::cast(f));
}

void test_stride_funcs() {
    // normal
    test_stride_func(256, 48, 448);
    // k tail
    test_stride_func(256, 48, 449);
    // M tail == unroll 8
    test_stride_func(256 + 8, 48, 449);
    // M tail == unroll 8 + 2
    test_stride_func(256 + 10, 48, 449);
    // N tail
    test_stride_func(256, 40, 448);
    // all tail
    test_stride_func(256 + 9, 47, 449);
    // dyn M
    test_stride_func_dyn_m(256 + 9, 38, 449);
}

TEST(GemmTestFunc, Func) {
    //FAIL() << "Expected divide() method to throw DivisionByZeroException";
    test_stride_funcs();
}

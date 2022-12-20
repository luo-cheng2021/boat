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

class GemmDriverTest : public ::testing::Test {

protected:
    virtual void SetUp() {
    };

    virtual void TearDown() {
    };

    virtual void verify(int index) {
        EXPECT_EQ(index + 1, 1);
    }
};

TEST_F(GemmDriverTest, Case1) {
    Jit::driver::test();
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

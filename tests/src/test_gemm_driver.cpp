#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iostream>
#include "gtest/gtest.h"
#include "boat.h"
#include "test_gemm_common.h"

using namespace std;
using namespace boat;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

using GemmDriverTestParamSet = std::tuple<
        int,                                         // M
        int,                                         // N
        int                                          // K
        >;

class GemmDriverTest : public TestWithParam<GemmDriverTestParamSet> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GemmDriverTestParamSet>& obj) {
        int M, N, K;
        std::tie(M, N, K) = obj.param;

        std::ostringstream result;
        result << "M_" << M << "_N_" << N << "_K_" << K;
        return result.str();
    }

protected:
    virtual void SetUp() {
        auto [M, N, K] = GetParam();
        _M = M; _N = N; _K = K;
        GemmDynMStaticParam param = {
            dnnl_f32, dnnl_f32, dnnl_f32,
            N, K, K * 4, N * 4, N * 4
        };
        PostOpStaticParams& post_ops = param.post_static_params;
        post_ops.num = 2;
        post_ops.ops[0].alg_type = AlgType::Abs;
        post_ops.ops[1].alg_type = AlgType::Add;
        post_ops.ops[1].binary_param.layout = BinaryDataLayout::PerChannel;
        EXPECT_TRUE(_gemm.init(param));
    };

    virtual void TearDown() {
    };

    virtual void verify(int index) {
        EXPECT_EQ(index + 1, 1);
    }
    gemm_driver _gemm;
    int _M, _N, _K;
};

TEST_P(GemmDriverTest, Normal) {
    std::vector<float> d(_N, 0);
    std::iota(d.begin(), d.end(), 10000000.0f);
    d[1] += d[0];
    d[2] += d[1];

    std::vector<float> a(_M * _K, 2), b(_K * _N, 1), c(_M * _N), c_ref(_M * _N);
    std::iota(a.begin(), a.end(), 1.0f);
    std::iota(b.begin(), b.end(), 2.0f);
    GemmDynMRuntimeParam rtParam = {
        _M, a.data(), b.data(), c.data()
    };
    PostOpRuntimeParams& ops = rtParam.post_runtime_params;    
    ops.params[1].right_addr = d.data();

    _gemm(rtParam);
    matmul_ref(a.data(), b.data(), c_ref.data(), _M, _N, _K, _K, _N, _N, d.data());
    int status = 1;
    if (c == c_ref) {
        status = 0;
    }
    else {
        for (int i = 0; i < (int)c.size(); i++) {
            if (std::abs(c[i] - c_ref[i]) > 0.00001f * std::abs(c[i])) {
                status = -1;
                printf("first error at %d, cur %f ref %f\n", i, c[i], c_ref[i]);
                break;
            }
        }
        if (status == 1)
            printf("M %d, N %d, K %d: correct with minor error\n", _M, _N, _K);
    }
    EXPECT_TRUE(status >= 0);
}

TEST_P(GemmDriverTest, DynM) {
    std::vector<float> d(_N, 0);
    std::iota(d.begin(), d.end(), 10000000.0f);
    d[1] += d[0];
    d[2] += d[1];

    std::vector<float> b(_K * _N, 1);
    std::iota(b.begin(), b.end(), 2.0f);
    GemmDynMRuntimeParam rtParam;
    PostOpRuntimeParams& ops = rtParam.post_runtime_params;
    ops.params[1].right_addr = d.data();

    for (int m = _M; m < 3 * _M; m += _M) {
        std::vector<float> a(m * _K, 2), c(m * _N), c_ref(m * _N);
        std::iota(a.begin(), a.end(), 1.0f);
        rtParam.m = m;
        rtParam.a = a.data();
        rtParam.b = b.data();
        rtParam.c = c.data();
        _gemm(rtParam);
        matmul_ref(a.data(), b.data(), c_ref.data(), m, _N, _K, _K, _N, _N, d.data());
        int status = 1;
        if (c == c_ref) {
            status = 0;
        }
        else {
            for (int i = 0; i < (int)c.size(); i++) {
                if (std::abs(c[i] - c_ref[i]) > 0.00001f * std::abs(c[i])) {
                    status = -1;
                    printf("first error at %d, cur %f ref %f\n", i, c[i], c_ref[i]);
                    break;
                }
            }
            if (status == 1)
                printf("M %d, N %d, K %d: correct with minor error\n", m, _N, _K);
        }
        EXPECT_TRUE(status >= 0);
    }
}

static std::vector<int> Ms = {
    128, 129, 254, 499, 2048
};

static std::vector<int> Ns = {
    12, 19, 34, 55
};

static std::vector<int> Ks = {
    134, 255, 666, 888, 1025
};

const auto kernelCase = ::testing::Combine(
    ValuesIn(Ms),
    ValuesIn(Ns),
    ValuesIn(Ks)
);
INSTANTIATE_TEST_SUITE_P(smoke_GemmDriver, GemmDriverTest, kernelCase, GemmDriverTest::getTestCaseName);

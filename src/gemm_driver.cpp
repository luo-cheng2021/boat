#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iostream>

#include "dnnl_thread.hpp"
#include "tool.h"
#include "boat.h"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;

namespace boat {

struct gemm_driver::gemm_driver_impl {
    std::unordered_map<int, gemm_kernel<cpu_isa_t::avx512_core>> _kernels;
    int _nthread = 0;
    int _N_block_num = 0;
    int _N_block = 0;
    int _N_block_tail = 0;
    unsigned int _L2;
    GemmDynMStaticParam _dynMStaticParam;

    gemm_driver_impl() {
        _L2 = getDataCacheSize(2);
    }

    int get_N_block(const GemmDynMStaticParam& static_param) {
        if (static_param.N <= 64) return static_param.N;

        auto N = (static_param.N + 15) / 16 * 16;
        if (N % 64 == 0 && N % 48 != 0)
            return 64;

        return 48;
    }

    bool init(const GemmDynMStaticParam& static_param) {
        _nthread = dnnl_get_max_threads();
        auto N = static_param.N;
        _N_block = get_N_block(static_param);
        GemmDynMStaticParam param = static_param;
        param.N = _N_block;
        if (!_kernels[_N_block].init(param))
            return false;
        if (N % _N_block) {
            _N_block_tail = N % _N_block;
            param.N = _N_block_tail;
            if (!_kernels[_N_block_tail].init(param))
                return false;
        }
        _N_block_num = (N + _N_block - 1) / _N_block;
        _dynMStaticParam = static_param;
        return true;
    }

    int get_M_block(int M) {
        auto N_block = _N_block ? _N_block : _N_block_tail;
        auto B_size = N_block * _dynMStaticParam.K;
        auto AC_line_size = (_dynMStaticParam.K + N_block);
        auto AC_lines = static_cast<int>((_L2 - B_size) / sizeof(float) * 8 / AC_line_size / 10);

        return std::min(M, AC_lines);
    }

    void init_postops_offset(int ocb, int n_block, GemmDynMRuntimeParam &param, const GemmDynMRuntimeParam& orgParam) {
        for (int i = 0; i < _dynMStaticParam.post_static_params.num; i++) {
            if (_dynMStaticParam.post_static_params.ops[i].alg_type >= AlgType::Add &&
                _dynMStaticParam.post_static_params.ops[i].binary_param.layout == BinaryDataLayout::PerChannel) {
                param.post_runtime_params.params[i].right_addr = orgParam.post_runtime_params.params[i].right_addr + ocb * n_block;
            }
        }
    }

    void exec(const GemmDynMRuntimeParam& runtime_param) {
        auto M = get_M_block(runtime_param.m);
        auto M_tail = runtime_param.m % M;
        auto M_block = (runtime_param.m + M - 1) / M;
        int work_amount = M_block * _N_block_num;

        parallel(_nthread, [&](const int ithr, const int nthr) {
            if (ithr >= work_amount) return;

            GemmDynMRuntimeParam param = runtime_param;
            int start, end;
            balance211(work_amount, nthr, ithr, start, end);
            int ocb {0}, osb {0};
            nd_iterator_init(start, osb, M_block, ocb, _N_block_num);
            while (start++ < end) {
                init_postops_offset(ocb, _N_block, param, runtime_param);
                param.a = static_cast<uint8_t*>(runtime_param.a) + osb * M * _dynMStaticParam.lda;
                param.b = static_cast<uint8_t*>(runtime_param.b) + ocb * _N_block * sizeof(float);
                param.c = static_cast<uint8_t*>(runtime_param.c) + osb * M * _dynMStaticParam.ldc + ocb * _N_block * sizeof(float);
                if (osb == M_block - 1 && M_tail)
                    param.m = M_tail;
                else
                    param.m = M;
                if (ocb == _N_block_num - 1 && _N_block_tail)
                    _kernels[_N_block_tail](param);
                else
                    _kernels[_N_block](param);

                nd_iterator_step(osb, M_block, ocb, _N_block_num);
            }
        });
    }
    ~gemm_driver_impl() {

    }
};

gemm_driver::gemm_driver() :
    _impl(std::make_shared<gemm_driver_impl>()) {
}

bool gemm_driver::init(const GemmDynMStaticParam& static_param) {
    return _impl->init(static_param);
}

void gemm_driver::operator()(const GemmDynMRuntimeParam& runtime_param) {
    _impl->exec(runtime_param);
}


}
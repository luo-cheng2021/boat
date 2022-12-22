#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iostream>

#include "dnnl_thread.hpp"
#include "boat.h"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;

namespace boat {

struct gemm_driver::gemm_driver_impl {
    gemm_kernel<cpu_isa_t::avx512_core> _kernel;

    gemm_driver_impl() {
    }
    bool init(const GemmDynMStaticParam& static_param) {
        return _kernel.init(static_param);
    }
    void exec(const GemmDynMRuntimeParam& runtime_param) {
        int work_amount = 1;
        parallel(1, [&](const int ithr, const int nthr) {
            if (ithr >= work_amount) return;

            int start, end;
            balance211(work_amount, nthr, ithr, start, end);
            int n {0}, g {0}, ocb {0}, odb {0}, ohb {0}, owb {0};
            nd_iterator_init(start, n, 10, odb, 10, ohb, 10,
                    owb, 10, g, 1, ocb, 10);
            int N = 40, K = 1;

            _kernel(runtime_param);
            nd_iterator_step(n, 10, odb, 10, ohb, 10, owb,
                    10, g, 10, ocb, 10);
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
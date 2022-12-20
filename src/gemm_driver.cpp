#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iostream>

#include "dnnl_thread.hpp"
#include "boat.h"
using namespace Jit;
using namespace dnnl::impl;
using namespace dnnl::impl::utils;

void driver::test() {
    int work_amount = 1;
    parallel(0, [&](const int ithr, const int nthr) {
        if (ithr >= work_amount) return;

        int start, end;
        balance211(work_amount, nthr, ithr, start, end);
        int n {0}, g {0}, ocb {0}, odb {0}, ohb {0}, owb {0};
        nd_iterator_init(start, n, 10, odb, 10, ohb, 10,
                owb, 10, g, 1, ocb, 10);
        int N = 1, K = 1;
        Jit::PostOpStaticParams post_ops;

        auto f = Jit::kernel<16>::make_gemm_stride(N, K, K * 4, N * 4, N * 4, &post_ops);

        nd_iterator_step(n, 10, odb, 10, ohb, 10, owb,
                10, g, 10, ocb, 10);
    });

    return ;
}

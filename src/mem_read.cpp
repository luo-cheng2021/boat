#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iostream>
#include <memory.h>
#include <oneapi/tbb.h>
#include <sched.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>

#include "dnnl_thread.hpp"
#include "tool.h"
#include "boat.h"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;

namespace boat {

class pinning_observer : public oneapi::tbb::task_scheduler_observer {
public:
    pinning_observer() {
        observe(true); // activate the observer
    }
    //affinity_mask_t m_mask; // HW affinity mask to be used for threads in an arena
    pinning_observer(oneapi::tbb::task_arena &a)
        : oneapi::tbb::task_scheduler_observer(a) {
        observe(true); // activate the observer
    }
    void on_scheduler_entry( bool worker ) override {
        //set_thread_affinity(oneapi::tbb::this_task_arena::current_thread_index(), m_mask);
        cpu_set_t my_set;        /* Define your cpu_set bit mask. */
        CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */
        CPU_SET(oneapi::tbb::this_task_arena::current_thread_index(), &my_set);     /* set the bit that represents core 7. */
        sched_setaffinity(0, sizeof(cpu_set_t), &my_set); /* Set affinity of tihs process to */

        char buf[512];
        sprintf(buf, "id=%d, tid=%d\n", oneapi::tbb::this_task_arena::current_thread_index(), syscall(__NR_gettid));
        //printf(buf);
    }
    void on_scheduler_exit( bool worker ) override {
        //restore_thread_affinity();
    }
};

struct mem_read::mem_read_impl {
    mem_read_kernel<cpu_isa_t::avx512_core> _kernels;
    template<typename T>
    using deleted_unique_ptr = std::unique_ptr<T,std::function<void(T*)>>;
    std::vector<deleted_unique_ptr<int8_t>> _bufs;
    int _nthread = 0;
    unsigned int _L2;
    std::shared_ptr<pinning_observer> _pin;

    mem_read_impl() {
        _L2 = getDataCacheSize(2);
    }

    bool init(int size) {
        //_pin = std::make_shared<pinning_observer>();
        _nthread = dnnl_get_max_threads();
        _kernels.init(size);
        _bufs.resize(_nthread);
        parallel(_nthread, [&](const int ithr, const int nthr) {
            _bufs[ithr] = std::move(deleted_unique_ptr<int8_t>(
                        reinterpret_cast<int8_t*>(aligned_alloc(64, size)),
                        [](void * p) { ::free(p); }));
            memset(_bufs[ithr].get(), 1, size);
        });

        return true;
    }

    void exec(int times) {

        parallel(_nthread, [&](const int ithr, const int nthr) {
            for (int i = 0; i < times; i++)
                (_kernels)(_bufs[ithr].get());
        });
    }
    ~mem_read_impl() {

    }
};

mem_read::mem_read() :
    _impl(std::make_shared<mem_read_impl>()) {
}

bool mem_read::init(int size) {
    return _impl->init(size);
}

void mem_read::operator()(int times) {
    _impl->exec(times);
}


}
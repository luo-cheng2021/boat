#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iostream>
#include <assert.h>

#include <coat/Function.h>
#include <coat/ControlFlow.h>
#include <coat/Vec.h>
#include <coat/Mask.h>
#include "boat.h"
#include "gdb/gdbjit.h"

#define ENABLE_DUMP 0

namespace boat {

// size should be runtime const
template<int vectorsize>
void jit_memread0(coat::Ptr<coat::Value<float>> p, int size) {
    int offset = 0;
    size /= sizeof(float);
    if (size > vectorsize * (int)sizeof(float)) {
        coat::Vec<float, vectorsize> zero(false, "zero");
        const int size_4 = size / vectorsize / sizeof(float) / 4 * 4 * sizeof(float) * vectorsize;
        if (size_4 > 2 * 4 * (int)sizeof(float) * vectorsize) {
            coat::Value pos(int(0), "pos");
            coat::loop_while(pos < size_4, [&] {
                // p[pos + 1 * vectorsize * sizeof(float)]
                zero.load(p.index(pos, 0 * vectorsize * sizeof(float)));
                zero.load(p.index(pos, 1 * vectorsize * sizeof(float)));
                zero.load(p.index(pos, 2 * vectorsize * sizeof(float)));
                zero.load(p.index(pos, 3 * vectorsize * sizeof(float)));
                pos += 4 * vectorsize;
            });
            offset += size_4;
        }
        for (; offset < (int)(size / (sizeof(float) * vectorsize) * sizeof(float) * vectorsize); offset += sizeof(float) * vectorsize) {
            zero.load(p[offset]);
        }
    }
}

using func_t = void (*)(int8_t* a);
template <unsigned width>
static func_t make_func(int size) {
    auto fn = coat::createFunction<func_t>("readtest");
    if constexpr (width == 16)
        fn.funcNode->frame().setAvx512Enabled();
    else if  constexpr (width == 8)
        fn.funcNode->frame().setAvxEnabled();
#if ENABLE_DUMP
    fn.enableCodeDump();
#endif
    {
        auto [j_a_] = fn.getArguments("a");
        auto j_a = j_a_.cast<float>();
        jit_memread0<width>(j_a, size);
        // specify return value
        coat::ret();
    }

    // finalize code generation and get function pointer to the generated function
    auto foo = fn.finalize();
    return foo;
}

template <cpu_isa_t isa>
struct mem_read_kernel<isa>::mem_read_kernel_impl {
    func_t _func;
    mem_read_kernel_impl() : _func(nullptr) {
    }
    bool init(int size) {
        if constexpr (static_cast<unsigned>(isa) & avx512_core_bit)
            _func = make_func<16>(size);
        
        static int8_t a[102400000];
        _func(a);
        return _func != nullptr;
    }
    ~mem_read_kernel_impl() {
        if (_func)
            coat::getJitRuntimeEnv().release_func(_func);
    }
};

template <cpu_isa_t isa>
mem_read_kernel<isa>::mem_read_kernel() :
    _impl(std::make_shared<mem_read_kernel_impl>()) {
}

template <cpu_isa_t isa>
bool mem_read_kernel<isa>::init(int size) {
    return _impl->init(size);
}

template <cpu_isa_t isa>
void mem_read_kernel<isa>::operator()(int8_t* buf) {
    assert(_impl->_func);
    _impl->_func(buf);
}

template struct mem_read_kernel<cpu_isa_t::avx512_core>;

};

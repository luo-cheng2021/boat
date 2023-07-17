#pragma once

#include <vector>
#include <thread>
#include <assert.h>
#include <asmjit/asmjit.h>
#include "gdb/gdbjit.h"
#include "perfcompiler.h"

namespace coat {

struct JitRuntimeEnv {
    asmjit::JitRuntime rt;
    template<typename Func>
    void release_func(Func p) {
        rt.release(p);
    }
};

inline JitRuntimeEnv& getJitRuntimeEnv() {
   static JitRuntimeEnv env;

   return env;
}

struct ThreadCompilerContext {
#if 1 //def PROFILING_SOURCE
    PerfCompiler cc;
    GDBJit gdb;
#else
    asmjit::x86::Compiler cc;
#endif
};

// in order to avoid pass cc anywhere make it a global
inline ThreadCompilerContext& getCcContext() {
   static thread_local ThreadCompilerContext g;

   return g;
}

// helper macro
#define _CC ::coat::getCcContext().cc
#define NONCOPYABLE(Type) Type(const Type&)=delete; Type& operator=(const Type&)=delete

}
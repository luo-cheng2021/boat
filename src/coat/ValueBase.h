#pragma once

#include "Global.h"

namespace coat {

struct ValueBase {
    asmjit::x86::Gp reg;

    ValueBase() {}
    ValueBase(asmjit::x86::Gp reg) : reg(reg) {}
    //ValueBase(const ValueBase& other) : reg(other.reg) {}
    ValueBase(const ValueBase&& other) : reg(other.reg) {}

    operator const asmjit::x86::Gp&() const { return reg; }
    operator       asmjit::x86::Gp&()       { return reg; }
};


// identical operations for signed and unsigned, or different sizes
// pre-increment, post-increment not supported as it leads to temporary
// free-standing functions as we need "this" as explicit parameter, stupidly called "other" here because of macros...
inline ValueBase& operator++(const D<ValueBase>& other) {
    _CC.inc(OP.reg);
#ifdef PROFILING_SOURCE
    _CC.attachDebugLine(other.file, other.line);
#endif
    return const_cast<ValueBase&>(OP); //HACK
}
// pre-decrement
inline ValueBase& operator--(const D<ValueBase>& other) {
    _CC.dec(OP.reg);
#ifdef PROFILING_SOURCE
    _CC.attachDebugLine(other.file, other.line);
#endif
    return const_cast<ValueBase&>(OP); //HACK
}

} // namespace

#pragma once

#include <array>

namespace Jit {
// post ops setting params, interface
enum class AlgType {
    // Unary: x = f(x)
    Abs,
    // Binary: x = f(x, y), x and y are variable
    Add,
    Sub,
    Mul,
    ReLU,
    BatchNorm,
    // BinaryConst: x = f(x, c1, c2), x is varible and c1/c2 is const
    Add_C,
    Sub_C,
    Mul_C,
};

struct UnaryStaticParam {
    float x1;
    float x2;
    float x3;
    float x4;
};

enum class BinaryDataLayout {
    PerTensor,
    PerChannel,
    PerElement
};
struct BinaryStaticParam {
    BinaryDataLayout layout;
};

struct PostOpStaticParam {
    AlgType alg_type;
    union {
        UnaryStaticParam unary_param;
        BinaryStaticParam binary_param;
    };
};

#define MAX_POSTOPS_NUM 10
struct PostOpStaticParams {
    int num = 0;
    PostOpStaticParam ops[MAX_POSTOPS_NUM];
};

////////////////////////////////////////////////////
#ifndef COAT_NAME
#define COAT_NAME(str) public: static constexpr const char* name = str

#define COAT_STRUCT_MEMBER(ty, id) ty id;
#define COAT_ENUM_MEMBER(ty, id) member_##id,
#define COAT_STRING_MEMBER(ty, id) #id,
#define COAT_TYPE_MEMBER(ty, id) ty,

// declares (public) members, enum and tuple containing types
#define COAT_DECLARE(members)                  \
    members(COAT_STRUCT_MEMBER)                \
    enum member_ids : int {                    \
        members(COAT_ENUM_MEMBER)              \
    };                                         \
    static constexpr std::array member_names { \
        members(COAT_STRING_MEMBER)            \
    };                                         \
    using types = std::tuple<                  \
        members(COAT_TYPE_MEMBER)              \
    void>;

// declares private members and public enum and types
#define COAT_DECLARE_PRIVATE(members)          \
public:                                        \
    members(COAT_STRUCT_MEMBER)                \
public:                                        \
    enum member_ids : int {                    \
        members(COAT_ENUM_MEMBER)              \
    };                                         \
    static constexpr std::array member_names { \
        members(COAT_STRING_MEMBER)            \
    };                                         \
    using types = std::tuple<                  \
        members(COAT_TYPE_MEMBER)              \
    void>;

#endif
// jit kernel param when calling kernels
struct PostOpRuntimeParam {
    COAT_NAME("PostOpRuntimeParam");
    #define MEMBERS(x)    \
        x(float*, right_addr)

    COAT_DECLARE_PRIVATE(MEMBERS)
    #undef MEMBERS
    // int8_t* right_addr; // second param address
};

// array should use alias to workaround macro
using JitParamArr = PostOpRuntimeParam[MAX_POSTOPS_NUM];
struct PostOpRuntimeParams {
    COAT_NAME("PostOpRuntimeParams");
    #define MEMBERS(x)    \
        x(JitParamArr, params)

    COAT_DECLARE_PRIVATE(MEMBERS)
    #undef MEMBERS
    // PostOpRuntimeParam params[MAX_POSTOPS_NUM];
};

using func_m_t = void (*)(int m, float* a, float* b, float* c, Jit::PostOpRuntimeParams* post_runtime_params);
template <unsigned width>
struct gemm {
    static func_m_t make_gemm_stride(int N, int K, int lda, int ldb, int ldc, Jit::PostOpStaticParams* post_static_params, const int ur_num = 8, const int oc_num = 3);
};

template<typename T>
void *cast(T *ptr) {
    return reinterpret_cast<void *>(ptr);
}

void delete_func(void *p);
};

#pragma once

#include <array>
#include <memory>

namespace boat {
// copy from oneDNN
// Maximum number of features + hints that can be specified via bits
static constexpr int cpu_isa_total_bits = sizeof(unsigned) * 8;

enum cpu_isa_bit_t : unsigned {
    // Fill in features from least significant bit to most significant bit
    sse41_bit = 1u << 0,
    avx_bit = 1u << 1,
    avx2_bit = 1u << 2,
    avx512_core_bit = 1u << 6,
    avx512_core_vnni_bit = 1u << 7,
    avx512_core_bf16_bit = 1u << 8,
    amx_tile_bit = 1u << 9,
    amx_int8_bit = 1u << 10,
    amx_bf16_bit = 1u << 11,
    avx_vnni_bit = 1u << 12,
    avx512_core_fp16_bit = 1u << 13,

    // Fill in hints from most significant bit to least significant bit
    prefer_ymm_bit = 1u << (cpu_isa_total_bits - 1),
};

static constexpr unsigned hints_mask = prefer_ymm_bit;

enum class cpu_isa_t : unsigned {
    isa_any = 0u,
    sse41 = sse41_bit,
    avx = avx_bit | sse41,
    avx2 = avx2_bit | avx,
    avx_vnni = avx_vnni_bit | avx_bit,
    avx2_vnni = avx_vnni | avx2,
    avx512_core = avx512_core_bit | avx2,
    avx512_core_vnni = avx512_core_vnni_bit | avx512_core,
    avx512_core_bf16 = avx512_core_bf16_bit | avx512_core_vnni,
    avx512_core_bf16_ymm = prefer_ymm_bit | avx512_core_bf16,
    amx_tile = amx_tile_bit,
    amx_int8 = amx_int8_bit | amx_tile,
    amx_bf16 = amx_bf16_bit | amx_tile,
    avx512_core_bf16_amx_int8 = avx512_core_bf16 | amx_int8,
    avx512_core_bf16_amx_bf16 = avx512_core_bf16 | amx_bf16,
    avx512_core_fp16 = avx512_core_fp16_bit | avx512_core_bf16,
    avx512_core_amx = avx512_core_fp16 | amx_int8 | amx_bf16,
    // NOTES: 1. isa_all by default has no isa specific hints
    isa_all = ~0u & ~hints_mask,
};

/// Data type specification
typedef enum {
    /// Undefined data type, used for empty memory descriptors.
    dnnl_data_type_undef = 0,
    /// 16-bit/half-precision floating point.
    dnnl_f16 = 1,
    /// non-standard 16-bit (bfloat16 w/ 7 bit mantissa) floating point.
    dnnl_bf16 = 2,
    /// 32-bit/single-precision floating point.
    dnnl_f32 = 3,
    /// 32-bit signed integer.
    dnnl_s32 = 4,
    /// 8-bit signed integer.
    dnnl_s8 = 5,
    /// 8-bit unsigned integer.
    dnnl_u8 = 6,
    /// 64-bit/double-precision floating point.
    dnnl_f64 = 7,

    /// Parameter to allow internal only data_types without undefined behavior.
    /// This parameter is chosen to be valid for so long as sizeof(int) >= 2.
    dnnl_data_type_max = 0x7fff,
} dnnl_data_type_t;

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

// compile time constant
struct GemmDynMStaticParam {
    dnnl_data_type_t a_type, b_type, c_type;
    int N, K;   // for kernel N must be in [1, 64]
    int lda, ldb, ldc;
    PostOpStaticParams post_static_params;
};
// runtime changable
struct GemmDynMRuntimeParam {
    int m;
    void* a;
    void* b;
    void* c;
    PostOpRuntimeParams post_runtime_params;
};
template <cpu_isa_t isa>
struct gemm_kernel {
    gemm_kernel();
    bool init(const GemmDynMStaticParam& static_param);
    void operator()(const GemmDynMRuntimeParam& runtime_param);

    struct gemm_kernel_impl;
    std::shared_ptr<gemm_kernel_impl> _impl;
};

struct matmul {
    matmul();
    bool init(const GemmDynMStaticParam& static_param);
    void operator()(const GemmDynMRuntimeParam& runtime_param);

    struct matmul_impl;
    std::shared_ptr<matmul_impl> _impl;
};

template <cpu_isa_t isa>
struct mem_read_kernel {
    mem_read_kernel();
    bool init(int size);
    void operator()(int8_t* buf);

    struct mem_read_kernel_impl;
    std::shared_ptr<mem_read_kernel_impl> _impl;
};

struct mem_read {
    mem_read();
    bool init(int size);
    void operator()(int times);

    struct mem_read_impl;
    std::shared_ptr<mem_read_impl> _impl;
};

};

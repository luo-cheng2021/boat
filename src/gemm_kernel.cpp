#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iostream>

#include <coat/Function.h>
#include <coat/ControlFlow.h>
#include <coat/Vec.h>
#include <coat/Mask.h>
#include "boat.h"

#define ENABLE_DUMP 0

namespace Jit {
////////////////////////////////////////////////////
// generate jit kernel needed param when injecting kernels, private
struct PostOpInjectParam {
    BinaryDataLayout layout;
    coat::Ptr<coat::Value<float>> right_addrs_base;
    std::vector<int> right_addrs_offset;
};

struct PostOpInjectParams {
    PostOpInjectParam params[MAX_POSTOPS_NUM];
};

template<unsigned width>
using share_vec = std::shared_ptr<coat::Vec<float, width>>;

template <unsigned width>
void inject_postops(int vecs_num, std::vector<share_vec<width>> vecs, PostOpStaticParams* ops_param, PostOpInjectParams* inject_ops_param) {
    for (auto i = 0; i < ops_param->num; i++) {
        switch (ops_param->ops[i].alg_type) {
            case AlgType::Abs: {
                coat::Vec<float, width> tmp;
                std::for_each(vecs.begin(), vecs.begin() + vecs_num, [&] (share_vec<width> vec) {
                    tmp = -0.f;
                    tmp -= *vec;
                    vec->max_(tmp);
                });
                break;
            }
            case AlgType::Add: {
                if (inject_ops_param->params[i].layout == BinaryDataLayout::PerTensor) {
                    coat::Vec<float, width> tmp;
                    tmp.load(inject_ops_param->params[i].right_addrs_base[0], true);
                    std::for_each(vecs.begin(), vecs.begin() + vecs_num, [&] (share_vec<width> vec) {
                        *vec += tmp;
                    });
                } else if (inject_ops_param->params[i].layout == BinaryDataLayout::PerChannel) {
                    auto& base = inject_ops_param->params[i].right_addrs_base;
                    for (int j = 0; j < vecs_num; j++) {
                        auto idx = inject_ops_param->params[i].right_addrs_offset[j];
                        *vecs[j] += base[idx];
                    }
                }
                break;
            }
            default:
                // TODO: add other post ops
                _CC.int3();
                break;
        }
    }
}

// size should be runtime const
template<int vectorsize>
void jit_memset0(coat::Ptr<coat::Value<int8_t>> p, int size) {
    int offset = 0;
    int tail = size % (vectorsize * sizeof(float));
    if (size > vectorsize * (int)sizeof(float)) {
        coat::Vec<float, vectorsize> zero(true, "zero");
        const int size_4 = size / vectorsize / sizeof(float) / 4 * 4 * sizeof(float) * vectorsize;
        if (size_4 > 2 * 4 * (int)sizeof(float) * vectorsize) {
            coat::Value pos(int(0), "pos");
            coat::loop_while(pos < size_4, [&] {
                // p[pos + 1 * vectorsize * sizeof(float)]
                zero.store(p.index(pos, 0 * vectorsize * sizeof(float)));
                zero.store(p.index(pos, 1 * vectorsize * sizeof(float)));
                zero.store(p.index(pos, 2 * vectorsize * sizeof(float)));
                zero.store(p.index(pos, 3 * vectorsize * sizeof(float)));
                pos += 4 * vectorsize * sizeof(float);
            });
            offset += size_4;
        }
        for (; offset < (int)(size / (sizeof(float) * vectorsize) * sizeof(float) * vectorsize); offset += sizeof(float) * vectorsize) {
            zero.store(p[offset]);
        }
        if constexpr(vectorsize >= 16) {
            if (tail >= 8 * (int)sizeof(float)) {
                coat::Vec<float, 8> zero_y(zero.reg.half());
                zero_y.store(p[offset]);
                offset += 8 * sizeof(float);
                tail -= 8 * sizeof(float);
            }
            if (tail >= 4 * (int)sizeof(float)) {
                coat::Vec<float, 4> zero_y(zero.reg.half().half());
                zero_y.store(p[offset]);
                offset += 4 * sizeof(float);
                tail -= 4 * sizeof(float);
            }
        } else if constexpr(vectorsize >= 8) {
            if (tail >= 4 * (int)sizeof(float)) {
                coat::Vec<float, 4> zero_y(zero.reg.half());
                zero_y.store(p[offset]);
                offset += 4 * sizeof(float);
                tail -= 4 * sizeof(float);
            }
        }
    } else if (tail >= 4 * (int)sizeof(float)) {
        coat::Vec<float, vectorsize> zero(0, "zero");
        if constexpr(vectorsize >= 16) {
            if (tail >= 8 * (int)sizeof(float)) {
                coat::Vec<float, 8> zero_y(zero.reg.half());
                zero_y.store(p[offset]);
                offset += 8 * sizeof(float);
                tail -= 8 * sizeof(float);
            }
            if (tail >= 4 * (int)sizeof(float)) {
                coat::Vec<float, 4> zero_y(zero.reg.half().half());
                zero_y.store(p[offset]);
                offset += 4 * sizeof(float);
                tail -= 4 * sizeof(float);
            }
        } else if constexpr(vectorsize >= 8) {
            if (tail >= 4 * (int)sizeof(float)) {
                coat::Vec<float, 4> zero_y(zero.reg.half());
                zero_y.store(p[offset]);
                offset += 4 * sizeof(float);
                tail -= 4 * sizeof(float);
            }
        }
    }
    if (tail) {
        coat::Value<int64_t> zero;
        zero = 0;
        if (tail >= 8) {
            p.cast<int64_t>()[offset / 8] = zero;
            offset += 8;
            tail -= 8;
        }
        if (tail >= 4) {
            p.cast<int32_t>()[offset / 4] = coat::Value<int32_t>(zero.reg.r32());
            offset += 4;
            tail -= 4;
        }
        if (tail >= 2) {
            p.cast<int16_t>()[offset / 2] = coat::Value<int16_t>(zero.reg.r16());
            offset += 2;
            tail -= 2;
        }
        if (tail >= 1) {
            p.cast<int8_t>()[offset] = coat::Value<int8_t>(zero.reg.r8());
        }
    }
}

//
// M: ur_num * m_group * M' + M_tail, M is runtime changeable
// N: 16/32/48/64(may have tail)
// K: 16 * K' + K_tail
//
// loop order
// for m_block in 0..M
//   for sub_m in m_block
//     for k_block in 0..K
//       for k in k_block  --> fma
//         for m in ur
//           for n in n_block
//     for k_block_tail in ..K
// for m_block_tail in m_block
//   for sub_m_block_tail in m_block_tail
//     for k_block in 0..K
//       for k in k_block  --> fma
//         for m in ur
//           for n in n_block
//     for k_block_tail in ..K
using func_m_t = void (*)(int m, float* a, float* b, float* c, Jit::PostOpRuntimeParams* post_runtime_params);
template <unsigned width>
func_m_t gemm<width>::make_gemm_stride(int N, int K, int lda, int ldb, int ldc, Jit::PostOpStaticParams* post_static_params, const int ur_num, const int oc_num) {
    auto fn = coat::createFunction<func_m_t>("brgemm");
    if constexpr (width == 16)
        fn.funcNode->frame().setAvx512Enabled();
    else if  constexpr (width == 8)
        fn.funcNode->frame().setAvxEnabled();
#if ENABLE_DUMP
    fn.enableCodeDump();
#endif
    if ((N + width - 1) / width != static_cast<unsigned>(oc_num)) {
        std::cout << "oc_num must be equal ceil(N/width)" << std::endl;
        return nullptr;
    }
    {
        bool has_n_tail = (N % width) != 0;
        if (has_n_tail) {
            coat::Value<int> j_mask((1 << (N % width)) - 1);
            _CC.kmovq(asmjit::x86::k1, j_mask);
        }
        lda /= sizeof(float);
        ldb /= sizeof(float);
        ldc /= sizeof(float);
        auto [j_M, j_a, j_b, j_c, j_post_runtime_params] = fn.getArguments("m", "a", "b", "c", "ops");
        std::vector<Jit::share_vec<width>> j_weight(oc_num);
        std::vector<Jit::share_vec<width>> j_result;
        for (int i = 0; i < oc_num; i++) {
            j_weight[i] = std::make_shared<coat::Vec<float, width>>();
        }
        for (int i = 0; i < oc_num * ur_num; i++) {
            j_result.push_back(std::make_shared<coat::Vec<float, width>>());
        }
        coat::Vec<float, width> j_data;

        // postops
        Jit::PostOpInjectParams inject_postops_param;
        using share_p = std::shared_ptr<coat::Ptr<coat::Value<float>>>;
        std::vector<share_p> post_ops_runtime_addrs;
        // extract all second address from parameter 'ops' of jit func
        for (auto i = 0; i < post_static_params->num; i++) {
            if (post_static_params->ops[i].alg_type >= Jit::AlgType::Add) {
                auto params = j_post_runtime_params.get_value<Jit::PostOpRuntimeParams::member_params>("params");
                auto addr = params[i].get_value<Jit::PostOpRuntimeParam::member_right_addr>("addr");
                // TODO: ptr has no 'operator= addr'
                auto op = std::make_shared<share_p::element_type>(addr);
                post_ops_runtime_addrs.push_back(op);
            } else {
                auto op = std::make_shared<share_p::element_type>();
                // no need, just a placeholder
                post_ops_runtime_addrs.push_back(op);
            }
        }
        // compute all address for binary ops
        auto prepare_inject_param = [&] (int ur_num, int oc_num) {
            for (auto i = 0; i < post_static_params->num; i++) {
                if (post_static_params->ops[i].alg_type >= Jit::AlgType::Add && post_static_params->ops[i].binary_param.layout != Jit::BinaryDataLayout::PerTensor) {
                    auto& ptr = *post_ops_runtime_addrs[i];
                    inject_postops_param.params[i].right_addrs_offset.clear();
                    inject_postops_param.params[i].layout = post_static_params->ops[i].binary_param.layout;
                    inject_postops_param.params[i].right_addrs_base.reg = ptr.reg;
                    for (int j = 0; j < ur_num; j++)
                        for (int k = 0; k < oc_num; k++)
                            inject_postops_param.params[i].right_addrs_offset.push_back(k * width);
                }
            }
        };
        // several lines fall in one page
        int m_group = 1;
        if (lda < 128) m_group = 16;
        else if (lda < 256) m_group = 8;
        else if (lda < 512) m_group = 4;
        else if (lda < 1024) m_group = 2;
        coat::Value<int> j_m(int(0), "m");
        auto fma = [&has_n_tail, &j_weight, &j_data, &j_result](int ur_num, int k_num, int oc_num,
            coat::wrapper_type<float*>& j_a, coat::wrapper_type<float*>& j_b,
            int lda, int ldb) {
            for (int j = 0; j < k_num; j++) {
                for (int n = 0; n < oc_num - has_n_tail; n++) {
                    j_weight[n]->load(j_b[j * ldb + n * width]);
                }
                if (has_n_tail) {
                    j_weight[oc_num - 1]->kzload(j_b[j * ldb + (oc_num - 1) * width], asmjit::x86::k1);
                }
                for (int m = 0; m < ur_num; m++) {
                    j_data.load(j_a[m * lda + j], true);
                    for (int n = 0; n < oc_num; n++) {
                        j_result[m * oc_num + n]->fma231(*j_weight[n], j_data);
                    }
                }
            }
        };
        auto save_post = [&] (int ur_num, int oc_num, bool has_n_tail, int ldc, coat::wrapper_type<float *>& j_c) {
            prepare_inject_param(ur_num, oc_num);
            Jit::inject_postops<width>(ur_num * oc_num, j_result, post_static_params, &inject_postops_param);
            for (int m = 0; m < ur_num; m++) {
                for (int n = 0; n < oc_num - has_n_tail; n++) {
                    j_result[m * oc_num + n]->store(j_c[m * ldc + n * width]);
                }
                if (has_n_tail) {
                    j_result[m * oc_num + oc_num - 1]->kstore(j_c[m * ldc + (oc_num - 1) * width], asmjit::x86::k1);
                }
            }
        };
        auto j_M_block = j_M;
        j_M_block %= (ur_num * m_group);
        j_M_block = j_M - j_M_block;
        //for (m = 0; m < M; m += 8) {
        coat::for_loop(j_m < j_M_block,
        [&] {
            j_m += ur_num * m_group;
            j_a += ur_num * lda * m_group;
            j_c += ur_num * ldc * m_group;
        },
        [&] {
            coat::Value<int> j_sub_m(int(0), "sub_m");
            auto j_aa = j_a; // a ptr inside a group
            auto j_cc = j_c;
            //for (int sub_m = 0; sub_m < m_group; sub_m++) {
            coat::for_loop(j_sub_m < m_group,
            [&] {
                j_sub_m += 1;
                j_aa += lda;
                j_cc += ldc;
            },
            [&] {
                for (int i = 0; i < oc_num * ur_num; i++) {
                    (*j_result[i]) = 0;
                }
                coat::Value<int> j_k(int(0), "k");
                auto j_b_row = j_b;
                auto j_a_row = j_aa;
                //for (k = 0; k < K; k += width) {
                coat::for_loop(j_k < K / width * width,
                    [&] {
                        j_k += width;
                        j_b_row += width * ldb;
                        j_a_row += width;
                    },
                    [&] {
                        fma(ur_num, width, oc_num, j_a_row, j_b_row, lda * m_group, ldb);
                    });
                // K tail
                if (K % width != 0)
                    fma(ur_num, K % width, oc_num, j_a_row, j_b_row, lda * m_group, ldb);
                save_post(ur_num, oc_num, has_n_tail, ldc * m_group, j_cc);
            });
        });
 
        // M tail
        coat::if_then(j_M_block != j_M, [&] {
            auto j_M_block = j_M;
            j_M_block /= ur_num;
            j_M_block *= ur_num;
            // tail: handle multiple of ur_num tail
            //for (m = 0; m < M; m += 8) {
            coat::for_loop(j_m < j_M_block,
            [&] {
                j_m += ur_num;
                j_a += ur_num * lda;
                j_c += ur_num * ldc;
            },
            [&] {
                for (int i = 0; i < oc_num * ur_num; i++) {
                    (*j_result[i]) = 0;
                }
                coat::Value<int> j_k(int(0), "k");
                auto j_b_row = j_b;
                auto j_a_row = j_a;
                //for (k = 0; k < K; k += width) {
                coat::for_loop(j_k < K / width * width,
                    [&] {
                        j_k += width;
                        j_b_row += width * ldb;
                        j_a_row += width;
                    },
                    [&] {
                        fma(ur_num, width, oc_num, j_a_row, j_b_row, lda, ldb);
                    });
                // K tail
                if (K % width != 0)
                    fma(ur_num, K % width, oc_num, j_a_row, j_b_row, lda, ldb);
                save_post(ur_num, oc_num, has_n_tail, ldc, j_c);
            });
            // tail: handle not enough ur_num tail
            // TODO: try jump table fma(7/6/5/.../1)
            coat::if_then(j_M_block != j_M, [&] {
                j_M -= j_M_block;
                for (int i = 0; i < oc_num * ur_num; i++) {
                    (*j_result[i]) = 0;
                }
                coat::Value<int> j_k(int(0), "k");
                auto j_b_row = j_b;
                auto j_a_row = j_a;
                auto unroll_n = [&](int ur_num) {
                    //for (k = 0; k < K; k += width) {
                    coat::for_loop(j_k < K / width * width,
                        [&] {
                            j_k += width;
                            j_b_row += width * ldb;
                            j_a_row += width;
                        },
                        [&] {
                            fma(ur_num, width, oc_num, j_a_row, j_b_row, lda, ldb);
                        });
                    // K tail
                    if (K % width != 0)
                        fma(ur_num, K % width, oc_num, j_a_row, j_b_row, lda, ldb);
                    save_post(ur_num, oc_num, has_n_tail, ldc, j_c);
                };
                asmjit::Label L_End = _CC.newLabel();
                for (int i = 1; i < ur_num; i++) {
                    auto n = i;
                    coat::if_then(j_M == n, [&] {
                        unroll_n(n);
                        _CC.jmp(L_End);
                    });
                }
                _CC.bind(L_End);
            });
        });
        // specify return value
        coat::ret();
    }

    // finalize code generation and get function pointer to the generated function
    auto foo = fn.finalize();
    return foo;
}

void delete_func(void *p) {
    coat::getJitRuntimeEnv().release_func(p);
}

template struct gemm<16>;

};

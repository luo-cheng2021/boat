#pragma once

#include <tuple>
#include <array>
#include "Global.h"
#include "constexpr_helper.h"
#include "Function.h"
#include "Ref.h"

#include <type_traits>
#include <functional>
#include <cstdint>

// https://stackoverflow.com/questions/50080560/constexpr-offsetof-with-pointer-to-member-data
template<typename> struct struct_member_types;
template<typename T, typename F> struct struct_member_types<F T::*> {
  using struct_type = T;
  using member_type = F;
};
template<typename T> using struct_type_t = typename struct_member_types<T>::struct_type;
template<typename T> using member_type_t = typename struct_member_types<T>::member_type;

template<auto field> struct constexpr_field_offset {
  using T = struct_type_t<decltype(field)>;
  using F = member_type_t<decltype(field)>;
  static constexpr const char const_holder = 0;
  static constexpr const T& const_value = reinterpret_cast<const T&>(const_holder);
  static constexpr const F T::* const_field = field;
  static constexpr const char* base_addr = &reinterpret_cast<char const&>(const_value);
  static constexpr const char* field_addr = &reinterpret_cast<char const&>(const_value.*const_field);
  static constexpr const size_t offset = field_addr - base_addr;
};

template<auto field> inline constexpr std::size_t constexpr_field_offset_v = constexpr_field_offset<field>::offset;
template<auto field>
constexpr auto get_offset_type() {
    using type = std::remove_cv_t<std::decay_t<typename constexpr_field_offset<field>::F>>;
    auto offset = constexpr_field_offset_v<field>;
    return std::make_tuple(offset, type{});
}

namespace coat {

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


template<typename T>
struct has_custom_base : std::false_type {};

template<typename T>
struct StructBase;

struct StructBaseEmpty {};

// Note:
// 1, Constructor: 
//    Default constructor: allocate a Gp register
//    Construct from T: allocate a Gp register, copy T pointer
// 2, Assign:
//    Assign from T
template<typename T>
struct Struct
    : public std::conditional_t<has_custom_base<T>::value,
                                StructBase<Struct<T>>,
                                StructBaseEmpty
            >
{
    using struct_type = T;

    static_assert(std::is_standard_layout_v<T>, "wrapped class needs to have standard layout");

    asmjit::x86::Gp reg;
    size_t offset = 0;

    Struct(const char* name="") {
        reg = _CC.newIntPtr(name);
    }
#ifdef PROFILING_SOURCE
    Struct(T* val, const char* name="", const char* file=__builtin_FILE(), int line=__builtin_LINE()) : Struct(name) {
        *this = D<T*>{val, file, line};
    }
    Struct(const T* val, const char* name="", const char* file=__builtin_FILE(), int line=__builtin_LINE()) : Struct(name) {
        *this = D<T*>{const_cast<T*>(val), file, line};
    }
#else
    Struct(T* val, const char* name="") : Struct(name) {
        *this = val;
    }
    Struct(const T* val, const char* name="") : Struct(name) {
        *this = const_cast<T*>(val);
    }
#endif

    operator const asmjit::x86::Gp&() const { return reg; }
    operator       asmjit::x86::Gp&()       { return reg; }

    // load base pointer
    Struct& operator=(const D<T*>& other) {
        _CC.mov(reg, asmjit::imm(OP));
        DL;
        offset = 0;
        return *this;
    }

    // pre-increment
    Struct& operator++() { _CC.add(reg, sizeof(T)); return *this; }
    Struct operator[] (int amount) {
        Struct res;
        res.reg = reg; // pass ptr register
        res.offset = amount * sizeof(T) + offset; // change offset
        return res;
    }

    Struct operator+ (int amount) const {
        Struct res;
        res.reg = reg; // pass ptr register
        res.offset = amount * sizeof(T) + offset; // change offset
        return res;
    }

    // template<int I>
    // Ref<reg_type<std::tuple_element_t<I, typename T::types>>> get_reference() {
    //     static_assert(sizeof(std::tuple_element_t<I, typename T::types>) <= 8, "data length must less than 8");
    //     switch(sizeof(std::tuple_element_t<I, typename T::types>)) {
    //         case 1: return { asmjit::x86:: byte_ptr(reg, (int32_t)(offset_of_v<I, typename T::types> + offset)) };
    //         case 2: return { asmjit::x86:: word_ptr(reg, (int32_t)(offset_of_v<I, typename T::types> + offset)) };
    //         case 4: return { asmjit::x86::dword_ptr(reg, (int32_t)(offset_of_v<I, typename T::types> + offset)) };
    //         case 8: return { asmjit::x86::qword_ptr(reg, (int32_t)(offset_of_v<I, typename T::types> + offset)) };
    //     }
    //     assert(false);
    //     return { asmjit::x86:: byte_ptr(reg, (int32_t)(offset_of_v<I, typename T::types> + offset)) };
    // }
    template<typename MemberType, size_t MemberOffset>
    wrapper_type<MemberType> get_value1() const {
        wrapper_type<MemberType> ret;
        if constexpr(std::is_floating_point_v<MemberType>) {
            _CC.movss(ret.reg, asmjit::x86::dword_ptr(reg, (int32_t)(MemberOffset + offset)));
        } else if constexpr(std::is_array_v<MemberType>) {
            // array decay to pointer, just add offset to struct pointer
            //TODO: could just use struct pointer with fixed offset, no need for new register, similar to nested struct
            //_CC.lea(ret.reg, asmjit::x86::ptr(reg, offset_of_v<I, typename T::types> + offset));
            ret.reg = reg; // pass ptr register
            ret.offset = offset + MemberOffset; // change offset

#ifdef PROFILING_SOURCE
            ((PerfCompiler&)_CC).attachDebugLine(file, line);
#endif
        } else if constexpr(std::is_arithmetic_v<std::remove_pointer_t<MemberType>>) {
#if 0
            //FIXME: VRegMem not defined for pointer types currently
            ret = get_reference<I>();
#else
            switch(sizeof(MemberType)) {
                case 1: _CC.mov(ret.reg, asmjit::x86:: byte_ptr(reg, (int32_t)(MemberOffset + offset))); break;
                case 2: _CC.mov(ret.reg, asmjit::x86:: word_ptr(reg, (int32_t)(MemberOffset + offset))); break;
                case 4: _CC.mov(ret.reg, asmjit::x86::dword_ptr(reg, (int32_t)(MemberOffset + offset))); break;
                case 8: _CC.mov(ret.reg, asmjit::x86::qword_ptr(reg, (int32_t)(MemberOffset + offset))); break;
            }
#endif
#ifdef PROFILING_SOURCE
            ((PerfCompiler&)_CC).attachDebugLine(file, line);
#endif
        } else if constexpr(std::is_pointer_v<MemberType>) {
            // pointer to struct, load pointer
            _CC.mov(ret.reg, asmjit::x86::qword_ptr(reg, MemberOffset + offset));
#ifdef PROFILING_SOURCE
            ((PerfCompiler&)_CC).attachDebugLine(file, line);
#endif
        } else {
            // nested struct
            ret.reg = reg; // pass ptr register
            ret.offset = offset + MemberOffset; // change offset
        }

        return ret;
    }

    template<typename MemberType>
    wrapper_type<MemberType> get_value(int MemberOffset) const {
        wrapper_type<MemberType> ret;
        if constexpr(std::is_floating_point_v<MemberType>) {
            _CC.movss(ret.reg, asmjit::x86::dword_ptr(reg, (int32_t)(MemberOffset + offset)));
        } else if constexpr(std::is_array_v<MemberType>) {
            // array decay to pointer, just add offset to struct pointer
            //TODO: could just use struct pointer with fixed offset, no need for new register, similar to nested struct
            //_CC.lea(ret.reg, asmjit::x86::ptr(reg, offset_of_v<I, typename T::types> + offset));
            ret.reg = reg; // pass ptr register
            ret.offset = offset + MemberOffset; // change offset

#ifdef PROFILING_SOURCE
            ((PerfCompiler&)_CC).attachDebugLine(file, line);
#endif
        } else if constexpr(std::is_arithmetic_v<std::remove_pointer_t<MemberType>>) {
#if 0
            //FIXME: VRegMem not defined for pointer types currently
            ret = get_reference<I>();
#else
            switch(sizeof(MemberType)) {
                case 1: _CC.mov(ret.reg, asmjit::x86:: byte_ptr(reg, (int32_t)(MemberOffset + offset))); break;
                case 2: _CC.mov(ret.reg, asmjit::x86:: word_ptr(reg, (int32_t)(MemberOffset + offset))); break;
                case 4: _CC.mov(ret.reg, asmjit::x86::dword_ptr(reg, (int32_t)(MemberOffset + offset))); break;
                case 8: _CC.mov(ret.reg, asmjit::x86::qword_ptr(reg, (int32_t)(MemberOffset + offset))); break;
            }
#endif
#ifdef PROFILING_SOURCE
            ((PerfCompiler&)_CC).attachDebugLine(file, line);
#endif
        } else if constexpr(std::is_pointer_v<MemberType>) {
            // pointer to struct, load pointer
            _CC.mov(ret.reg, asmjit::x86::qword_ptr(reg, MemberOffset + offset));
#ifdef PROFILING_SOURCE
            ((PerfCompiler&)_CC).attachDebugLine(file, line);
#endif
        } else {
            // nested struct
            ret.reg = reg; // pass ptr register
            ret.offset = offset + MemberOffset; // change offset
        }

        return ret;
    }

    template<auto field>
    auto get() const {
        using type = member_type_t<decltype(field)>;
        static constexpr const type struct_type::* const_field = field;

        int member_offset = static_cast<int>(reinterpret_cast<size_t>(&(((struct_type*)nullptr)->*const_field)));

        return get_value<type>(member_offset);
    }

#define GET(obj, var)   \
    [copy = obj] ()  {  \
        using struct_type = typename std::remove_reference_t<decltype(copy)>::struct_type;    \
        using type = typename std::remove_cv_t<std::decay_t<typename constexpr_field_offset<&struct_type::var>::F>>; \
        constexpr auto offset = constexpr_field_offset_v<&struct_type::var>; \
        return (copy). template get_value1<type, offset>();   \
    } ();

//     template<int I>
//     wrapper_type<std::tuple_element_t<I, typename T::types>> get_value (
//         const char* name=""
// #ifdef PROFILING_SOURCE
//         ,const char* file=__builtin_FILE(), int line=__builtin_LINE()
// #endif
//     ) const {
//         using type = std::tuple_element_t<I, typename T::types>;
//         wrapper_type<type> ret(name);
//         if constexpr(std::is_floating_point_v<type>) {
//             _CC.movss(ret.reg, asmjit::x86::dword_ptr(reg, (int32_t)(offset_of_v<I, typename T::types> + offset)));
//         } else if constexpr(std::is_array_v<type>) {
//             // array decay to pointer, just add offset to struct pointer
//             //TODO: could just use struct pointer with fixed offset, no need for new register, similar to nested struct
//             //_CC.lea(ret.reg, asmjit::x86::ptr(reg, offset_of_v<I, typename T::types> + offset));
//             ret.reg = reg; // pass ptr register
//             ret.offset = offset + offset_of_v<I, typename T::types>; // change offset

// #ifdef PROFILING_SOURCE
//             ((PerfCompiler&)_CC).attachDebugLine(file, line);
// #endif
//         } else if constexpr(std::is_arithmetic_v<std::remove_pointer_t<type>>) {
// #if 0
//             //FIXME: VRegMem not defined for pointer types currently
//             ret = get_reference<I>();
// #else
//             switch(sizeof(type)) {
//                 case 1: _CC.mov(ret.reg, asmjit::x86:: byte_ptr(reg, (int32_t)(offset_of_v<I, typename T::types> + offset))); break;
//                 case 2: _CC.mov(ret.reg, asmjit::x86:: word_ptr(reg, (int32_t)(offset_of_v<I, typename T::types> + offset))); break;
//                 case 4: _CC.mov(ret.reg, asmjit::x86::dword_ptr(reg, (int32_t)(offset_of_v<I, typename T::types> + offset))); break;
//                 case 8: _CC.mov(ret.reg, asmjit::x86::qword_ptr(reg, (int32_t)(offset_of_v<I, typename T::types> + offset))); break;
//             }
// #endif
// #ifdef PROFILING_SOURCE
//             ((PerfCompiler&)_CC).attachDebugLine(file, line);
// #endif
//         } else if constexpr(std::is_pointer_v<type>) {
//             // pointer to struct, load pointer
//             _CC.mov(ret.reg, asmjit::x86::qword_ptr(reg, offset_of_v<I, typename T::types> + offset));
// #ifdef PROFILING_SOURCE
//             ((PerfCompiler&)_CC).attachDebugLine(file, line);
// #endif
//         } else {
//             // nested struct
//             ret.reg = reg; // pass ptr register
//             ret.offset = offset + offset_of_v<I, typename T::types>; // change offset
//         }
//         return ret;
//     }
};

} // namespace

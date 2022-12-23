#include <stdint.h>
#include <assert.h>
#include <algorithm>
// Required by `__cpuidex()` and `_xgetbv()`.
#ifdef _WIN32
  #include <intrin.h>
#else
	#ifndef __GNUC_PREREQ
    	#define __GNUC_PREREQ(major, minor) ((((__GNUC__) << 16) + (__GNUC_MINOR__)) >= (((major) << 16) + (minor)))
	#endif
	#if __GNUC_PREREQ(4, 3) && !defined(__APPLE__)
		#include <cpuid.h>
	#else
		#if defined(__APPLE__) && defined(XBYAK32) // avoid err : can't find a register in class `BREG' while reloading `asm'
			#define __cpuid(eaxIn, a, b, c, d) __asm__ __volatile__("pushl %%ebx\ncpuid\nmovl %%ebp, %%esi\npopl %%ebx" : "=a"(a), "=S"(b), "=c"(c), "=d"(d) : "0"(eaxIn))
			#define __cpuid_count(eaxIn, ecxIn, a, b, c, d) __asm__ __volatile__("pushl %%ebx\ncpuid\nmovl %%ebp, %%esi\npopl %%ebx" : "=a"(a), "=S"(b), "=c"(c), "=d"(d) : "0"(eaxIn), "2"(ecxIn))
		#else
			#define __cpuid(eaxIn, a, b, c, d) __asm__ __volatile__("cpuid\n" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "0"(eaxIn))
			#define __cpuid_count(eaxIn, ecxIn, a, b, c, d) __asm__ __volatile__("cpuid\n" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "0"(eaxIn), "2"(ecxIn))
		#endif
	#endif
#endif
#include "tool.h"

struct cpuid_t { uint32_t eax, ebx, ecx, edx; };
struct xgetbv_t { uint32_t eax, edx; };

static inline void getCpuidEx(unsigned int eaxIn, unsigned int ecxIn, unsigned int data[4])
{
#ifdef _MSC_VER
    __cpuidex(reinterpret_cast<int*>(data), eaxIn, ecxIn);
#else
    __cpuid_count(eaxIn, ecxIn, data[0], data[1], data[2], data[3]);
#endif
}

static unsigned int extractBit(unsigned int val, unsigned int base, unsigned int end)
{
    return (val >> base) & ((1u << (end - base)) - 1);
}
static const unsigned int maxNumberCacheLevels = 10;
static unsigned int dataCacheSize_[maxNumberCacheLevels];
static unsigned int coresSharignDataCache_[maxNumberCacheLevels];
static unsigned int dataCacheLevels_ = 0;
unsigned int getDataCacheSize(unsigned int level) {
    if (level == 0 || (level > dataCacheLevels_ && dataCacheLevels_))
        return 0;

    if (dataCacheSize_[0] == 0) {
        const unsigned int NO_CACHE = 0;
        const unsigned int DATA_CACHE = 1;
        // const unsigned int INSTRUCTION_CACHE = 2;
        const unsigned int UNIFIED_CACHE = 3;
        unsigned int smt_width = 0;
        unsigned int logical_cores = 0;
        unsigned int data[4] = {};
        /*
            Assumptions:
            the first level of data cache is not shared (which is the
            case for every existing architecture) and use this to
            determine the SMT width for arch not supporting leaf 11.
            when leaf 4 reports a number of core less than numCores_
            on socket reported by leaf 11, then it is a correct number
            of cores not an upperbound.
        */
        for (int i = 0; dataCacheLevels_ < maxNumberCacheLevels; i++) {
            getCpuidEx(0x4, i, data);
            unsigned int cacheType = extractBit(data[0], 0, 4);
            if (cacheType == NO_CACHE) break;
            if (cacheType == DATA_CACHE || cacheType == UNIFIED_CACHE) {
                unsigned int actual_logical_cores = extractBit(data[0], 14, 25) + 1;
                if (logical_cores != 0) { // true only if leaf 0xB is supported and valid
                    actual_logical_cores = (std::min)(actual_logical_cores, logical_cores);
                }
                assert(actual_logical_cores != 0);
                dataCacheSize_[dataCacheLevels_] =
                    (extractBit(data[1], 22, 31) + 1)
                    * (extractBit(data[1], 12, 21) + 1)
                    * (extractBit(data[1], 0, 11) + 1)
                    * (data[2] + 1);
                if (cacheType == DATA_CACHE && smt_width == 0) smt_width = actual_logical_cores;
                assert(smt_width != 0);
                coresSharignDataCache_[dataCacheLevels_] = (std::max)(actual_logical_cores / smt_width, 1u);
                dataCacheLevels_++;
            }
        }
    }
    return dataCacheSize_[level - 1];
}

#include "vectorpulse/avx2_search_backend.h"
#include "avx2_kernel.h"
#include <stdexcept>

#if defined(VECTORPULSE_HAS_AVX2_KERNEL) && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace vectorpulse {
bool AVX2SearchBackend::is_supported() noexcept {
#if defined(VECTORPULSE_HAS_AVX2_KERNEL) && defined(_MSC_VER)
    int regs[4]{};
    __cpuid(regs, 0);
    if (regs[0] < 7) return false;
    __cpuidex(regs, 1, 0);
    constexpr int required = (1 << 12) | (1 << 27) | (1 << 28);
    if ((regs[2] & required) != required) return false;
    if ((_xgetbv(0) & 6) != 6) return false;
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;
#elif defined(VECTORPULSE_HAS_AVX2_KERNEL) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

#ifndef VECTORPULSE_HAS_AVX2_KERNEL
float detail::avx2_cosine_similarity(std::span<const float>, std::span<const float>) {
    throw std::runtime_error("AVX2 kernel was not built");
}
#endif
}

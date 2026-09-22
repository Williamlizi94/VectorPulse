#include "avx2_kernel.h"
#include <cmath>
#include <immintrin.h>

namespace vectorpulse::detail {
namespace {
double sum(__m256d value) {
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, value);
    return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
}
}

float avx2_cosine_similarity(std::span<const float> lhs, std::span<const float> rhs) {
    __m256d dot0 = _mm256_setzero_pd(), dot1 = _mm256_setzero_pd();
    __m256d norm_a0 = _mm256_setzero_pd(), norm_a1 = _mm256_setzero_pd();
    __m256d norm_b0 = _mm256_setzero_pd(), norm_b1 = _mm256_setzero_pd();
    std::size_t i = 0;
    // Widen floats before multiplication, matching the scalar baseline's range
    // and precision. Two independent groups process eight dimensions per step.
    for (; lhs.size() - i >= 8; i += 8) {
        const __m256d a0 = _mm256_cvtps_pd(_mm_loadu_ps(lhs.data() + i));
        const __m256d a1 = _mm256_cvtps_pd(_mm_loadu_ps(lhs.data() + i + 4));
        const __m256d b0 = _mm256_cvtps_pd(_mm_loadu_ps(rhs.data() + i));
        const __m256d b1 = _mm256_cvtps_pd(_mm_loadu_ps(rhs.data() + i + 4));
        dot0 = _mm256_fmadd_pd(a0, b0, dot0);
        dot1 = _mm256_fmadd_pd(a1, b1, dot1);
        norm_a0 = _mm256_fmadd_pd(a0, a0, norm_a0);
        norm_a1 = _mm256_fmadd_pd(a1, a1, norm_a1);
        norm_b0 = _mm256_fmadd_pd(b0, b0, norm_b0);
        norm_b1 = _mm256_fmadd_pd(b1, b1, norm_b1);
    }
    double dot = sum(dot0) + sum(dot1);
    double norm_a = sum(norm_a0) + sum(norm_a1);
    double norm_b = sum(norm_b0) + sum(norm_b1);
    for (; i < lhs.size(); ++i) {
        const double a = lhs[i], b = rhs[i];
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }
    if (norm_a == 0.0 || norm_b == 0.0) return 0.0F;
    return static_cast<float>(dot / std::sqrt(norm_a * norm_b));
}
}

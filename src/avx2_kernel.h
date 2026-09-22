#pragma once
#include <span>

namespace vectorpulse::detail {
// Internal: caller validates dimensions and CPU support before entry.
float avx2_cosine_similarity(std::span<const float> lhs, std::span<const float> rhs);
}

#pragma once

#include <span>

namespace vectorpulse {

// Zero-magnitude vectors have a defined similarity of 0.0f.
[[nodiscard]] float cosine_similarity(
    std::span<const float> lhs,
    std::span<const float> rhs);

}  // namespace vectorpulse

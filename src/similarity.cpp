#include "vectorpulse/similarity.h"

#include <cmath>
#include <stdexcept>

namespace vectorpulse {

float cosine_similarity(
    const std::span<const float> lhs,
    const std::span<const float> rhs) {
    if (lhs.empty() || rhs.empty()) {
        throw std::invalid_argument("cosine similarity requires non-empty vectors");
    }
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("cosine similarity requires equal dimensions");
    }

    double dot_product = 0.0;
    double lhs_squared_norm = 0.0;
    double rhs_squared_norm = 0.0;

    for (std::size_t index = 0; index < lhs.size(); ++index) {
        const double lhs_value = static_cast<double>(lhs[index]);
        const double rhs_value = static_cast<double>(rhs[index]);
        dot_product += lhs_value * rhs_value;
        lhs_squared_norm += lhs_value * lhs_value;
        rhs_squared_norm += rhs_value * rhs_value;
    }

    if (lhs_squared_norm == 0.0 || rhs_squared_norm == 0.0) {
        return 0.0F;
    }

    return static_cast<float>(
        dot_product / std::sqrt(lhs_squared_norm * rhs_squared_norm));
}

}  // namespace vectorpulse

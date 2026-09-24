#pragma once

#include "vectorpulse/search_result.h"

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vectorpulse {

class SearchBackend {
public:
    virtual ~SearchBackend() = default;

    using VectorVisitor = std::function<void(std::string_view, std::span<const float>)>;
    // Visits host entries in insertion order. Visitors must not mutate the backend
    // or retain views beyond the call. Exclude concurrent insertion.
    // Optional for third-party backends; all built-in backends implement it.
    virtual void for_each_vector(const VectorVisitor&) const {
        throw std::runtime_error("backend does not support vector enumeration");
    }

    [[nodiscard]] virtual std::size_t dimension() const noexcept = 0;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;

    virtual void add(std::string id, std::vector<float> values) = 0;
    [[nodiscard]] virtual const std::vector<float>& get(std::string_view id) const = 0;
    [[nodiscard]] virtual std::vector<SearchResult> search(
        std::span<const float> query,
        std::size_t k) const = 0;
};

}  // namespace vectorpulse

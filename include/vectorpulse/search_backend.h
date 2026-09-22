#pragma once

#include "vectorpulse/search_result.h"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vectorpulse {

class SearchBackend {
public:
    virtual ~SearchBackend() = default;

    [[nodiscard]] virtual std::size_t dimension() const noexcept = 0;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;

    virtual void add(std::string id, std::vector<float> values) = 0;
    [[nodiscard]] virtual const std::vector<float>& get(std::string_view id) const = 0;
    [[nodiscard]] virtual std::vector<SearchResult> search(
        std::span<const float> query,
        std::size_t k) const = 0;
};

}  // namespace vectorpulse

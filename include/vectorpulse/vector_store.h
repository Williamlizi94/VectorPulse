#pragma once

#include "vectorpulse/search_backend.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vectorpulse {

class VectorStore {
public:
    explicit VectorStore(
        std::size_t dimension,
        std::unique_ptr<SearchBackend> backend = nullptr);

    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

    void add(std::string id, std::vector<float> values);
    [[nodiscard]] const std::vector<float>& get(std::string_view id) const;
    [[nodiscard]] std::vector<SearchResult> search(
        std::span<const float> query,
        std::size_t k) const;

private:
    std::size_t dimension_;
    std::unique_ptr<SearchBackend> backend_;
};

}  // namespace vectorpulse

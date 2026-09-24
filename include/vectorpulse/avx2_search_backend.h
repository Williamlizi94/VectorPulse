#pragma once

#include "vectorpulse/search_backend.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace vectorpulse {

class AVX2SearchBackend final : public SearchBackend {
public:
    // Requires AVX2, FMA, and OS YMM support. No fallback.
    [[nodiscard]] static bool is_supported() noexcept;
    explicit AVX2SearchBackend(std::size_t dimension);

    [[nodiscard]] std::size_t dimension() const noexcept override;
    [[nodiscard]] std::size_t size() const noexcept override;

    void add(std::string id, std::vector<float> values) override;
    [[nodiscard]] const std::vector<float>& get(std::string_view id) const override;
    [[nodiscard]] std::vector<SearchResult> search(
        std::span<const float> query,
        std::size_t k) const override;

    void for_each_vector(const VectorVisitor& visitor) const override {
        for (const auto& entry : entries_) {
            visitor(entry.id, entry.values);
        }
    }
private:
    struct Entry {
        std::string id;
        std::vector<float> values;
    };

    std::size_t dimension_;
    std::vector<Entry> entries_;
    std::unordered_map<std::string, std::size_t> index_by_id_;
};

}  // namespace vectorpulse


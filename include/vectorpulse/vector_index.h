#pragma once

#include "vectorpulse/vector_store.h"

#include <filesystem>
#include <utility>

namespace vectorpulse {

// Unified exact cosine-search API. Defaults to the scalar backend.
// Read-only searches may run concurrently; callers must exclude concurrent add().
class VectorIndex {
public:
    explicit VectorIndex(std::size_t dimension,
                         std::unique_ptr<SearchBackend> backend = nullptr)
        : store_(dimension, std::move(backend)) {}

    void add(std::string id, std::vector<float> values) {
        store_.add(std::move(id), std::move(values));
    }

    // Descending similarity, then ascending ID for exactly equal scores.
    // K=0 returns no results; K greater than size() returns all entries.
    [[nodiscard]] std::vector<SearchResult> search(
        std::span<const float> query, std::size_t k) const {
        return store_.search(query, k);
    }

    [[nodiscard]] std::size_t size() const noexcept { return store_.size(); }
    [[nodiscard]] std::size_t dimension() const noexcept { return store_.dimension(); }

    // Stores host IDs and exact FP32 bits, never backend/GPU state.
    // Like search(), save() requires callers to exclude concurrent insertion.
    // File/format errors throw runtime_error; failed saves may leave a partial file.
    void save(const std::filesystem::path& path) const;

    // Defaults to scalar. A supplied backend must be empty and match the file's
    // dimension. Same backend/settings reproduce identical search results;
    // switching arithmetic backends retains their existing rounding behavior.
    [[nodiscard]] static VectorIndex load(
        const std::filesystem::path& path,
        std::unique_ptr<SearchBackend> backend = nullptr);

private:
    VectorStore store_;
};

}  // namespace vectorpulse

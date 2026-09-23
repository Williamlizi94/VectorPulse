#pragma once

#include "vectorpulse/vector_store.h"

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

private:
    VectorStore store_;
};

}  // namespace vectorpulse

#pragma once

#include "vectorpulse/search_result.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace vectorpulse {

struct HnswConfig {
    std::size_t M = 16;
    std::size_t efConstruction = 200;
    std::size_t efSearch = 50;
    std::uint64_t seed = 42;
};

// Append-only, CPU approximate cosine search, independent of the exact backends.
// Read-only searches may run concurrently. Exclude add()/set_ef_search() from
// all concurrent operations. Input vectors/queries must have finite components.
class HnswIndex {
public:
    explicit HnswIndex(std::size_t dimension, HnswConfig config = {});

    void add(std::string id, std::vector<float> values);
    [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query,
                                                  std::size_t k) const;
    // Per-query override does not mutate the index. Effective ef is at least K.
    [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query,
                                                  std::size_t k,
                                                  std::size_t efSearch) const;
    [[nodiscard]] std::size_t size() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
    [[nodiscard]] HnswConfig config() const noexcept { return config_; }
    void set_ef_search(std::size_t efSearch);

private:
    friend struct HnswDiagnosticAccess; // Read-only benchmark/test inspection.
    struct Node {
        std::string id;
        std::vector<float> values;
        std::vector<std::vector<std::size_t>> links;
    };
    struct Candidate { std::size_t node; float score; };
    std::size_t dimension_;
    HnswConfig config_;
    std::mt19937_64 random_;
    std::vector<Node> nodes_;
    std::unordered_map<std::string, std::size_t> index_by_id_;
    std::size_t entry_ = 0;
    std::size_t max_level_ = 0;

    void validate(std::span<const float> values) const;
    [[nodiscard]] std::size_t sample_level(std::mt19937_64& random) const;
    [[nodiscard]] bool better(const Candidate& lhs, const Candidate& rhs) const;
    [[nodiscard]] Candidate greedy(std::span<const float> query, Candidate entry,
                                   std::size_t layer) const;
    [[nodiscard]] std::vector<Candidate> search_layer(std::span<const float> query,
        std::span<const Candidate> entries, std::size_t ef, std::size_t layer) const;
    [[nodiscard]] std::vector<std::size_t> select_neighbors(
        std::vector<Candidate> candidates, std::size_t limit,
        std::vector<std::size_t> required = {}) const;
};

}  // namespace vectorpulse

#include "vectorpulse/hnsw_index.h"

#include "vectorpulse/similarity.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace vectorpulse {

HnswIndex::HnswIndex(std::size_t dimension, HnswConfig config)
    : dimension_(dimension), config_(config), random_(config.seed) {
    if (dimension == 0) throw std::invalid_argument("vector dimension must be greater than zero");
    if (config.M < 2 || config.M > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::invalid_argument("HNSW M must be at least 2 and allow a 2*M base-layer degree");
    }
    if (config.efConstruction < config.M) {
        throw std::invalid_argument("HNSW efConstruction must be at least M");
    }
    if (config.efSearch == 0) throw std::invalid_argument("HNSW efSearch must be positive");
}

void HnswIndex::validate(std::span<const float> values) const {
    if (values.size() != dimension_) throw std::invalid_argument("vector dimension does not match the index");
    for (float value : values) {
        if (!std::isfinite(value)) throw std::invalid_argument("HNSW requires finite vector components");
    }
}

void HnswIndex::set_ef_search(std::size_t efSearch) {
    if (efSearch == 0) throw std::invalid_argument("HNSW efSearch must be positive");
    config_.efSearch = efSearch;
}

std::size_t HnswIndex::sample_level(std::mt19937_64& random) const {
    // Geometric promotion probability 1/M, using integer rejection sampling so
    // level generation is independent of standard-library real distributions.
    const auto bound = static_cast<std::uint64_t>(config_.M);
    const auto threshold = (std::uint64_t{0} - bound) % bound;
    std::size_t level = 0;
    while (level < 32) {
        std::uint64_t draw;
        do { draw = random(); } while (draw < threshold);
        if (draw % bound != 0) break;
        ++level;
    }
    return level;
}

bool HnswIndex::better(const Candidate& lhs, const Candidate& rhs) const {
    return lhs.score != rhs.score ? lhs.score > rhs.score
                                 : nodes_[lhs.node].id < nodes_[rhs.node].id;
}

HnswIndex::Candidate HnswIndex::greedy(std::span<const float> query, Candidate entry,
                                      std::size_t layer) const {
    for (;;) {
        auto next = entry;
        for (const auto neighbor : nodes_[entry.node].links[layer]) {
            const Candidate candidate{neighbor, cosine_similarity(query, nodes_[neighbor].values)};
            if (better(candidate, next)) next = candidate;
        }
        if (next.node == entry.node) return entry;
        entry = next;
    }
}

std::vector<HnswIndex::Candidate> HnswIndex::search_layer(std::span<const float> query,
    std::span<const Candidate> entries, std::size_t ef, std::size_t layer) const {
    const auto best_first = [this](const Candidate& a, const Candidate& b) { return better(b, a); };
    const auto worst_first = [this](const Candidate& a, const Candidate& b) { return better(a, b); };
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(best_first)> pending(best_first);
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(worst_first)> nearest(worst_first);
    std::unordered_set<std::size_t> visited;
    visited.reserve(std::min(ef, nodes_.size()));
    for (const auto candidate : entries) {
        if (visited.insert(candidate.node).second) {
            pending.push(candidate);
            nearest.push(candidate);
            if (nearest.size() > ef) nearest.pop();
        }
    }
    while (!pending.empty()) {
        const auto current = pending.top();
        pending.pop();
        if (nearest.size() == ef && better(nearest.top(), current)) break;
        for (const auto neighbor : nodes_[current.node].links[layer]) {
            if (!visited.insert(neighbor).second) continue;
            const Candidate candidate{neighbor, cosine_similarity(query, nodes_[neighbor].values)};
            if (nearest.size() < ef || better(candidate, nearest.top())) {
                pending.push(candidate);
                nearest.push(candidate);
                if (nearest.size() > ef) nearest.pop();
            }
        }
    }
    std::vector<Candidate> result;
    result.reserve(nearest.size());
    while (!nearest.empty()) { result.push_back(nearest.top()); nearest.pop(); }
    std::sort(result.begin(), result.end(), worst_first);
    return result;
}

std::vector<std::size_t> HnswIndex::select_neighbors(std::vector<Candidate> candidates,
    std::size_t limit, std::vector<std::size_t> required) const {
    std::sort(candidates.begin(), candidates.end(),
              [this](const Candidate& a, const Candidate& b) { return better(a, b); });
    auto selected = std::move(required);
    selected.reserve(std::min(limit, candidates.size() + selected.size()));
    std::vector<std::size_t> pruned;
    for (const auto candidate : candidates) {
        if (selected.size() == limit) break;
        if (std::find(selected.begin(), selected.end(), candidate.node) != selected.end()) continue;
        bool diverse = true;
        for (const auto neighbor : selected) {
            if (cosine_similarity(nodes_[candidate.node].values, nodes_[neighbor].values) >= candidate.score) {
                diverse = false;
                break;
            }
        }
        if (diverse) selected.push_back(candidate.node);
        else pruned.push_back(candidate.node);
    }
    // Keep pruned candidates if diversity alone does not fill the list.
    for (const auto node : pruned) {
        if (selected.size() == limit) break;
        selected.push_back(node);
    }
    return selected;
}

void HnswIndex::add(std::string id, std::vector<float> values) {
    validate(values);
    if (index_by_id_.contains(id)) throw std::invalid_argument("vector ID already exists: " + id);
    auto next_random = random_;
    const auto level = sample_level(next_random);
    const auto inserted = nodes_.size();
    Node node{std::move(id), std::move(values), std::vector<std::vector<std::size_t>>(level + 1)};
    nodes_.push_back(std::move(node));
    struct Update { std::size_t node, layer; std::vector<std::size_t> links; };
    std::vector<Update> updates;
    try {
        if (inserted != 0) {
            const auto& query = nodes_[inserted].values;
            Candidate entry{entry_, cosine_similarity(query, nodes_[entry_].values)};
            for (auto layer = max_level_; layer > level; --layer) entry = greedy(query, entry, layer);
            std::vector<Candidate> entries{entry};
            auto layer = std::min(level, max_level_);
            for (;;) {
                auto candidates = search_layer(query, entries, std::min(config_.efConstruction, inserted), layer);
                auto neighbors = select_neighbors(candidates, config_.M);
                // Preserve an insertion-order chain inside the base-layer degree
                // cap, preventing unreachable components on duplicate/zero data.
                if (layer == 0 && std::find(neighbors.begin(), neighbors.end(), inserted - 1) == neighbors.end()) {
                    neighbors.push_back(inserted - 1);
                }
                nodes_[inserted].links[layer] = neighbors;
                for (const auto neighbor : neighbors) {
                    auto links = nodes_[neighbor].links[layer];
                    links.push_back(inserted);
                    const auto limit = layer == 0 ? 2 * config_.M : config_.M;
                    if (links.size() > limit) {
                        std::vector<Candidate> choices;
                        choices.reserve(links.size());
                        for (const auto adjacent : links) {
                            choices.push_back({adjacent,
                                cosine_similarity(nodes_[neighbor].values, nodes_[adjacent].values)});
                        }
                        std::vector<std::size_t> required;
                        if (layer == 0) {
                            if (neighbor > 0) required.push_back(neighbor - 1);
                            if (neighbor + 1 < nodes_.size()) required.push_back(neighbor + 1);
                        }
                        links = select_neighbors(std::move(choices), limit, std::move(required));
                    }
                    updates.push_back({neighbor, layer, std::move(links)});
                }
                entries = std::move(candidates);
                if (layer == 0) break;
                --layer;
            }
        }
        index_by_id_.emplace(nodes_[inserted].id, inserted);
    } catch (...) {
        nodes_.pop_back();
        throw;
    }
    // No throwing operations follow: failed insertion leaves graph and RNG intact.
    for (auto& update : updates) nodes_[update.node].links[update.layer].swap(update.links);
    if (inserted == 0 || level > max_level_) { entry_ = inserted; max_level_ = level; }
    random_ = next_random;
}

std::vector<SearchResult> HnswIndex::search(std::span<const float> query, std::size_t k) const {
    return search(query, k, config_.efSearch);
}

std::vector<SearchResult> HnswIndex::search(std::span<const float> query, std::size_t k,
                                          std::size_t efSearch) const {
    validate(query);
    if (efSearch == 0) throw std::invalid_argument("HNSW efSearch must be positive");
    if (k == 0 || nodes_.empty()) return {};
    const auto count = std::min(k, nodes_.size());
    const auto ef = std::min(nodes_.size(), std::max(count, efSearch));
    Candidate entry{entry_, cosine_similarity(query, nodes_[entry_].values)};
    for (auto layer = max_level_; layer > 0; --layer) entry = greedy(query, entry, layer);
    const auto candidates = search_layer(query, std::span{&entry, 1}, ef, 0);
    std::vector<SearchResult> results;
    results.reserve(count);
    for (std::size_t i = 0; i < std::min(count, candidates.size()); ++i) {
        results.push_back({nodes_[candidates[i].node].id, candidates[i].score});
    }
    return results;
}

}  // namespace vectorpulse

#pragma once
// Diagnostic/test access only. No graph mutation or production search replacement.
#include "vectorpulse/hnsw_index.h"
#include "vectorpulse/similarity.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace vectorpulse {
struct HnswDiagnosticAccess {
    struct Layer {
        std::size_t nodes = 0, edges = 0, min_degree = 0, max_degree = 0, reachable = 0;
    };
    struct Audit {
        std::vector<Layer> layers;
        std::size_t base_reachable_without_adjacent_ids = 0;
        double reciprocal_fraction = 0;
    };
    static Audit audit(const HnswIndex& index) {
        Audit result;
        const auto n = index.nodes_.size();
        if (n == 0) return result;
        if (index.entry_ >= n || index.nodes_[index.entry_].links.size() != index.max_level_ + 1 ||
            index.index_by_id_.size() != n) throw std::runtime_error("invalid entry point or ID map");
        result.layers.resize(index.max_level_ + 1);
        for (auto& layer : result.layers) layer.min_degree = n;
        std::size_t reciprocal = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto& node = index.nodes_[i];
            if (node.links.empty() || node.links.size() > result.layers.size() ||
                node.values.size() != index.dimension_ || index.index_by_id_.at(node.id) != i)
                throw std::runtime_error("invalid node metadata");
            for (std::size_t level = 0; level < node.links.size(); ++level) {
                const auto& links = node.links[level];
                const auto cap = level == 0 ? 2 * index.config_.M : index.config_.M;
                if (links.size() > cap) throw std::runtime_error("degree cap exceeded");
                auto& layer = result.layers[level];
                ++layer.nodes; layer.edges += links.size();
                layer.min_degree = std::min(layer.min_degree, links.size());
                layer.max_degree = std::max(layer.max_degree, links.size());
                std::unordered_set<std::size_t> unique;
                for (const auto j : links) {
                    if (j >= n || j == i || !unique.insert(j).second ||
                        index.nodes_[j].links.size() <= level)
                        throw std::runtime_error("invalid, duplicate, self, or wrong-layer edge");
                    if (level == 0) {
                        const auto& back = index.nodes_[j].links[0];
                        reciprocal += std::find(back.begin(), back.end(), i) != back.end() ? 1U : 0U;
                    }
                }
            }
            const auto& base = node.links[0];
            if ((i > 0 && std::find(base.begin(), base.end(), i - 1) == base.end()) ||
                (i + 1 < n && std::find(base.begin(), base.end(), i + 1) == base.end()))
                throw std::runtime_error("base connectivity chain missing");
        }
        const auto reachable = [&](std::size_t level, bool remove_adjacent) {
            std::vector<bool> seen(n, false);
            std::vector<std::size_t> queue{index.entry_};
            seen[index.entry_] = true;
            for (std::size_t p = 0; p < queue.size(); ++p) {
                const auto i = queue[p];
                for (auto j : index.nodes_[i].links[level]) {
                    if (remove_adjacent && (i + 1 == j || j + 1 == i)) continue;
                    if (!seen[j]) { seen[j] = true; queue.push_back(j); }
                }
            }
            return queue.size();
        };
        for (std::size_t l = 0; l < result.layers.size(); ++l)
            result.layers[l].reachable = reachable(l, false);
        if (result.layers[0].reachable != n) throw std::runtime_error("disconnected base graph");
        result.base_reachable_without_adjacent_ids = reachable(0, true);
        result.reciprocal_fraction = result.layers[0].edges == 0 ? 1.0 :
            static_cast<double>(reciprocal) / static_cast<double>(result.layers[0].edges);
        return result;
    }

    // Independent sorted-vector frontier implementation on the existing graph.
    // Used only on tiny test graphs, never in timed benchmark queries.
    static std::vector<SearchResult> reference_search(const HnswIndex& index,
        std::span<const float> query, std::size_t k, std::size_t ef) {
        using Candidate = HnswIndex::Candidate;
        const auto score = [&](std::size_t node) {
            return Candidate{node, cosine_similarity(query, index.nodes_[node].values)};
        };
        const auto better = [&](Candidate a, Candidate b) {
            return a.score > b.score || (a.score == b.score &&
                index.nodes_[a.node].id < index.nodes_[b.node].id);
        };
        if (index.size() == 0 || k == 0) return {};
        ef = std::min(index.size(), std::max(ef, k));
        Candidate entry = score(index.entry_);
        for (std::size_t level = index.max_level_; level > 0; --level) {
            for (;;) {
                auto next = entry;
                for (auto j : index.nodes_[entry.node].links[level]) {
                    auto candidate = score(j);
                    if (better(candidate, next)) next = candidate;
                }
                if (next.node == entry.node) break;
                entry = next;
            }
        }
        std::vector<Candidate> frontier{entry}, retained{entry};
        std::vector<bool> seen(index.size(), false);
        seen[entry.node] = true;
        while (!frontier.empty()) {
            std::sort(frontier.begin(), frontier.end(), better);
            const auto current = frontier.front();
            frontier.erase(frontier.begin());
            std::sort(retained.begin(), retained.end(), better);
            if (retained.size() == ef && better(retained.back(), current)) break;
            for (auto j : index.nodes_[current.node].links[0]) {
                if (seen[j]) continue;
                seen[j] = true;
                auto candidate = score(j);
                if (retained.size() < ef || better(candidate, retained.back())) {
                    frontier.push_back(candidate); retained.push_back(candidate);
                    std::sort(retained.begin(), retained.end(), better);
                    if (retained.size() > ef) retained.pop_back();
                }
            }
        }
        std::sort(retained.begin(), retained.end(), better);
        std::vector<SearchResult> result;
        for (std::size_t i = 0; i < std::min(k, retained.size()); ++i)
            result.push_back({index.nodes_[retained[i].node].id, retained[i].score});
        return result;
    }

    static std::vector<std::size_t> select(const HnswIndex& index,
        std::span<const float> query, std::size_t limit) {
        std::vector<HnswIndex::Candidate> candidates;
        for (std::size_t i = 0; i < index.size(); ++i)
            candidates.push_back({i, cosine_similarity(query, index.nodes_[i].values)});
        return index.select_neighbors(std::move(candidates), limit);
    }
};
} // namespace vectorpulse

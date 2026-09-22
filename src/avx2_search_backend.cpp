#include "vectorpulse/avx2_search_backend.h"

#include "avx2_kernel.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace vectorpulse {

AVX2SearchBackend::AVX2SearchBackend(const std::size_t dimension)
    : dimension_(dimension) {
    if (!is_supported()) {
        throw std::runtime_error("AVX2 backend unsupported: requires AVX2, FMA and OS YMM support");
    }
    if (dimension_ == 0) {
        throw std::invalid_argument("vector dimension must be greater than zero");
    }
}

std::size_t AVX2SearchBackend::dimension() const noexcept {
    return dimension_;
}

std::size_t AVX2SearchBackend::size() const noexcept {
    return entries_.size();
}

void AVX2SearchBackend::add(std::string id, std::vector<float> values) {
    if (values.size() != dimension_) {
        throw std::invalid_argument("vector dimension does not match the store");
    }
    if (index_by_id_.contains(id)) {
        throw std::invalid_argument("vector ID already exists: " + id);
    }

    const std::size_t index = entries_.size();
    entries_.push_back(Entry{std::move(id), std::move(values)});

    try {
        index_by_id_.emplace(entries_.back().id, index);
    } catch (...) {
        entries_.pop_back();
        throw;
    }
}

const std::vector<float>& AVX2SearchBackend::get(const std::string_view id) const {
    const auto iterator = index_by_id_.find(std::string{id});
    if (iterator == index_by_id_.end()) {
        throw std::out_of_range("vector ID was not found: " + std::string{id});
    }
    return entries_[iterator->second].values;
}

std::vector<SearchResult> AVX2SearchBackend::search(
    const std::span<const float> query,
    const std::size_t k) const {
    if (query.size() != dimension_) {
        throw std::invalid_argument("query dimension does not match the store");
    }
    if (k == 0 || entries_.empty()) {
        return {};
    }

    std::vector<SearchResult> results;
    results.reserve(entries_.size());

    for (const Entry& entry : entries_) {
        results.push_back(SearchResult{
            entry.id,
            detail::avx2_cosine_similarity(query, entry.values),
        });
    }

    const auto better_result = [](const SearchResult& lhs, const SearchResult& rhs) {
        if (lhs.score != rhs.score) {
            return lhs.score > rhs.score;
        }
        return lhs.id < rhs.id;
    };

    const std::size_t result_count = std::min(k, results.size());
    if (result_count < results.size()) {
        std::partial_sort(
            results.begin(),
            results.begin() + static_cast<std::ptrdiff_t>(result_count),
            results.end(),
            better_result);
        results.resize(result_count);
    } else {
        std::sort(results.begin(), results.end(), better_result);
    }

    return results;
}

}  // namespace vectorpulse


#include "vectorpulse/multithreaded_scalar_search_backend.h"
#include "vectorpulse/similarity.h"

#include <algorithm>
#include <exception>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <utility>

namespace vectorpulse {
namespace {
bool better_result(const SearchResult& lhs, const SearchResult& rhs) {
    if (lhs.score != rhs.score) return lhs.score > rhs.score;
    return lhs.id < rhs.id;
}
void retain_top_k(std::vector<SearchResult>& results, std::size_t k) {
    if (k < results.size()) {
        std::partial_sort(results.begin(), results.begin() + static_cast<std::ptrdiff_t>(k),
                          results.end(), better_result);
        results.resize(k);
    } else {
        std::sort(results.begin(), results.end(), better_result);
    }
}
}  // namespace

MultithreadedScalarSearchBackend::MultithreadedScalarSearchBackend(
    std::size_t dimension, std::size_t thread_count)
    : dimension_(dimension), thread_count_(thread_count) {
    if (dimension == 0) throw std::invalid_argument("vector dimension must be greater than zero");
    if (thread_count == 0) throw std::invalid_argument("thread count must be greater than zero");
}
std::size_t MultithreadedScalarSearchBackend::dimension() const noexcept { return dimension_; }
std::size_t MultithreadedScalarSearchBackend::size() const noexcept { return entries_.size(); }
std::size_t MultithreadedScalarSearchBackend::thread_count() const noexcept { return thread_count_; }

void MultithreadedScalarSearchBackend::add(std::string id, std::vector<float> values) {
    if (values.size() != dimension_) {
        throw std::invalid_argument("vector dimension does not match the store");
    }
    if (index_by_id_.contains(id)) throw std::invalid_argument("vector ID already exists: " + id);
    const auto index = entries_.size();
    entries_.push_back(Entry{std::move(id), std::move(values)});
    try {
        index_by_id_.emplace(entries_.back().id, index);
    } catch (...) {
        entries_.pop_back();
        throw;
    }
}
const std::vector<float>& MultithreadedScalarSearchBackend::get(std::string_view id) const {
    const auto found = index_by_id_.find(std::string{id});
    if (found == index_by_id_.end()) throw std::out_of_range("vector ID was not found: " + std::string{id});
    return entries_[found->second].values;
}
std::vector<SearchResult> MultithreadedScalarSearchBackend::search(
    std::span<const float> query, std::size_t k) const {
    if (query.size() != dimension_) throw std::invalid_argument("query dimension does not match the store");
    if (k == 0 || entries_.empty()) return {};

    const auto workers = std::min(thread_count_, entries_.size());
    const auto count = std::min(k, entries_.size());
    // Each worker publishes only once into its own slot; no locks in the scan.
    std::vector<std::vector<SearchResult>> local_results(workers);
    std::vector<std::exception_ptr> errors(workers);
    const auto quotient = entries_.size() / workers;
    const auto remainder = entries_.size() % workers;
    auto scan = [&](std::size_t worker) {
        try {
            const auto begin = worker * quotient + std::min(worker, remainder);
            const auto end = begin + quotient + (worker < remainder ? 1U : 0U);
            std::vector<SearchResult> results;
            results.reserve(end - begin);
            for (auto i = begin; i < end; ++i) {
                results.push_back({entries_[i].id, cosine_similarity(query, entries_[i].values)});
            }
            retain_top_k(results, count);
            local_results[worker] = std::move(results);
        } catch (...) {
            errors[worker] = std::current_exception();
        }
    };
    {
        // RAII also joins already-started threads if subsequent creation fails.
        std::vector<std::jthread> threads;
        threads.reserve(workers - 1);
        for (std::size_t worker = 1; worker < workers; ++worker) threads.emplace_back(scan, worker);
        scan(0);
    }
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    if (workers == 1) return std::move(local_results.front());

    std::size_t candidate_count = 0;
    for (const auto& local : local_results) candidate_count += local.size();
    std::vector<SearchResult> results;
    results.reserve(candidate_count);
    for (auto& local : local_results) {
        results.insert(results.end(), std::make_move_iterator(local.begin()),
                       std::make_move_iterator(local.end()));
    }
    retain_top_k(results, count);
    return results;
}
}  // namespace vectorpulse

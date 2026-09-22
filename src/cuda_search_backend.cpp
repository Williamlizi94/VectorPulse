#include "vectorpulse/cuda_search_backend.h"
#include "cuda_runtime_bridge.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vectorpulse {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void require_finite(std::span<const float> values, CudaKernel kernel) {
    const bool fp32 = kernel == CudaKernel::NaiveFP32 || kernel == CudaKernel::BlockParallelFP32;
    // Conservative FP32 domain: normal products and headroom for accumulation.
    const double lower = std::sqrt(static_cast<double>(std::numeric_limits<float>::min()));
    const double upper = std::sqrt(static_cast<double>(std::numeric_limits<float>::max()) /
                                   (4.0 * static_cast<double>(values.size())));
    for (float value : values) {
        if (!std::isfinite(value)) throw std::invalid_argument("CUDA backend requires finite vector values");
        const auto magnitude = std::abs(static_cast<double>(value));
        if (fp32 && magnitude != 0.0 && (magnitude < lower || magnitude > upper)) {
            throw std::invalid_argument("FP32 input magnitude outside safe accumulation range; rescale or use FP64");
        }
    }
}
}
bool CudaSearchBackend::is_supported() { return device_info().available; }
CudaSearchBackend::CudaSearchBackend(std::size_t dimension, CudaStorageMode mode, CudaKernel kernel)
    : dimension_(dimension), mode_(mode), kernel_(kernel) {
    if (dimension == 0) throw std::invalid_argument("vector dimension must be greater than zero");
    if (kernel != CudaKernel::Naive && mode != CudaStorageMode::Persistent) {
        throw std::invalid_argument("selected CUDA kernel requires persistent storage");
    }
    const auto info = device_info();
    if (!info.available) throw std::runtime_error("CUDA backend unavailable: " + info.reason);
}
std::size_t CudaSearchBackend::dimension() const noexcept { return dimension_; }
std::size_t CudaSearchBackend::size() const noexcept { return entries_.size(); }
void CudaSearchBackend::add(std::string id, std::vector<float> values) {
    if (values.size() != dimension_) throw std::invalid_argument("vector dimension does not match the store");
    require_finite(values, kernel_);
    if (index_by_id_.contains(id)) throw std::invalid_argument("vector ID already exists: " + id);
    const auto index = entries_.size();
    entries_.push_back({std::move(id), std::move(values)});
    try { index_by_id_.emplace(entries_.back().id, index); }
    catch (...) { entries_.pop_back(); throw; }
}
const std::vector<float>& CudaSearchBackend::get(std::string_view id) const {
    const auto found = index_by_id_.find(std::string{id});
    if (found == index_by_id_.end()) throw std::out_of_range("vector ID was not found: " + std::string{id});
    return entries_[found->second].values;
}
std::shared_ptr<const detail::CudaIndex> CudaSearchBackend::acquire_index(
    CudaIndexBuildTimings& timings) const {
    // Insertions are externally excluded. Serialize only cache construction/publication;
    // concurrent readers retain independent shared ownership after unlocking.
    const std::lock_guard lock(index_mutex_);
    if (entries_.empty() || mode_ == CudaStorageMode::PerQuery) return {};
    if (gpu_index_ && indexed_count_ == entries_.size()) return gpu_index_;
    if (entries_.size() > std::numeric_limits<std::size_t>::max() / sizeof(float) / dimension_) {
        throw std::length_error("CUDA vector matrix byte size overflows size_t");
    }
    const auto start = Clock::now();
    {
        std::vector<float> flat;
        flat.reserve(entries_.size() * dimension_);
        for (const auto& entry : entries_) flat.insert(flat.end(), entry.values.begin(), entry.values.end());
        timings.flatten_ms = milliseconds(start);
        // Transactional replacement: failures leave the previous snapshot intact and
        // the count mismatch forces a retry; stale data is never searched.
        auto replacement = detail::cuda_build_index(flat.data(), entries_.size(), dimension_, timings);
        gpu_index_ = std::move(replacement);
        indexed_count_ = entries_.size();
        timings.rebuilt = true;
    }
    timings.total_ms = milliseconds(start);
    return gpu_index_;
}
CudaIndexBuildTimings CudaSearchBackend::build_index() const {
    CudaIndexBuildTimings timings;
    static_cast<void>(acquire_index(timings));
    return timings;
}
std::vector<SearchResult> CudaSearchBackend::search(std::span<const float> query, std::size_t k) const {
    return search_profiled(query, k).results;
}
CudaProfiledSearch CudaSearchBackend::search_profiled(std::span<const float> query, std::size_t k) const {
    const auto start = Clock::now();
    if (query.size() != dimension_) throw std::invalid_argument("query dimension does not match the store");
    require_finite(query, kernel_);
    CudaProfiledSearch output;
    if (k == 0 || entries_.empty()) {
        output.timings.total_ms = milliseconds(start);
        return output;
    }
    if (entries_.size() > std::numeric_limits<std::size_t>::max() / sizeof(float) / dimension_) {
        throw std::length_error("CUDA vector matrix byte size overflows size_t");
    }
    {
        const auto flatten_start = Clock::now();
        std::vector<float> flat;
        if (mode_ == CudaStorageMode::PerQuery) {
            flat.reserve(entries_.size() * dimension_);
            for (const auto& entry : entries_) flat.insert(flat.end(), entry.values.begin(), entry.values.end());
        }
        double query_norm = 0.0;
        for (float value : query) {
            const double widened = value;
            query_norm += widened * widened;
        }
        output.timings.flatten_ms = milliseconds(flatten_start);
        std::vector<float> scores;
        if (mode_ == CudaStorageMode::Persistent) {
            const auto index = acquire_index(output.timings.index_build);
            scores = detail::cuda_persistent_scores(*index, query.data(), query_norm, kernel_, output.timings);
        } else {
            scores = detail::cuda_scores(flat.data(), query.data(), entries_.size(), dimension_,
                                        query_norm, output.timings);
        }
        const auto top_k_start = Clock::now();
        auto& results = output.results;
        results.reserve(entries_.size());
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            if (!std::isfinite(scores[i])) throw std::runtime_error("CUDA returned a nonfinite cosine score");
            results.push_back({entries_[i].id, scores[i]});
        }
        const auto better = [](const SearchResult& a, const SearchResult& b) {
            return a.score != b.score ? a.score > b.score : a.id < b.id;
        };
        const auto count = std::min(k, results.size());
        if (count < results.size()) {
            std::partial_sort(results.begin(), results.begin() + static_cast<std::ptrdiff_t>(count),
                              results.end(), better);
            results.resize(count);
        } else { std::sort(results.begin(), results.end(), better); }
        output.timings.top_k_ms = milliseconds(top_k_start);
    } // Host scratch and per-query device resources are released before ending total timing.
    output.timings.total_ms = milliseconds(start);
    return output;
}
}  // namespace vectorpulse

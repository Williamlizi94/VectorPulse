#pragma once

#include "vectorpulse/search_backend.h"
#include <unordered_map>
#include <memory>
#include <mutex>

namespace vectorpulse {
namespace detail { class CudaIndex; }
enum class CudaStorageMode { Persistent, PerQuery };
enum class CudaKernel { Naive, BlockParallel, NaiveFP32, BlockParallelFP32 };
struct CudaIndexBuildTimings {
    bool rebuilt = false;
    std::size_t database_bytes = 0;
    double flatten_ms = 0.0, h2d_ms = 0.0, total_ms = 0.0;
};
struct CudaDeviceInfo {
    bool compiled = false;
    bool available = false;
    std::string name;
    std::string reason;
    int compute_major = 0, compute_minor = 0;
    int driver_version = 0, runtime_version = 0;
    std::size_t global_memory_bytes = 0;
};
struct CudaSearchTimings {
    CudaIndexBuildTimings index_build;
    std::size_t query_h2d_bytes = 0, database_h2d_bytes = 0;
    double flatten_ms = 0.0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double top_k_ms = 0.0;
    double total_ms = 0.0;
};
struct CudaProfiledSearch {
    std::vector<SearchResult> results;
    CudaSearchTimings timings;
};

// Persistent database by default; PerQuery retains the naive transfer baseline.
// Insertion must be externally excluded from search/get/other insertion.
class CudaSearchBackend final : public SearchBackend {
public:
    static constexpr unsigned int block_size = 256;
    static constexpr float fp32_score_tolerance = 1e-5F;
    [[nodiscard]] static bool is_compiled() noexcept;
    [[nodiscard]] static CudaDeviceInfo device_info();
    [[nodiscard]] static bool is_supported();
    explicit CudaSearchBackend(std::size_t dimension,
                              CudaStorageMode mode = CudaStorageMode::Persistent,
                              CudaKernel kernel = CudaKernel::Naive);
    // Optional eager build. Search lazily builds/rebuilds after successful insertion.
    // Repeated builds without insertion and PerQuery builds are no-ops.
    [[nodiscard]] CudaIndexBuildTimings build_index() const;
    [[nodiscard]] std::size_t dimension() const noexcept override;
    [[nodiscard]] std::size_t size() const noexcept override;
    void add(std::string id, std::vector<float> values) override;
    [[nodiscard]] const std::vector<float>& get(std::string_view id) const override;
    [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query,
                                                  std::size_t k) const override;
    // Optional profiling API; ordinary SearchBackend/VectorStore API is unchanged.
    [[nodiscard]] CudaProfiledSearch search_profiled(std::span<const float> query,
                                                     std::size_t k) const;
    void for_each_vector(const VectorVisitor& visitor) const override {
        for (const auto& entry : entries_) {
            visitor(entry.id, entry.values);
        }
    }
private:
    struct Entry { std::string id; std::vector<float> values; };
    std::size_t dimension_;
    CudaStorageMode mode_;
    CudaKernel kernel_;
    mutable std::mutex index_mutex_;
    mutable std::shared_ptr<const detail::CudaIndex> gpu_index_;
    mutable std::size_t indexed_count_ = 0;
    [[nodiscard]] std::shared_ptr<const detail::CudaIndex> acquire_index(
        CudaIndexBuildTimings& timings) const;
    std::vector<Entry> entries_;
    std::unordered_map<std::string, std::size_t> index_by_id_;
};
}  // namespace vectorpulse

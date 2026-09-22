#pragma once

#include "vectorpulse/search_backend.h"
#include <unordered_map>

namespace vectorpulse {

// Read-only searches may run concurrently; callers must exclude concurrent add().
class MultithreadedAVX2SearchBackend final : public SearchBackend {
public:
    // Requires the same compiled kernel, CPU features and OS support as AVX2SearchBackend.
    [[nodiscard]] static bool is_supported() noexcept;
    // Positive thread budget, including the calling thread. Defaults to one.
    explicit MultithreadedAVX2SearchBackend(std::size_t dimension,
                                             std::size_t thread_count = 1);
    [[nodiscard]] std::size_t dimension() const noexcept override;
    [[nodiscard]] std::size_t size() const noexcept override;
    [[nodiscard]] std::size_t thread_count() const noexcept;
    void add(std::string id, std::vector<float> values) override;
    [[nodiscard]] const std::vector<float>& get(std::string_view id) const override;
    [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query,
                                                   std::size_t k) const override;
private:
    struct Entry {
        std::string id;
        std::vector<float> values;
    };
    std::size_t dimension_;
    std::size_t thread_count_;
    std::vector<Entry> entries_;
    std::unordered_map<std::string, std::size_t> index_by_id_;
};
}  // namespace vectorpulse


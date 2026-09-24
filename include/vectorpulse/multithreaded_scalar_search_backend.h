#pragma once

#include "vectorpulse/search_backend.h"
#include <unordered_map>

namespace vectorpulse {

// Read-only searches may run concurrently; callers must exclude concurrent add().
class MultithreadedScalarSearchBackend final : public SearchBackend {
public:
    // Positive thread budget, including the calling thread. Defaults to one.
    explicit MultithreadedScalarSearchBackend(std::size_t dimension,
                                             std::size_t thread_count = 1);
    [[nodiscard]] std::size_t dimension() const noexcept override;
    [[nodiscard]] std::size_t size() const noexcept override;
    [[nodiscard]] std::size_t thread_count() const noexcept;
    void add(std::string id, std::vector<float> values) override;
    [[nodiscard]] const std::vector<float>& get(std::string_view id) const override;
    [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query,
                                                   std::size_t k) const override;
    void for_each_vector(const VectorVisitor& visitor) const override {
        for (const auto& entry : entries_) {
            visitor(entry.id, entry.values);
        }
    }
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

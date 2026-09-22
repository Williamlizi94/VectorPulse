#include "vectorpulse/vector_store.h"

#include "vectorpulse/scalar_search_backend.h"

#include <stdexcept>
#include <utility>

namespace vectorpulse {

VectorStore::VectorStore(
    const std::size_t dimension,
    std::unique_ptr<SearchBackend> backend)
    : dimension_(dimension),
      backend_(backend != nullptr
                   ? std::move(backend)
                   : std::make_unique<ScalarSearchBackend>(dimension)) {
    if (dimension_ == 0) {
        throw std::invalid_argument("vector dimension must be greater than zero");
    }
    if (backend_->dimension() != dimension_) {
        throw std::invalid_argument("backend dimension does not match the store");
    }
}

std::size_t VectorStore::dimension() const noexcept {
    return dimension_;
}

std::size_t VectorStore::size() const noexcept {
    return backend_->size();
}

void VectorStore::add(std::string id, std::vector<float> values) {
    backend_->add(std::move(id), std::move(values));
}

const std::vector<float>& VectorStore::get(const std::string_view id) const {
    return backend_->get(id);
}

std::vector<SearchResult> VectorStore::search(
    const std::span<const float> query,
    const std::size_t k) const {
    return backend_->search(query, k);
}

}  // namespace vectorpulse

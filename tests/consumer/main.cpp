#include <vectorpulse/vector_index.h>
#include <vectorpulse/multithreaded_scalar_search_backend.h>
#include <vectorpulse/cuda_search_backend.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

int main() {
    vectorpulse::VectorIndex index{2,
        std::make_unique<vectorpulse::MultithreadedScalarSearchBackend>(2, 2)};
    index.add("north", {0.0F, 1.0F});
    index.add("east", {1.0F, 0.0F});
    index.add("diagonal", {1.0F, 1.0F});
    const std::vector<float> query{1.0F, 0.0F};
    const auto results = index.search(query, 2);
    if (index.dimension() != 2 || index.size() != 3 || results.size() != 2 ||
        results[0].id != "east" || results[1].id != "diagonal" ||
        std::abs(results[0].score - 1.0F) > 1e-6F ||
        std::abs(results[1].score - std::sqrt(0.5F)) > 1e-6F) {
        return 1;
    }
    // Reference CUDA symbols even in CPU-only builds to verify transitive linking.
    // Running this consumer never requires a GPU.
    std::cout << "Installed VectorPulse search passed; CUDA compiled: "
              << vectorpulse::CudaSearchBackend::is_compiled() << '\n';
}

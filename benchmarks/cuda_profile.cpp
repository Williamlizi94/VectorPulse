// Profiling-only driver: production kernels and backend algorithms are unchanged.
#include "vectorpulse/cuda_search_backend.h"
#include "vectorpulse/scalar_search_backend.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("usage: vectorpulse_cuda_profile <vector-count> <naive|block|naive32|block32>");
        const std::string count_text = argv[1], mode = argv[2];
        if (count_text.empty() || count_text.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("invalid vector count");
        const auto count = static_cast<std::size_t>(std::stoull(count_text));
        if (count == 0 || (mode != "naive" && mode != "block" && mode != "naive32" && mode != "block32")) throw std::invalid_argument("invalid arguments");
        constexpr std::size_t dimension = 768, k = 10, warmups = 3, measured = 5;
        vectorpulse::CudaSearchBackend gpu{dimension, vectorpulse::CudaStorageMode::Persistent,
            mode == "naive" ? vectorpulse::CudaKernel::Naive :
            mode == "block" ? vectorpulse::CudaKernel::BlockParallel :
            mode == "naive32" ? vectorpulse::CudaKernel::NaiveFP32 : vectorpulse::CudaKernel::BlockParallelFP32};
        vectorpulse::ScalarSearchBackend scalar{dimension};
        std::mt19937 generator{42U};
        std::uniform_real_distribution<float> distribution{-1.0F, 1.0F};
        auto generate = [&] {
            std::vector<float> values(dimension);
            for (auto& value : values) value = distribution(generator);
            return values;
        };
        for (std::size_t i = 0; i < count; ++i) {
            const auto values = generate(); const auto id = "vector-" + std::to_string(i);
            gpu.add(id, values); scalar.add(id, values);
        }
        const auto query = generate();
        const auto reference = scalar.search(query, k);
        const auto build = gpu.build_index();
        std::cout << "mode=" << mode << " vectors=" << count << " dimension=" << dimension
                  << " warmups=" << warmups << " measured=" << measured
                  << " matrix_bytes=" << build.database_bytes << std::endl;
        const double tolerance = (mode == "naive32" || mode == "block32") ?
            vectorpulse::CudaSearchBackend::fp32_score_tolerance : 1e-6;
        double checksum = 0.0, max_error = 0.0;
        for (std::size_t iteration = 0; iteration < warmups + measured; ++iteration) {
            const auto result = gpu.search_profiled(query, k);
            if (result.timings.index_build.rebuilt || result.timings.database_h2d_bytes != 0 ||
                result.timings.query_h2d_bytes != dimension * sizeof(float))
                throw std::runtime_error("unexpected database rebuild/transfer");
            if (result.results.size() != reference.size()) throw std::runtime_error("result count mismatch");
            for (std::size_t rank = 0; rank < reference.size(); ++rank) {
                const auto error = std::abs(static_cast<double>(result.results[rank].score) - reference[rank].score);
                if (result.results[rank].id != reference[rank].id || !std::isfinite(error) || error > tolerance)
                    throw std::runtime_error("scalar correctness mismatch");
                if (error > max_error) max_error = error;
                checksum += result.results[rank].score;
            }
            std::cout << "iteration=" << iteration << " kernel_ms=" << result.timings.kernel_ms
                      << " total_ms=" << result.timings.total_ms << std::endl;
        }
        std::cout << "checksum=" << checksum << " max_score_error=" << max_error << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}

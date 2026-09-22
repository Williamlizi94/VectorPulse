#include "vectorpulse/avx2_search_backend.h"
#include <cmath>
#include <memory>
#include "vectorpulse/vector_store.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::size_t vector_count = 10'000;
    std::size_t dimension = 128;
    std::size_t query_count = 100;
    std::size_t top_k = 10;
};

[[nodiscard]] std::size_t parse_positive_size(
    const char* text,
    const std::string_view option_name) {
    const std::string value{text};
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(std::string{option_name} + " must be a positive integer");
    }
    std::size_t parsed_characters = 0;
    const unsigned long long parsed = std::stoull(value, &parsed_characters);
    if (parsed_characters != value.size() || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string{option_name} +
                                    " must be a positive integer");
    }
    return static_cast<std::size_t>(parsed);
}

[[nodiscard]] Options parse_options(const int argc, char* argv[]) {
    Options options;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + std::string{argv[index]});
        }

        const std::string_view name{argv[index]};
        const std::size_t value = parse_positive_size(argv[index + 1], name);
        if (name == "--vectors") {
            options.vector_count = value;
        } else if (name == "--dimension") {
            options.dimension = value;
        } else if (name == "--queries") {
            options.query_count = value;
        } else if (name == "--top-k") {
            options.top_k = value;
        } else {
            throw std::invalid_argument("unknown option: " + std::string{name});
        }
    }
    return options;
}

[[nodiscard]] std::vector<float> random_vector(
    const std::size_t dimension,
    std::mt19937& generator) {
    std::uniform_real_distribution<float> distribution{-1.0F, 1.0F};
    std::vector<float> values(dimension);
    for (float& value : values) {
        value = distribution(generator);
    }
    return values;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const Options options = parse_options(argc, argv);
        std::mt19937 generator{42U};
        vectorpulse::VectorStore scalar{options.dimension};
        std::unique_ptr<vectorpulse::VectorStore> avx;
        if (vectorpulse::AVX2SearchBackend::is_supported()) {
            avx = std::make_unique<vectorpulse::VectorStore>(options.dimension,
                std::make_unique<vectorpulse::AVX2SearchBackend>(options.dimension));
        }
        // Generate once, copy the identical values into each backend.
        for (std::size_t index = 0; index < options.vector_count; ++index) {
            auto values = random_vector(options.dimension, generator);
            const auto id = "vector-" + std::to_string(index);
            scalar.add(id, values);
            if (avx) avx->add(id, values);
        }
        std::vector<std::vector<float>> queries;
        queries.reserve(options.query_count);
        for (std::size_t index = 0; index < options.query_count; ++index) {
            queries.push_back(random_vector(options.dimension, generator));
        }
        std::cout << "VectorPulse single-threaded CPU benchmark (seed 42)\n"
                  << "Vectors: " << options.vector_count << "\nDimensions: " << options.dimension
                  << "\nQueries: " << options.query_count << "\nTop-K: " << options.top_k << '\n';

        using Results = std::vector<std::vector<vectorpulse::SearchResult>>;
        auto measure = [&](const char* name, const vectorpulse::VectorStore& store, Results& all) {
            all.reserve(queries.size());
            // One untimed query warms code and data consistently for both backends.
            static_cast<void>(store.search(queries.front(), options.top_k));
            const auto start = std::chrono::steady_clock::now();
            for (const auto& query : queries) all.push_back(store.search(query, options.top_k));
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            double checksum = 0.0;
            // Consume every returned score and ID outside the measured region.
            std::uint64_t id_checksum = 14695981039346656037ULL;
            for (const auto& results : all) {
                for (const auto& result : results) {
                    checksum += result.score;
                    for (unsigned char c : result.id) {
                        id_checksum ^= c;
                        id_checksum *= 1099511628211ULL;
                    }
                }
            }
            std::cout << std::fixed << std::setprecision(3)
                      << "\nBackend: " << name << "\nTotal search time: " << seconds * 1000.0 << " ms\n"
                      << "Average latency: " << seconds * 1000.0 / static_cast<double>(queries.size()) << " ms\n"
                      << "QPS: " << static_cast<double>(queries.size()) / seconds << '\n'
                      << std::setprecision(9) << "Score checksum: " << checksum
                      << "\nID checksum: " << id_checksum << '\n';
            return seconds;
        };
        Results scalar_results, avx_results;
        const double scalar_seconds = measure("Scalar", scalar, scalar_results);
        if (!avx) {
            std::cout << "\nBackend: AVX2\nUnsupported: requires a built kernel, AVX2, FMA and OS YMM support.\n";
            return EXIT_SUCCESS;
        }
        const double avx_seconds = measure("AVX2", *avx, avx_results);
        std::cout << std::setprecision(3) << "Speedup vs Scalar: " << scalar_seconds / avx_seconds << "x\n";
        double max_error = 0.0;
        for (std::size_t q = 0; q < queries.size(); ++q) {
            if (scalar_results[q].size() != avx_results[q].size()) throw std::runtime_error("result count mismatch");
            for (std::size_t rank = 0; rank < scalar_results[q].size(); ++rank) {
                const auto& expected = scalar_results[q][rank];
                const auto& actual = avx_results[q][rank];
                const double error = std::abs(static_cast<double>(actual.score) - expected.score);
                if (expected.id != actual.id || !std::isfinite(error) || error > 1e-6) {
                    throw std::runtime_error("scalar/AVX2 correctness mismatch at query " +
                        std::to_string(q) + " rank " + std::to_string(rank));
                }
                if (error > max_error) max_error = error;
            }
        }
        std::cout << "Correctness: all Top-K IDs and ranks match; max score error: "
                  << std::scientific << max_error << " (tolerance 1e-6)\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n'
                  << "usage: vectorpulse_benchmark [--vectors N] [--dimension N] "
                     "[--queries N] [--top-k N]\n";
        return EXIT_FAILURE;
    }
}

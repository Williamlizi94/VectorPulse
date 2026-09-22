#include "vectorpulse/avx2_search_backend.h"
#include "vectorpulse/multithreaded_scalar_search_backend.h"
#include "vectorpulse/multithreaded_avx2_search_backend.h"
#include "vectorpulse/vector_store.h"
#include "vectorpulse/cuda_search_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {
struct Options {
    std::size_t vectors = 10'000, dimension = 128, queries = 100, k = 10;
    std::size_t repetitions = 5, warmups = 1;
    bool include_transfer_baseline = true;
    bool all_cpu_backends = true;
    std::vector<std::size_t> threads{1, 2, 4, 8, 16};
};
std::size_t positive_size(const std::string& value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("expected a positive integer: " + value);
    }
    const auto parsed = std::stoull(value);
    if (parsed == 0 || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("integer out of range: " + value);
    }
    return static_cast<std::size_t>(parsed);
}
Options parse_options(int argc, char* argv[]) {
    Options result;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::invalid_argument("missing option value");
        const std::string_view name{argv[i]};
        const std::string value{argv[i + 1]};
        if (name == "--cpu-paths") {
            if (value != "all" && value != "scalar") throw std::invalid_argument("--cpu-paths expects all or scalar");
            result.all_cpu_backends = value == "all";
        } else if (name == "--cuda-paths") {
            if (value != "all" && value != "persistent") throw std::invalid_argument("--cuda-paths expects all or persistent");
            result.include_transfer_baseline = value == "all";
        } else if (name == "--threads") {
            result.threads.clear();
            std::size_t begin = 0;
            for (;;) {
                const auto end = value.find(',', begin);
                const auto count = positive_size(value.substr(begin, end - begin));
                if (std::find(result.threads.begin(), result.threads.end(), count) != result.threads.end()) {
                    throw std::invalid_argument("duplicate thread count");
                }
                result.threads.push_back(count);
                if (end == std::string::npos) break;
                begin = end + 1;
            }
        } else {
            const auto count = positive_size(value);
            if (name == "--vectors") result.vectors = count;
            else if (name == "--dimension") result.dimension = count;
            else if (name == "--queries") result.queries = count;
            else if (name == "--top-k") result.k = count;
            else if (name == "--repetitions") result.repetitions = count;
            else if (name == "--warmups") result.warmups = count;
            else throw std::invalid_argument("unknown option: " + std::string{name});
        }
    }
    return result;
}
using Results = std::vector<std::vector<vectorpulse::SearchResult>>;
struct Backend {
    std::string name;
    std::unique_ptr<vectorpulse::VectorStore> store;
    vectorpulse::CudaSearchBackend* cuda = nullptr; // Owned by store.
    bool persistent_cuda = false;
    std::vector<vectorpulse::CudaSearchTimings> cuda_timings;
    std::vector<double> latencies;
    std::vector<double> round_means;
    double score_checksum = 0.0;
    std::uint64_t id_checksum = 14695981039346656037ULL;
    double max_error = 0.0;
    double tolerance = 1e-6;
};
void validate(const Results& reference, const Results& actual, Backend& backend, bool consume) {
    if (actual.size() != reference.size()) throw std::runtime_error("query count mismatch");
    for (std::size_t q = 0; q < reference.size(); ++q) {
        if (actual[q].size() != reference[q].size()) throw std::runtime_error("result count mismatch");
        for (std::size_t rank = 0; rank < reference[q].size(); ++rank) {
            const auto& expected = reference[q][rank];
            const auto& found = actual[q][rank];
            const auto error = std::abs(static_cast<double>(found.score) - expected.score);
            if (found.id != expected.id || !std::isfinite(error) || error > backend.tolerance) {
                throw std::runtime_error(backend.name + " correctness mismatch at query " +
                                         std::to_string(q) + " rank " + std::to_string(rank));
            }
            backend.max_error = std::max(backend.max_error, error);
            if (consume) {
                backend.score_checksum += found.score;
                for (unsigned char c : found.id) {
                    backend.id_checksum ^= c;
                    backend.id_checksum *= 1099511628211ULL;
                }
            }
        }
    }
}
struct Stats { double mean, median, minimum, maximum; };
Stats statistics(std::vector<double> samples) {
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                        static_cast<double>(samples.size());
    std::sort(samples.begin(), samples.end());
    const auto middle = samples.size() / 2;
    const double median = samples.size() % 2 == 0 ? (samples[middle - 1] + samples[middle]) / 2.0
                                                : samples[middle];
    return {mean, median, samples.front(), samples.back()};
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
        const auto options = parse_options(argc, argv);
        std::cout << "VectorPulse CPU/CUDA benchmark (input seed 42, order seed 2026)\n"
                  << "hardware_concurrency(): " << std::thread::hardware_concurrency()
                  << "\nVectors: " << options.vectors << "\nDimensions: " << options.dimension
                  << "\nQueries: " << options.queries << "\nTop-K: " << options.k
                  << "\nWarm-up rounds: " << options.warmups
                  << "\nMeasured repetitions: " << options.repetitions << std::endl;
        std::vector<Backend> backends;
        auto add_backend = [&](std::string name, std::unique_ptr<vectorpulse::VectorStore> store) {
            Backend backend;
            backend.name = std::move(name);
            backend.store = std::move(store);
            backends.push_back(std::move(backend));
        };
        add_backend("Scalar", std::make_unique<vectorpulse::VectorStore>(options.dimension));
        if (options.all_cpu_backends) {
        if (vectorpulse::AVX2SearchBackend::is_supported()) {
            add_backend("AVX2", std::make_unique<vectorpulse::VectorStore>(options.dimension,
                std::make_unique<vectorpulse::AVX2SearchBackend>(options.dimension)));
        } else {
            std::cout << "AVX2 and MultithreadedAVX2 unsupported: requires built kernel, AVX2, FMA and OS YMM support.\n";
        }
        for (const auto threads : options.threads) {
            add_backend("MultithreadedScalar[" + std::to_string(threads) + "]",
                std::make_unique<vectorpulse::VectorStore>(options.dimension,
                    std::make_unique<vectorpulse::MultithreadedScalarSearchBackend>(options.dimension, threads)));
        }
        if (vectorpulse::MultithreadedAVX2SearchBackend::is_supported()) {
            for (const auto threads : options.threads) {
                add_backend("MultithreadedAVX2[" + std::to_string(threads) + "]",
                    std::make_unique<vectorpulse::VectorStore>(options.dimension,
                        std::make_unique<vectorpulse::MultithreadedAVX2SearchBackend>(options.dimension, threads)));
            }
        }
        }
        const auto cuda_info = vectorpulse::CudaSearchBackend::device_info();
        std::cout << "CUDA compiled: " << (cuda_info.compiled ? "yes" : "no") << '\n';
        if (cuda_info.available) {
            std::cout << "CUDA device 0: " << cuda_info.name
                      << "\nCompute capability: " << cuda_info.compute_major << '.' << cuda_info.compute_minor
                      << "\nCUDA driver API version: " << cuda_info.driver_version
                      << "\nCUDA runtime version: " << cuda_info.runtime_version
                      << "\nGPU memory bytes: " << cuda_info.global_memory_bytes
                      << "\nCUDA block size: " << vectorpulse::CudaSearchBackend::block_size
                      << "\nCUDA grid blocks: " << options.vectors / vectorpulse::CudaSearchBackend::block_size +
                          (options.vectors % vectorpulse::CudaSearchBackend::block_size != 0)
                      << "\nBlock-parallel CUDA grid blocks: " << options.vectors
                      << "\nCUDA paths: naive persistent and block-parallel persistent; CPU Top-K\n";
            auto add_cuda = [&](const char* name, vectorpulse::CudaStorageMode mode, vectorpulse::CudaKernel kernel) {
                auto cuda = std::make_unique<vectorpulse::CudaSearchBackend>(options.dimension, mode, kernel);
                auto* cuda_pointer = cuda.get();
                add_backend(name, std::make_unique<vectorpulse::VectorStore>(options.dimension, std::move(cuda)));
                backends.back().cuda = cuda_pointer;
                if (kernel == vectorpulse::CudaKernel::NaiveFP32 || kernel == vectorpulse::CudaKernel::BlockParallelFP32)
                    backends.back().tolerance = vectorpulse::CudaSearchBackend::fp32_score_tolerance;
                backends.back().persistent_cuda = mode == vectorpulse::CudaStorageMode::Persistent;
            };
            if (options.include_transfer_baseline) {
                add_cuda("CUDA-Naive", vectorpulse::CudaStorageMode::PerQuery, vectorpulse::CudaKernel::Naive);
            }
            add_cuda("CUDA-Persistent", vectorpulse::CudaStorageMode::Persistent, vectorpulse::CudaKernel::Naive);
            add_cuda("CUDA-BlockParallel", vectorpulse::CudaStorageMode::Persistent, vectorpulse::CudaKernel::BlockParallel);
            add_cuda("CUDA-NaiveFP32", vectorpulse::CudaStorageMode::Persistent, vectorpulse::CudaKernel::NaiveFP32);
            add_cuda("CUDA-BlockParallelFP32", vectorpulse::CudaStorageMode::Persistent, vectorpulse::CudaKernel::BlockParallelFP32);
        } else { std::cout << "CUDA unavailable: " << cuda_info.reason << '\n'; }
        std::mt19937 generator{42U};
        std::uniform_real_distribution<float> distribution{-1.0F, 1.0F};
        auto generate = [&] {
            std::vector<float> values(options.dimension);
            for (auto& value : values) value = distribution(generator);
            return values;
        };
        // Generate once and copy to each store. No generation/insertion is timed.
        for (std::size_t i = 0; i < options.vectors; ++i) {
            const auto values = generate();
            const auto id = "vector-" + std::to_string(i);
            for (auto& backend : backends) backend.store->add(id, values);
        }
        for (auto& backend : backends) {
            if (!backend.cuda) continue;
            const auto build = backend.cuda->build_index();
            if (build.rebuilt) {
                std::cout << backend.name << " index build: bytes=" << build.database_bytes
                          << std::fixed << std::setprecision(6) << " flatten_ms=" << build.flatten_ms
                          << " H2D_ms=" << build.h2d_ms << " total_ms=" << build.total_ms << std::endl;
            }
        }
        std::vector<std::vector<float>> queries;
        queries.reserve(options.queries);
        for (std::size_t i = 0; i < options.queries; ++i) queries.push_back(generate());
        Results reference;
        reference.reserve(queries.size());
        for (const auto& query : queries) reference.push_back(backends.front().store->search(query, options.k));
        std::cout << "Untimed scalar reference complete.\n";

        auto run = [&](Backend& backend, bool measured) {
            Results results(queries.size());
            std::vector<double> elapsed(queries.size());
            for (std::size_t q = 0; q < queries.size(); ++q) {
                const auto start = std::chrono::steady_clock::now();
                vectorpulse::CudaSearchTimings cuda_timing;
                if (backend.cuda) {
                    auto profiled = backend.cuda->search_profiled(queries[q], options.k);
                    results[q] = std::move(profiled.results);
                    cuda_timing = profiled.timings;
                } else { results[q] = backend.store->search(queries[q], options.k); }
                elapsed[q] = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                if (backend.persistent_cuda &&
                    (cuda_timing.index_build.rebuilt || cuda_timing.database_h2d_bytes != 0 ||
                     cuda_timing.query_h2d_bytes != options.dimension * sizeof(float))) {
                    throw std::runtime_error("persistent query unexpectedly rebuilt/uploaded the database");
                }
                if (measured && backend.cuda) backend.cuda_timings.push_back(cuda_timing);
            }
            validate(reference, results, backend, measured);
            if (measured) {
                const auto round = statistics(elapsed);
                backend.round_means.push_back(round.mean);
                backend.latencies.insert(backend.latencies.end(), elapsed.begin(), elapsed.end());
                std::cout << backend.name << " mean ms/query: " << std::fixed << std::setprecision(3)
                          << round.mean << std::endl;
            }
        };
        for (std::size_t round = 0; round < options.warmups; ++round) {
            for (auto& backend : backends) run(backend, false);
            std::cout << "Warm-up round " << round + 1 << " complete." << std::endl;
        }
        std::vector<std::size_t> order(backends.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::mt19937 order_generator{2026U};
        for (std::size_t round = 0; round < options.repetitions; ++round) {
            std::shuffle(order.begin(), order.end(), order_generator);
            std::cout << "Measured round " << round + 1 << std::endl;
            for (const auto index : order) run(backends[index], true);
        }
        const auto scalar_mean = statistics(backends.front().latencies).mean;
        const auto avx_backend = std::find_if(backends.begin(), backends.end(),
            [](const Backend& backend) { return backend.name == "AVX2"; });
        for (const auto& backend : backends) {
            const auto stats = statistics(backend.latencies);
            const auto rounds = statistics(backend.round_means);
            std::cout << "\nBackend: " << backend.name << "\nMeasured queries: " << backend.latencies.size()
                      << std::fixed << std::setprecision(3)
                      << "\nLatency ms/query (mean / median / min / max): " << stats.mean << " / "
                      << stats.median << " / " << stats.minimum << " / " << stats.maximum
                      << "\nRound mean ms/query (median / min / max): " << rounds.median << " / "
                      << rounds.minimum << " / " << rounds.maximum
                      << "\nQPS: " << 1000.0 / stats.mean
                      << "\nSpeedup vs Scalar: " << scalar_mean / stats.mean << "x"
                      << std::setprecision(9) << "\nScore checksum (all measured results): " << backend.score_checksum
                      << "\nID checksum (all measured results): " << backend.id_checksum
                      << "\nCorrectness: every ID/rank matches scalar; max score error: "
                      << std::scientific << backend.max_error << " (tolerance " << backend.tolerance << ")\n";
            if (backend.cuda) {
                auto print_phase = [&](const char* label, double vectorpulse::CudaSearchTimings::*field) {
                    std::vector<double> samples;
                    for (const auto& timing : backend.cuda_timings) samples.push_back(timing.*field);
                    const auto phase = statistics(samples);
                    std::cout << std::fixed << std::setprecision(6) << "CUDA " << label
                              << " ms (mean / median / min / max): " << phase.mean << " / "
                              << phase.median << " / " << phase.minimum << " / " << phase.maximum << '\n';
                };
                print_phase("flatten + query norm (host)", &vectorpulse::CudaSearchTimings::flatten_ms);
                print_phase("query-path H2D (host, synchronized)", &vectorpulse::CudaSearchTimings::h2d_ms);
                print_phase("kernel-only (CUDA events)", &vectorpulse::CudaSearchTimings::kernel_ms);
                print_phase("D2H scores (host)", &vectorpulse::CudaSearchTimings::d2h_ms);
                print_phase("CPU Top-K", &vectorpulse::CudaSearchTimings::top_k_ms);
                print_phase("profiled end-to-end", &vectorpulse::CudaSearchTimings::total_ms);
                std::cout << "CUDA per-query H2D bytes: database=" << backend.cuda_timings.front().database_h2d_bytes
                          << " query=" << backend.cuda_timings.front().query_h2d_bytes << '\n';
                std::cout << "CUDA QPS and speedup use END-TO-END query latency, not kernel-only time.\n";
            }
        }
        if (avx_backend != backends.end()) {
            const auto avx_mean = statistics(avx_backend->latencies).mean;
            std::cout << "\nSpeedup versus single-thread AVX2 (mean latency):\n";
            for (const auto& backend : backends) {
                if (backend.name.starts_with("MultithreadedAVX2[")) {
                    std::cout << backend.name << ": " << std::fixed << std::setprecision(3)
                              << avx_mean / statistics(backend.latencies).mean << "x\n";
                }
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\nusage: vectorpulse_benchmark [--vectors N] "
                     "[--dimension N] [--queries N] [--top-k N] [--warmups N] "
                     "[--repetitions N] [--threads 1,2,4,8,16] [--cuda-paths all|persistent] [--cpu-paths all|scalar]\n";
        return EXIT_FAILURE;
    }
}

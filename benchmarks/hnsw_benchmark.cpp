#include "vectorpulse/avx2_search_backend.h"
#include "vectorpulse/multithreaded_avx2_search_backend.h"
#include "vectorpulse/cuda_search_backend.h"
#include "vectorpulse/hnsw_index.h"
#include "vectorpulse/similarity.h"
#include "vectorpulse/vector_index.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <memory>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#endif
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Options {
    std::size_t vectors = 10000, dimension = 64, queries = 100, k = 10;
    std::size_t M = 16, efConstruction = 200, warmups = 1, repetitions = 3;
    std::vector<std::size_t> efSearch{10, 20, 40, 80, 160, 320};
    std::uint64_t seed = 42;
    std::size_t threads = 16;
    std::string input_prefix;
};
std::size_t integer(const std::string& value, bool zero = false) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("expected an unsigned integer: " + value);
    }
    const auto parsed = std::stoull(value);
    if ((!zero && parsed == 0) || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("integer out of range: " + value);
    }
    return static_cast<std::size_t>(parsed);
}
Options parse(int argc, char** argv) {
    Options options;
    // Apply the named preset first so explicit numeric options always override it.
    for (int i = 1; i < argc; i += 2) {
        if (std::string_view(argv[i]) == "--preset") {
            if (i + 1 >= argc || std::string_view(argv[i + 1]) != "large") {
                throw std::invalid_argument("expected --preset large");
            }
            options.vectors = 1000000; options.dimension = 768; options.k = 10;
            options.efSearch = {40, 80, 160, 320};
        }
    }
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::invalid_argument("missing option value");
        const std::string_view name(argv[i]);
        const std::string value(argv[i + 1]);
        if (name == "--preset") continue;
        if (name == "--input-prefix") options.input_prefix = value;
        else if (name == "--threads") { options.threads = integer(value); }
        else if (name == "--ef-search") {
            options.efSearch.clear();
            std::size_t begin = 0;
            for (;;) {
                const auto end = value.find(',', begin);
                const auto ef = integer(value.substr(begin, end - begin));
                if (std::find(options.efSearch.begin(), options.efSearch.end(), ef) != options.efSearch.end()) {
                    throw std::invalid_argument("duplicate efSearch value");
                }
                options.efSearch.push_back(ef);
                if (end == std::string::npos) break;
                begin = end + 1;
            }
        } else if (name == "--vectors") options.vectors = integer(value);
        else if (name == "--dimension") options.dimension = integer(value);
        else if (name == "--queries") options.queries = integer(value);
        else if (name == "--top-k") options.k = integer(value);
        else if (name == "--m") options.M = integer(value);
        else if (name == "--ef-construction") options.efConstruction = integer(value);
        else if (name == "--warmups") options.warmups = integer(value, true);
        else if (name == "--repetitions") options.repetitions = integer(value);
        else if (name == "--seed") options.seed = integer(value, true);
        else throw std::invalid_argument("unknown option: " + std::string{name});
    }
    return options;
}
double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct Memory { std::int64_t rss = -1, peak = -1; };
Memory memory_usage() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return {static_cast<std::int64_t>(counters.WorkingSetSize),
                static_cast<std::int64_t>(counters.PeakWorkingSetSize)};
    }
#elif defined(__linux__)
    std::ifstream status("/proc/self/status");
    Memory result;
    std::string line;
    while (std::getline(status, line)) {
        std::istringstream fields(line);
        std::string key;
        std::int64_t kib = 0;
        if (fields >> key >> kib) {
            if (key == "VmRSS:") result.rss = kib * 1024;
            if (key == "VmHWM:") result.peak = kib * 1024;
        }
    }
    return result;
#endif
    return {};
}

struct Run {
    std::size_t ef = 0; // Zero denotes exact search, not an HNSW ef value.
    std::vector<double> latencies;
    double hits = 0.0;
    double score_checksum = 0.0;
    std::uint64_t id_checksum = 14695981039346656037ULL;
};
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "vectorpulse_hnsw_benchmark [--vectors N] [--dimension D] [--queries Q]\n"
                         "  [--top-k K] [--m M] [--ef-construction EF] [--ef-search 10,20,40,80,160,320]\n"
                         "  [--input-prefix PATH] (little-endian .vectors.f32/.queries.f32; exports .truth.csv)\n"
                         "  [--warmups 1] [--repetitions 3] [--seed 42] [--threads 16]\n"
                         "  [--preset large] (1M vectors, 768 dimensions, K=10, efSearch=40,80,160,320)\n";
            return 0;
        }
        const auto options = parse(argc, argv);
        // Reject graph options before allocating the dataset.
        (void)vectorpulse::HnswIndex{options.dimension,
            {options.M, options.efConstruction, options.efSearch.front(), options.seed}};
        std::mt19937_64 random(options.seed);
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
        std::vector<std::vector<float>> vectors(options.vectors, std::vector<float>(options.dimension));
        std::vector<std::vector<float>> queries(options.queries, std::vector<float>(options.dimension));
        if (options.input_prefix.empty()) {
            for (auto& values : vectors) for (auto& value : values) value = distribution(random);
            for (auto& values : queries) for (auto& value : values) value = distribution(random);
        } else {
            static_assert(std::endian::native == std::endian::little);
            auto read = [&](const std::string& suffix, auto& rows) {
                std::ifstream input(options.input_prefix + suffix, std::ios::binary);
                if (!input) throw std::runtime_error("cannot open embedding input " + suffix);
                for (auto& row : rows) {
                    input.read(reinterpret_cast<char*>(row.data()),
                               static_cast<std::streamsize>(row.size() * sizeof(float)));
                    if (!input) throw std::runtime_error("truncated embedding input " + suffix);
                    for (float v : row) if (!std::isfinite(v))
                        throw std::runtime_error("nonfinite embedding input");
                }
                if (input.peek() != std::char_traits<char>::eof())
                    throw std::runtime_error("embedding input size mismatch " + suffix);
            };
            read(".vectors.f32", vectors);
            read(".queries.f32", queries);
        }
        std::vector<std::string> ids;
        std::unordered_map<std::string, std::size_t> row_by_id;
        for (std::size_t i = 0; i < vectors.size(); ++i) {
            ids.push_back(std::to_string(i));
            row_by_id.emplace(ids.back(), i);
        }
        std::cout << "VectorPulse HNSW / exact backend benchmark\n"
                  << "vectors=" << options.vectors << " dimension=" << options.dimension
                  << " queries=" << options.queries << " k=" << options.k
                  << " M=" << options.M << " efConstruction=" << options.efConstruction
                  << " seed=" << options.seed << " warmups=" << options.warmups
                  << " repetitions=" << options.repetitions << " threads=" << options.threads
                  << " hardware_threads=" << std::thread::hardware_concurrency() << '\n';
        std::cout << "Sequential backend lifetimes reuse identical data/queries; scalar ground truth is retained.\n"
                     "CUDA uses persistent BlockParallel FP64; build includes eager upload.\n"
                     "RSS is whole-process resident memory, not isolated index allocation; peak is process-lifetime.\n"
                     "Memory unavailable on this platform is reported as -1. GPU bytes count database only.\n"
                  << "dataset_vector_payload_bytes=" << static_cast<double>(options.vectors) *
                        static_cast<double>(options.dimension) * sizeof(float) << '\n'
                  << "backend,efSearch,effective_ef,build_ms,mean_ms,median_ms,p95_ms,QPS,Recall@K,"
                     "rss_before_build_bytes,rss_after_build_bytes,rss_after_queries_bytes,"
                     "process_peak_rss_bytes,gpu_database_bytes,score_checksum,id_checksum\n";
        std::cout << std::fixed << std::setprecision(6) << std::flush;
        std::vector<std::vector<vectorpulse::SearchResult>> reference;
        std::vector<std::unordered_set<std::string>> expected_ids;
        const auto result_count = std::min(options.k, options.vectors);
        auto evaluate = [&](auto& index, Run& run, bool measured, bool scalar_scores) {
            for (std::size_t q = 0; q < queries.size(); ++q) {
                const auto start = Clock::now();
                const auto results = index.search(queries[q], options.k);
                const auto end = Clock::now();
                if (measured) run.latencies.push_back(milliseconds(start, end));
                // Scoring checks and Recall@K are outside timing. Never rerank results.
                if (results.size() != result_count) throw std::runtime_error("unexpected result count");
                std::unordered_set<std::string> seen;
                std::size_t hits = 0;
                for (std::size_t i = 0; i < results.size(); ++i) {
                    const auto& result = results[i];
                    if (!seen.insert(result.id).second) throw std::runtime_error("duplicate result ID");
                    const auto row = row_by_id.at(result.id);
                    const auto score = vectorpulse::cosine_similarity(queries[q], vectors[row]);
                    if (!std::isfinite(result.score) ||
                        (scalar_scores ? score != result.score : std::abs(score - result.score) > 1e-6F)) {
                        throw std::runtime_error("result score differs from scalar cosine beyond tolerance");
                    }
                    if (i > 0 && (results[i - 1].score < result.score ||
                        (results[i - 1].score == result.score && results[i - 1].id > result.id))) {
                        throw std::runtime_error("results are not sorted");
                    }
                    hits += expected_ids[q].contains(result.id) ? 1U : 0U;
                    // SIMD/CUDA can reorder near ties; measure their recall rather than
                    // assuming it is 1. Scalar and full-breadth HNSW must match bit-for-bit.
                    if (scalar_scores && (run.ef == 0 || std::max(run.ef, result_count) >= options.vectors)) {
                        if (result.id != reference[q][i].id || result.score != reference[q][i].score) {
                            throw std::runtime_error("scalar/full-breadth result mismatch");
                        }
                    }
                    if (measured) {
                        run.score_checksum += result.score;
                        for (const auto byte : result.id) {
                            run.id_checksum ^= static_cast<unsigned char>(byte);
                            run.id_checksum *= 1099511628211ULL;
                        }
                    }
                }
                if (measured) run.hits += static_cast<double>(hits);
            }
        };
        auto report = [&](const std::string& name, Run& run, double build_ms,
                          Memory before, Memory built, std::size_t gpu_bytes) {
            const auto memory = memory_usage();
            const auto mean = std::accumulate(run.latencies.begin(), run.latencies.end(), 0.0) /
                              static_cast<double>(run.latencies.size());
            std::sort(run.latencies.begin(), run.latencies.end());
            const auto middle = run.latencies.size() / 2;
            const auto median = run.latencies.size() % 2 != 0 ? run.latencies[middle]
                : (run.latencies[middle - 1] + run.latencies[middle]) / 2.0;
            const auto p95 = run.latencies[static_cast<std::size_t>(
                std::ceil(0.95 * static_cast<double>(run.latencies.size()))) - 1];
            const auto recall = run.hits / (static_cast<double>(run.latencies.size()) *
                                           static_cast<double>(result_count));
            std::cout << name << ',' << run.ef << ','
                      << (run.ef == 0 ? options.vectors : std::min(options.vectors, std::max(result_count, run.ef)))
                      << ',' << build_ms << ',' << mean << ',' << median << ',' << p95 << ','
                      << (mean > 0.0 ? 1000.0 / mean : 0.0) << ',' << recall << ','
                      << before.rss << ',' << built.rss << ',' << memory.rss << ',' << memory.peak
                      << ',' << gpu_bytes << ',' << run.score_checksum << ',' << run.id_checksum
                      << '\n' << std::flush;
        };
        // Keep only the generated dataset and one backend alive to bound memory.
        for (int backend = 0; backend < 4; ++backend) {
            if ((backend == 1 || backend == 2) && !vectorpulse::AVX2SearchBackend::is_supported()) {
                std::cerr << "SKIP exact " << (backend == 1 ? "AVX2" : "multithreaded AVX2")
                          << ": compiled AVX2/FMA or CPU/OS support unavailable\n";
                continue;
            }
            if (backend == 3) {
                const auto info = vectorpulse::CudaSearchBackend::device_info();
                if (!info.available) {
                    std::cerr << "SKIP ExactCUDA: " << info.reason << '\n';
                    continue;
                }
                std::cerr << "CUDA device=" << info.name << " memory_bytes=" << info.global_memory_bytes
                          << " driver=" << info.driver_version << " runtime=" << info.runtime_version << '\n';
            }
            const std::string name = backend == 0 ? "ExactScalar" : backend == 1 ? "ExactAVX2"
                : backend == 2 ? "ExactMultithreadedAVX2" : "ExactCUDA_BlockParallelFP64";
            const auto before = memory_usage();
            const auto start = Clock::now();
            std::unique_ptr<vectorpulse::SearchBackend> implementation;
            vectorpulse::CudaSearchBackend* cuda = nullptr;
            if (backend == 1) implementation = std::make_unique<vectorpulse::AVX2SearchBackend>(options.dimension);
            if (backend == 2) implementation = std::make_unique<vectorpulse::MultithreadedAVX2SearchBackend>(
                options.dimension, options.threads);
            if (backend == 3) {
                auto gpu = std::make_unique<vectorpulse::CudaSearchBackend>(options.dimension,
                    vectorpulse::CudaStorageMode::Persistent, vectorpulse::CudaKernel::BlockParallel);
                cuda = gpu.get();
                implementation = std::move(gpu);
            }
            vectorpulse::VectorIndex index{options.dimension, std::move(implementation)};
            for (std::size_t i = 0; i < vectors.size(); ++i) index.add(ids[i], vectors[i]);
            const auto gpu_bytes = cuda ? cuda->build_index().database_bytes : 0;
            const auto build_ms = milliseconds(start, Clock::now());
            const auto built = memory_usage();
            if (backend == 0) {
                for (const auto& query : queries) {
                    reference.push_back(index.search(query, options.k));
                    std::unordered_set<std::string> expected;
                    for (const auto& result : reference.back()) expected.insert(result.id);
                    expected_ids.push_back(std::move(expected));
                }
                if (!options.input_prefix.empty()) {
                    std::ofstream truth(options.input_prefix + ".truth.csv");
                    if (!truth) throw std::runtime_error("cannot write scalar truth");
                    for (const auto& row : reference) {
                        for (std::size_t i = 0; i < row.size(); ++i)
                            truth << (i ? "," : "") << row[i].id;
                        truth << '\n';
                    }
                    truth.close();
                    if (!truth) throw std::runtime_error("failed writing scalar truth");
                }
            }
            Run run;
            for (std::size_t round = 0; round < options.warmups; ++round) evaluate(index, run, false, backend == 0);
            for (std::size_t round = 0; round < options.repetitions; ++round) evaluate(index, run, true, backend == 0);
            report(name, run, build_ms, before, built, gpu_bytes);
        }
        const auto before = memory_usage();
        const auto start = Clock::now();
        vectorpulse::HnswIndex hnsw{options.dimension,
            {options.M, options.efConstruction, options.efSearch.front(), options.seed}};
        for (std::size_t i = 0; i < vectors.size(); ++i) {
            hnsw.add(ids[i], vectors[i]);
            if ((i + 1) % 10000 == 0) {
                std::cerr << "HNSW built " << i + 1 << '/' << vectors.size()
                          << " elapsed_s=" << milliseconds(start, Clock::now()) / 1000.0 << '\n';
            }
        }
        const auto build_ms = milliseconds(start, Clock::now());
        const auto built = memory_usage();
        std::vector<Run> runs;
        for (const auto ef : options.efSearch) { Run run; run.ef = ef; runs.push_back(std::move(run)); }
        for (auto& run : runs) {
            hnsw.set_ef_search(run.ef);
            for (std::size_t round = 0; round < options.warmups; ++round) evaluate(hnsw, run, false, true);
        }
        std::vector<std::size_t> order(runs.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::mt19937 ordering(2026);
        for (std::size_t round = 0; round < options.repetitions; ++round) {
            std::shuffle(order.begin(), order.end(), ordering);
            for (const auto i : order) {
                hnsw.set_ef_search(runs[i].ef);
                evaluate(hnsw, runs[i], true, true);
            }
        }
        for (auto& run : runs) report("HNSW", run, build_ms, before, built, 0);
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}

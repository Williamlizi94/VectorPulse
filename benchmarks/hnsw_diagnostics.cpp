#include "hnsw_diagnostics_support.h"
#include "vectorpulse/vector_index.h"
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string_view>

namespace {
using Clock = std::chrono::steady_clock;
struct Options {
    std::size_t n = 100000, d = 128, queries = 100, M = 16, efc = 200, repeats = 3;
    std::size_t clusters = 100;
    std::string dataset = "random", export_prefix;
};
std::size_t positive(const std::string& s) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected positive integer");
    const auto value = std::stoull(s);
    if (value == 0 || value > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("integer out of range");
    return static_cast<std::size_t>(value);
}
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::invalid_argument("missing option value");
        const std::string_view key{argv[i]};
        const std::string value{argv[i + 1]};
        if (key == "--dataset") o.dataset = value;
        else if (key == "--export-prefix") o.export_prefix = value;
        else if (key == "--vectors") o.n = positive(value);
        else if (key == "--dimension") o.d = positive(value);
        else if (key == "--queries") o.queries = positive(value);
        else if (key == "--m") o.M = positive(value);
        else if (key == "--ef-construction") o.efc = positive(value);
        else if (key == "--repetitions") o.repeats = positive(value);
        else if (key == "--clusters") o.clusters = positive(value);
        else throw std::invalid_argument("unknown option");
    }
    if (o.n < 10) throw std::invalid_argument("at least 10 vectors required for Recall@10");
    if (o.dataset != "random" && o.dataset != "clustered") throw std::invalid_argument("unknown dataset");
    (void)vectorpulse::HnswIndex{o.d, {o.M, o.efc, 40, 42}};
    return o;
}
double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void normalize(std::vector<float>& vector) {
    double norm = 0;
    for (float x : vector) norm += static_cast<double>(x) * x;
    norm = std::sqrt(norm);
    for (auto& x : vector) x = static_cast<float>(x / norm);
}
std::uint64_t checksum(const std::vector<std::vector<float>>& values) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto& row : values) for (float x : row) {
        hash ^= std::bit_cast<std::uint32_t>(x);
        hash *= 1099511628211ULL;
    }
    return hash;
}
void write_vectors(const std::string& path, const std::vector<std::vector<float>>& rows) {
    std::ofstream out(path, std::ios::binary);
    for (const auto& row : rows)
        out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
    if (!out) throw std::runtime_error("cannot write diagnostic vectors");
}
}
int main(int argc, char** argv) {
    try {
        const auto o = parse(argc, argv);
        std::mt19937_64 random(42);
        std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        std::vector<std::vector<float>> centers(o.clusters, std::vector<float>(o.d));
        if (o.dataset == "clustered") {
            for (auto& row : centers) { for (auto& x : row) x = normal(random); normalize(row); }
        }
        std::vector<std::vector<float>> vectors(o.n, std::vector<float>(o.d));
        std::vector<std::vector<float>> queries(o.queries, std::vector<float>(o.d));
        std::vector<std::size_t> labels(o.n), query_labels(o.queries);
        const auto fill = [&](auto& rows, auto& groups) {
            for (std::size_t i = 0; i < rows.size(); ++i) {
                auto& row = rows[i];
                if (o.dataset == "random") {
                    for (auto& x : row) x = uniform(random);
                } else {
                    // Round-robin interleaves clusters, avoiding grouped insertion.
                    const auto cluster = i % o.clusters;
                    groups[i] = cluster;
                    const auto sigma = 0.25 / std::sqrt(static_cast<double>(o.d));
                    for (std::size_t j = 0; j < o.d; ++j)
                        row[j] = centers[cluster][j] + static_cast<float>(sigma * normal(random));
                    normalize(row);
                }
            }
        };
        fill(vectors, labels); fill(queries, query_labels);
        const auto data_hash = checksum(vectors), query_hash = checksum(queries);
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "CONFIG,dataset=" << o.dataset << ",N=" << o.n << ",D=" << o.d
                  << ",K=10,M=" << o.M << ",efConstruction=" << o.efc
                  << ",queries=" << o.queries << ",repeats=" << o.repeats
                  << ",seed=42,data_hash=" << data_hash << ",query_hash=" << query_hash << '\n' << std::flush;
        std::vector<std::vector<vectorpulse::SearchResult>> truth;
        std::vector<std::unordered_set<std::string>> truth_ids;
        double exact_ms = 0, top1 = 0, top10 = 0, gap = 0, same_cluster = 0;
        {
            vectorpulse::VectorIndex exact{o.d};
            for (std::size_t i = 0; i < vectors.size(); ++i) exact.add(std::to_string(i), vectors[i]);
            for (std::size_t q = 0; q < queries.size(); ++q) {
                const auto start = Clock::now();
                auto found = exact.search(queries[q], 10);
                exact_ms += ms(start);
                std::unordered_set<std::string> ids;
                for (const auto& hit : found) {
                    ids.insert(hit.id);
                    same_cluster += labels[std::stoull(hit.id)] == query_labels[q] ? 1.0 : 0.0;
                }
                top1 += found[0].score; top10 += found[9].score;
                gap += found[0].score - found[9].score;
                truth_ids.push_back(std::move(ids)); truth.push_back(std::move(found));
            }
        }
        const auto qcount = static_cast<double>(queries.size());
        std::cout << "TRUTH,mean_ms=" << exact_ms / qcount << ",mean_top1=" << top1 / qcount
                  << ",mean_top10=" << top10 / qcount << ",mean_top1_top10_gap=" << gap / qcount
                  << ",same_cluster_fraction=" << (o.dataset == "clustered" ? same_cluster / (qcount * 10) : -1)
                  << '\n' << std::flush;
        if (!o.export_prefix.empty()) {
            write_vectors(o.export_prefix + ".vectors.f32", vectors);
            write_vectors(o.export_prefix + ".queries.f32", queries);
            std::ofstream out(o.export_prefix + ".truth.csv");
            for (const auto& row : truth) {
                for (std::size_t k = 0; k < row.size(); ++k) out << (k ? "," : "") << row[k].id;
                out << '\n';
            }
            if (!out) throw std::runtime_error("cannot write ground truth");
        }
        const auto start = Clock::now();
        vectorpulse::HnswIndex index{o.d, {o.M, o.efc, 40, 42}};
        for (std::size_t i = 0; i < vectors.size(); ++i) {
            index.add(std::to_string(i), vectors[i]);
            if ((i + 1) % 10000 == 0)
                std::cerr << "BUILD," << o.dataset << ",M=" << o.M << ",efC=" << o.efc
                          << ",count=" << i + 1 << ",seconds=" << ms(start) / 1000 << '\n';
        }
        const auto build_ms = ms(start);
        const auto audit = vectorpulse::HnswDiagnosticAccess::audit(index);
        for (std::size_t l = 0; l < audit.layers.size(); ++l) {
            const auto& layer = audit.layers[l];
            std::cout << "LAYER," << l << ",nodes=" << layer.nodes << ",edges=" << layer.edges
                      << ",min_degree=" << layer.min_degree << ",max_degree=" << layer.max_degree
                      << ",mean_degree=" << static_cast<double>(layer.edges) / static_cast<double>(layer.nodes)
                      << ",reachable=" << layer.reachable << '\n';
        }
        std::cout << "GRAPH,base_reachable_without_adjacent_ids=" << audit.base_reachable_without_adjacent_ids
                  << ",reciprocal_fraction=" << audit.reciprocal_fraction << '\n';
        const auto validate = [&](const auto& results, std::size_t q, bool require_exact) {
            if (results.size() != 10) throw std::runtime_error("result count mismatch");
            std::unordered_set<std::string> seen;
            std::size_t hits = 0;
            for (std::size_t k = 0; k < results.size(); ++k) {
                const auto& hit = results[k];
                const auto row = std::stoull(hit.id);
                if (row >= vectors.size() || !seen.insert(hit.id).second) throw std::runtime_error("bad ID");
                const auto score = vectorpulse::cosine_similarity(queries[q], vectors[row]);
                if (std::bit_cast<std::uint32_t>(score) != std::bit_cast<std::uint32_t>(hit.score))
                    throw std::runtime_error("score bits differ from scalar");
                if (k && (results[k - 1].score < hit.score ||
                    (results[k - 1].score == hit.score && results[k - 1].id > hit.id)))
                    throw std::runtime_error("incorrect result order");
                if (require_exact && (hit.id != truth[q][k].id || hit.score != truth[q][k].score))
                    throw std::runtime_error("full breadth mismatch");
                hits += truth_ids[q].contains(hit.id) ? 1U : 0U;
            }
            return hits;
        };
        // A small full-breadth control tests actual graph traversal, not a fallback.
        for (std::size_t q = 0; q < std::min(std::size_t{3}, queries.size()); ++q)
            (void)validate(index.search(queries[q], 10, index.size()), q, true);
        std::cout << "FULL_BREADTH,queries=" << std::min(std::size_t{3}, queries.size()) << ",recall=1\n";
        const std::vector<std::size_t> efs{40, 80, 160, 320};
        struct Measure { double elapsed = 0; std::size_t hits = 0, calls = 0; std::vector<double> latency; };
        std::vector<Measure> measurements(efs.size());
        for (auto ef : efs) for (std::size_t q = 0; q < queries.size(); ++q)
            (void)validate(index.search(queries[q], 10, ef), q, ef >= index.size());
        std::vector<std::size_t> order{0, 1, 2, 3};
        std::mt19937 shuffle(2026);
        for (std::size_t r = 0; r < o.repeats; ++r) {
            std::shuffle(order.begin(), order.end(), shuffle);
            for (auto setting : order) {
                auto& m = measurements[setting];
                for (std::size_t q = 0; q < queries.size(); ++q) {
                    const auto begin = Clock::now();
                    auto results = index.search(queries[q], 10, efs[setting]);
                    const auto elapsed = ms(begin);
                    m.elapsed += elapsed; ++m.calls; m.latency.push_back(elapsed);
                    m.hits += validate(results, q, efs[setting] >= index.size());
                }
            }
        }
        std::cout << "dataset,N,D,M,efConstruction,efSearch,build_ms,mean_ms,p95_ms,QPS,Recall@10\n";
        for (std::size_t i = 0; i < efs.size(); ++i) {
            auto& m = measurements[i];
            std::sort(m.latency.begin(), m.latency.end());
            const auto mean = m.elapsed / static_cast<double>(m.calls);
            const auto p95 = m.latency[static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(m.calls))) - 1];
            std::cout << o.dataset << ',' << o.n << ',' << o.d << ',' << o.M << ',' << o.efc << ','
                      << efs[i] << ',' << build_ms << ',' << mean << ',' << p95 << ',' << 1000.0 / mean
                      << ',' << static_cast<double>(m.hits) / (10.0 * static_cast<double>(m.calls)) << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "Diagnostic failed: " << e.what() << '\n';
        return 1;
    }
}

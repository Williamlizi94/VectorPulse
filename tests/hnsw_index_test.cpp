#include "vectorpulse/hnsw_index.h"
#include "../benchmarks/hnsw_diagnostics_support.h"
#include "vectorpulse/similarity.h"
#include "vectorpulse/vector_index.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <future>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace vectorpulse {
namespace {

std::vector<std::vector<float>> data(std::size_t count, std::size_t dimension, unsigned int seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<std::vector<float>> result(count, std::vector<float>(dimension));
    for (auto& vector : result) for (auto& value : vector) value = distribution(random);
    return result;
}

void expect_equal(const std::vector<SearchResult>& actual, const std::vector<SearchResult>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_EQ(actual[i].id, expected[i].id);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i].score),
                  std::bit_cast<std::uint32_t>(expected[i].score));
    }
}

TEST(HnswIndexTest, ConstructsWithConfigAndRejectsInvalidParameters) {
    HnswIndex index{3, {.M = 4, .efConstruction = 24, .efSearch = 12, .seed = 7}};
    EXPECT_EQ(index.dimension(), 3U);
    EXPECT_EQ(index.size(), 0U);
    EXPECT_EQ(index.config().M, 4U);
    EXPECT_EQ(index.config().efConstruction, 24U);
    EXPECT_EQ(index.config().efSearch, 12U);
    EXPECT_EQ(index.config().seed, 7U);
    EXPECT_THROW((void)HnswIndex{0}, std::invalid_argument);
    EXPECT_THROW((void)(HnswIndex{3, {.M = 0}}), std::invalid_argument);
    EXPECT_THROW((void)(HnswIndex{3, {.M = 1}}), std::invalid_argument);
    EXPECT_THROW((void)(HnswIndex{3, {.M = std::numeric_limits<std::size_t>::max()}}), std::invalid_argument);
    EXPECT_THROW((void)(HnswIndex{3, {.M = 4, .efConstruction = 3}}), std::invalid_argument);
    EXPECT_THROW((void)(HnswIndex{3, {.efSearch = 0}}), std::invalid_argument);
}

TEST(HnswIndexTest, EmptySingletonAndKBoundaries) {
    HnswIndex index{2};
    const std::vector<float> query{1, 0};
    EXPECT_TRUE(index.search(query, 10).empty());
    EXPECT_THROW((void)index.search(std::vector<float>{1}, 0), std::invalid_argument);
    index.add("one", query);
    EXPECT_TRUE(index.search(query, 0).empty());
    expect_equal(index.search(query, 1), {{"one", 1.0F}});
    expect_equal(index.search(query, std::numeric_limits<std::size_t>::max()), {{"one", 1.0F}});
    EXPECT_THROW((void)index.search(query, 0, 0), std::invalid_argument);
}

TEST(HnswIndexTest, CosineOrderingZeroVectorsAndStringIdsMatchExact) {
    HnswIndex graph{2};
    VectorIndex exact{2};
    const std::vector<std::pair<std::string, std::vector<float>>> entries{
        {"negative", {-1, 0}}, {"b", {1, 1}}, {"a", {1, 1}},
        {"", {0, 0}}, {std::string{"nul\0id", 6}, {0, 1}},
        {"\xe5\x90\x91\xe9\x87\x8f", {1, 0}}};
    for (const auto& [id, values] : entries) { graph.add(id, values); exact.add(id, values); }
    for (const auto& query : {std::vector<float>{1, 0}, std::vector<float>{0, 0},
                             std::vector<float>{-1, 0}, std::vector<float>{1, -1}}) {
        for (const auto k : {0U, 1U, 3U, 100U}) expect_equal(graph.search(query, k), exact.search(query, k));
    }
}

TEST(HnswIndexTest, InvalidInsertionsDoNotChangeDataOrRandomSequence) {
    HnswIndex graph{7, {.M = 4, .efConstruction = 24, .efSearch = 8}};
    HnswIndex control{7, graph.config()};
    const auto vectors = data(100, 7, 42);
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        const auto id = std::to_string(i);
        graph.add(id, vectors[i]);
        control.add(id, vectors[i]);
        EXPECT_THROW(graph.add(id, vectors[i]), std::invalid_argument);
        EXPECT_THROW(graph.add("invalid", {1}), std::invalid_argument);
    }
    EXPECT_EQ(graph.size(), 100U);
    for (const auto& query : data(20, 7, 43)) expect_equal(graph.search(query, 10), control.search(query, 10));
}

TEST(HnswIndexTest, RejectsNonfiniteInputsEvenForEmptyOrZeroKQueries) {
    HnswIndex graph{2};
    for (const float value : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()}) {
        EXPECT_THROW(graph.add("bad", {value, 0}), std::invalid_argument);
        EXPECT_THROW((void)graph.search(std::vector<float>{value, 0}, 0), std::invalid_argument);
    }
    EXPECT_EQ(graph.size(), 0U);
}

TEST(HnswIndexTest, FiniteExtremesUseExactCosineSemantics) {
    HnswIndex graph{3};
    VectorIndex exact{3};
    const auto large = std::numeric_limits<float>::max();
    const auto small = std::numeric_limits<float>::denorm_min();
    const std::vector<std::vector<float>> vectors{{large, large, -large}, {large, -large, large},
                                                {small, 0, small}, {0, 0, 0}};
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        graph.add(std::to_string(i), vectors[i]); exact.add(std::to_string(i), vectors[i]);
    }
    for (const auto& query : vectors) expect_equal(graph.search(query, 4), exact.search(query, 4));
}

TEST(HnswIndexTest, SameSeedAndInsertionOrderAreDeterministic) {
    HnswIndex first{16, {.M = 8, .efConstruction = 64, .efSearch = 12, .seed = 123}};
    HnswIndex second{16, first.config()};
    const auto vectors = data(250, 16, 42);
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        first.add(std::to_string(i), vectors[i]); second.add(std::to_string(i), vectors[i]);
    }
    for (const auto& query : data(20, 16, 99)) {
        const auto expected = first.search(query, 10);
        expect_equal(first.search(query, 10), expected);
        expect_equal(second.search(query, 10), expected);
    }
}

TEST(HnswIndexTest, QueryBreadthCanChangeWithoutRebuilding) {
    HnswIndex graph{7, {.M = 4, .efConstruction = 32, .efSearch = 1}};
    VectorIndex exact{7};
    const auto vectors = data(100, 7, 42);
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        graph.add(std::to_string(i), vectors[i]); exact.add(std::to_string(i), vectors[i]);
    }
    const auto query = data(1, 7, 9)[0];
    EXPECT_EQ(graph.search(query, 20).size(), 20U); // Effective ef >= K.
    expect_equal(graph.search(query, 10, 100), exact.search(query, 10));
    EXPECT_EQ(graph.config().efSearch, 1U);
    graph.set_ef_search(100);
    expect_equal(graph.search(query, 10), exact.search(query, 10));
    EXPECT_THROW(graph.set_ef_search(0), std::invalid_argument);
    EXPECT_EQ(graph.config().efSearch, 100U);
}

TEST(HnswIndexTest, DegenerateDataRemainsReachableAfterHeavyPruning) {
    for (const auto& values : {std::vector<float>{0, 0}, std::vector<float>{1, 1}}) {
        HnswIndex graph{2, {.M = 2, .efConstruction = 4, .efSearch = 1}};
        VectorIndex exact{2};
        for (int i = 99; i >= 0; --i) {
            graph.add(std::to_string(i), values); exact.add(std::to_string(i), values);
        }
        expect_equal(graph.search(values, 1000), exact.search(values, 1000));
        expect_equal(graph.search(values, 10, 100), exact.search(values, 10));
    }
}

TEST(HnswIndexTest, NewVectorsBecomeSearchableAfterEarlierQueries) {
    HnswIndex graph{3};
    graph.add("old", {1, 0, 0});
    (void)graph.search(std::vector<float>{0, 0, 1}, 1);
    graph.add("new", {0, 0, 1});
    expect_equal(graph.search(std::vector<float>{0, 0, 1}, 1), {{"new", 1.0F}});
}

TEST(HnswIndexTest, HighBreadthRecallAndScoresAgainstExactGroundTruth) {
    HnswIndex graph{16, {.M = 8, .efConstruction = 80, .efSearch = 64}};
    VectorIndex exact{16};
    const auto vectors = data(500, 16, 42);
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        graph.add(std::to_string(i), vectors[i]); exact.add(std::to_string(i), vectors[i]);
    }
    std::size_t hits = 0;
    const auto queries = data(40, 16, 77);
    for (const auto& query : queries) {
        const auto reference = exact.search(query, 10);
        const auto actual = graph.search(query, 10);
        ASSERT_EQ(actual.size(), reference.size());
        std::unordered_set<std::string> expected, seen;
        for (const auto& result : reference) expected.insert(result.id);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_TRUE(seen.insert(actual[i].id).second);
            hits += expected.contains(actual[i].id) ? 1U : 0U;
            const auto score = cosine_similarity(query, vectors[std::stoull(actual[i].id)]);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(score), std::bit_cast<std::uint32_t>(actual[i].score));
            if (i > 0) EXPECT_GE(actual[i - 1].score, actual[i].score);
        }
        expect_equal(graph.search(query, 10, vectors.size()), reference);
    }
    EXPECT_GE(static_cast<double>(hits) / static_cast<double>(queries.size() * 10), 0.95);
}

TEST(HnswIndexTest, ConcurrentReadOnlySearchesHaveIndependentScratch) {
    HnswIndex graph{7, {.M = 4, .efConstruction = 32, .efSearch = 16}};
    const auto vectors = data(100, 7, 42);
    for (std::size_t i = 0; i < vectors.size(); ++i) graph.add(std::to_string(i), vectors[i]);
    const auto queries = data(8, 7, 77);
    std::vector<std::vector<SearchResult>> expected;
    std::vector<std::future<std::vector<SearchResult>>> futures;
    for (const auto& query : queries) expected.push_back(graph.search(query, 10));
    for (const auto& query : queries) {
        futures.push_back(std::async(std::launch::async, [&graph, query] { return graph.search(query, 10); }));
    }
    for (std::size_t i = 0; i < futures.size(); ++i) expect_equal(futures[i].get(), expected[i]);
}


TEST(HnswDiagnostics, AuditsEveryInsertionAndAllLayers) {
    HnswIndex graph{16, {.M = 4, .efConstruction = 40, .efSearch = 16}};
    const auto vectors = data(300, 16, 42);
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        graph.add(std::to_string(i), vectors[i]);
        const auto audit = HnswDiagnosticAccess::audit(graph);
        ASSERT_FALSE(audit.layers.empty());
        EXPECT_EQ(audit.layers[0].reachable, i + 1);
        EXPECT_EQ(audit.layers[0].nodes, i + 1);
    }
}

TEST(HnswDiagnostics, PriorityQueuesMatchIndependentSortedListTraversal) {
    HnswIndex graph{16, {.M = 4, .efConstruction = 32, .efSearch = 16}};
    const auto vectors = data(300, 16, 42);
    for (std::size_t i = 0; i < vectors.size(); ++i) graph.add(std::to_string(i), vectors[i]);
    for (const auto& query : data(20, 16, 77)) {
        for (const auto ef : {1U, 10U, 40U, 300U}) {
            expect_equal(graph.search(query, 10, ef),
                HnswDiagnosticAccess::reference_search(graph, query, 10, ef));
        }
    }
    const std::vector<float> zero(16, 0);
    expect_equal(graph.search(zero, 10, 40),
        HnswDiagnosticAccess::reference_search(graph, zero, 10, 40));
}

TEST(HnswDiagnostics, NeighborHeuristicDiversifiesAndBackfills) {
    HnswIndex graph{2};
    graph.add("a", {1, 0});
    graph.add("b", {1, 0.02F});
    graph.add("c", {0, 1});
    // b is closest to the diagonal; a is redundant with b; c is diverse.
    EXPECT_EQ(HnswDiagnosticAccess::select(graph, std::vector<float>{1, 1}, 2),
              (std::vector<std::size_t>{1, 2}));
    HnswIndex duplicates{2};
    duplicates.add("a", {1, 0});
    duplicates.add("b", {2, 0});
    duplicates.add("c", {3, 0});
    EXPECT_EQ(HnswDiagnosticAccess::select(duplicates, std::vector<float>{1, 0}, 3),
              (std::vector<std::size_t>{0, 1, 2}));
}

TEST(HnswDiagnostics, PositivePowerOfTwoScalingPreservesCosineGraphAndQueries) {
    HnswIndex original{16, {.M = 8, .efConstruction = 40, .efSearch = 16}};
    HnswIndex scaled{16, original.config()};
    const auto vectors = data(300, 16, 42);
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        original.add(std::to_string(i), vectors[i]);
        auto values = vectors[i];
        const auto scale = std::ldexp(1.0F, static_cast<int>(i % 9) - 4);
        for (auto& x : values) x *= scale;
        scaled.add(std::to_string(i), values);
    }
    for (auto query : data(20, 16, 77)) {
        const auto expected = original.search(query, 10);
        for (auto& x : query) x *= 8;
        expect_equal(scaled.search(query, 10), expected);
    }
    (void)HnswDiagnosticAccess::audit(scaled);
}

class HnswShapeTest : public ::testing::TestWithParam<std::size_t> {};
TEST_P(HnswShapeTest, FullBreadthMatchesExactAcrossDimensionsAndSmallM) {
    const auto dimension = GetParam();
    HnswIndex graph{dimension, {.M = 2, .efConstruction = 8, .efSearch = 100}};
    VectorIndex exact{dimension};
    const auto vectors = data(100, dimension, 42);
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        graph.add(std::to_string(i), vectors[i]); exact.add(std::to_string(i), vectors[i]);
    }
    for (const auto& query : data(5, dimension, 99)) {
        for (auto k : {1U, 10U, 100U, 101U}) expect_equal(graph.search(query, k), exact.search(query, k));
    }
}
INSTANTIATE_TEST_SUITE_P(Dimensions, HnswShapeTest, ::testing::Values(1U, 3U, 128U, 769U));

}  // namespace
}  // namespace vectorpulse

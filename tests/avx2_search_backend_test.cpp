#include "vectorpulse/avx2_search_backend.h"
#include "vectorpulse/scalar_search_backend.h"
#include "vectorpulse/vector_store.h"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>

namespace vectorpulse {
namespace {
class AVX2Comparison : public testing::TestWithParam<std::size_t> {};

TEST(AVX2Support, ConstructionHonorsAvailability) {
    if (!AVX2SearchBackend::is_supported()) {
        EXPECT_THROW(AVX2SearchBackend{8}, std::runtime_error);
    } else {
        EXPECT_NO_THROW(AVX2SearchBackend{8});
        EXPECT_THROW(AVX2SearchBackend{0}, std::invalid_argument);
    }
}

TEST_P(AVX2Comparison, DeterministicScoresAndEveryRankMatch) {
    if (!AVX2SearchBackend::is_supported()) GTEST_SKIP() << "AVX2/FMA unavailable";
    const auto dimension = GetParam();
    ScalarSearchBackend scalar{dimension};
    AVX2SearchBackend avx{dimension};
    std::mt19937 generator{1729};
    auto generate = [&] {
        std::vector<float> values(dimension);
        for (auto& value : values) {
            value = static_cast<float>(static_cast<int>(generator() % 2001) - 1000) / 1000.0F;
        }
        return values;
    };
    for (int i = 0; i < 128; ++i) {
        auto values = generate();
        const auto id = std::to_string(i);
        scalar.add(id, values);
        avx.add(id, values);
    }
    scalar.add("zero", std::vector<float>(dimension));
    avx.add("zero", std::vector<float>(dimension));
    for (int q = 0; q < 12; ++q) {
        const auto query = q == 0 ? std::vector<float>(dimension) : generate();
        for (std::size_t k : {0U, 1U, 10U, 129U, 200U}) {
            const auto expected = scalar.search(query, k);
            const auto actual = avx.search(query, k);
            ASSERT_EQ(actual.size(), expected.size());
            for (std::size_t i = 0; i < actual.size(); ++i) {
                ASSERT_EQ(actual[i].id, expected[i].id) << "query=" << q << " rank=" << i;
                EXPECT_NEAR(actual[i].score, expected[i].score, 1e-6F);
            }
        }
    }
}
INSTANTIATE_TEST_SUITE_P(Dimensions, AVX2Comparison,
    testing::Values(1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U,
                    13U, 14U, 15U, 16U, 17U, 31U, 384U, 767U, 768U, 769U));

TEST(AVX2Backend, StoreContractAndTies) {
    if (!AVX2SearchBackend::is_supported()) GTEST_SKIP();
    VectorStore store{9, std::make_unique<AVX2SearchBackend>(9)};
    EXPECT_EQ(store.dimension(), 9U);
    EXPECT_EQ(store.size(), 0U);
    const std::vector<float> values(9, 1.0F), zero(9, 0.0F);
    EXPECT_TRUE(store.search(values, 10).empty());
    EXPECT_THROW(store.add("bad", {1.0F}), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(store.get("missing")), std::out_of_range);
    EXPECT_THROW(static_cast<void>(store.search(std::vector<float>(8), 0)), std::invalid_argument);
    store.add("b", values);
    store.add("a", values);
    store.add("z", zero);
    EXPECT_THROW(store.add("b", zero), std::invalid_argument);
    EXPECT_EQ(store.get("b"), values);
    EXPECT_EQ(store.size(), 3U);
    const auto results = store.search(values, 10);
    ASSERT_EQ(results.size(), 3U);
    EXPECT_EQ(results[0].id, "a");
    EXPECT_EQ(results[1].id, "b");
    EXPECT_FLOAT_EQ(results[0].score, 1.0F);
    EXPECT_FLOAT_EQ(results[2].score, 0.0F);
    for (const auto& result : store.search(zero, 10)) EXPECT_FLOAT_EQ(result.score, 0.0F);
}

TEST(AVX2Backend, ExtremeFiniteValuesRetainDoublePrecisionRange) {
    if (!AVX2SearchBackend::is_supported()) GTEST_SKIP();
    for (float scale : {std::numeric_limits<float>::max(), std::numeric_limits<float>::min(), 1.0F}) {
        ScalarSearchBackend scalar{17};
        AVX2SearchBackend avx{17};
        std::vector<float> query(17, scale);
        auto opposite = query;
        for (auto& value : opposite) value = -value;
        scalar.add("same", query); avx.add("same", query);
        scalar.add("opposite", opposite); avx.add("opposite", opposite);
        const auto expected = scalar.search(query, 2), actual = avx.search(query, 2);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_EQ(actual[i].id, expected[i].id);
            EXPECT_TRUE(std::isfinite(actual[i].score));
            EXPECT_NEAR(actual[i].score, expected[i].score, 1e-6F);
        }
    }
}
}
}

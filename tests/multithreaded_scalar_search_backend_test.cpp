#include "vectorpulse/multithreaded_scalar_search_backend.h"
#include "vectorpulse/scalar_search_backend.h"
#include "vectorpulse/vector_store.h"
#include <gtest/gtest.h>
#include <future>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <tuple>

namespace vectorpulse {
namespace {
void expect_equal(const std::vector<SearchResult>& expected,
                  const std::vector<SearchResult>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t rank = 0; rank < expected.size(); ++rank) {
        EXPECT_EQ(expected[rank].id, actual[rank].id) << "rank=" << rank;
        EXPECT_NEAR(expected[rank].score, actual[rank].score, 1e-6F) << "rank=" << rank;
    }
}
class MultithreadedComparison : public testing::TestWithParam<std::tuple<std::size_t, std::size_t>> {};
TEST_P(MultithreadedComparison, MatchesScalarForEveryRankAndK) {
    const auto [dimension, threads] = GetParam();
    ScalarSearchBackend scalar{dimension};
    MultithreadedScalarSearchBackend parallel{dimension, threads};
    std::mt19937 generator{1729};
    auto generate = [&] {
        std::vector<float> values(dimension);
        for (auto& value : values) {
            value = static_cast<float>(static_cast<int>(generator() % 2001) - 1000) / 1000.0F;
        }
        return values;
    };
    // 23 is deliberately indivisible by all tested worker counts above one.
    for (int i = 0; i < 22; ++i) {
        auto values = generate();
        scalar.add(std::to_string(i), values);
        parallel.add(std::to_string(i), values);
    }
    scalar.add("zero", std::vector<float>(dimension));
    parallel.add("zero", std::vector<float>(dimension));
    for (int q = 0; q < 5; ++q) {
        const auto query = q == 0 ? std::vector<float>(dimension) : generate();
        for (std::size_t k : {0U, 1U, 10U, 23U, 40U}) {
            expect_equal(scalar.search(query, k), parallel.search(query, k));
        }
    }
}
INSTANTIATE_TEST_SUITE_P(DimensionsAndThreads, MultithreadedComparison,
    testing::Combine(testing::Values(1U, 17U, 128U, 384U, 768U),
                     testing::Values(1U, 2U, 3U, 4U, 8U, 16U, 32U)));

TEST(MultithreadedBackend, ValidationAndStoreContract) {
    EXPECT_THROW((MultithreadedScalarSearchBackend{0, 2}), std::invalid_argument);
    EXPECT_THROW((MultithreadedScalarSearchBackend{3, 0}), std::invalid_argument);
    MultithreadedScalarSearchBackend default_backend{3};
    EXPECT_EQ(default_backend.thread_count(), 1U);
    auto backend = std::make_unique<MultithreadedScalarSearchBackend>(3, 4);
    EXPECT_EQ(backend->thread_count(), 4U);
    VectorStore store{3, std::move(backend)};
    EXPECT_EQ(store.dimension(), 3U);
    EXPECT_EQ(store.size(), 0U);
    EXPECT_TRUE(store.search(std::vector<float>(3), 10).empty());
    EXPECT_TRUE(store.search(std::vector<float>(3), 0).empty());
    EXPECT_THROW(static_cast<void>(store.search(std::vector<float>(2), 0)), std::invalid_argument);
    EXPECT_THROW(store.add("bad", {1.0F}), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(store.get("missing")), std::out_of_range);
    store.add("a", {1.0F, 2.0F, 3.0F});
    EXPECT_THROW(store.add("a", {0.0F, 0.0F, 0.0F}), std::invalid_argument);
    EXPECT_EQ(store.get("a"), (std::vector<float>{1.0F, 2.0F, 3.0F}));
    EXPECT_EQ(store.size(), 1U);
    const auto result = store.search(std::vector<float>{1.0F, 2.0F, 3.0F}, 100);
    ASSERT_EQ(result.size(), 1U);
    EXPECT_EQ(result.front().id, "a");
    EXPECT_NEAR(result.front().score, 1.0F, 1e-6F);
}

TEST(MultithreadedBackend, TiesAcrossPartitionsAndRepeatedSearches) {
    for (std::size_t threads : {1U, 2U, 3U, 4U, 8U, 16U}) {
        ScalarSearchBackend scalar{128};
        MultithreadedScalarSearchBackend parallel{128, threads};
        for (int i = 12; i >= 0; --i) {
            const std::vector<float> values(128, 1.0F);
            scalar.add(std::to_string(i), values);
            parallel.add(std::to_string(i), values);
        }
        for (int repeat = 0; repeat < 3; ++repeat) {
            for (const auto& query : {std::vector<float>(128, 1.0F), std::vector<float>(128)}) {
                for (std::size_t k : {1U, 5U, 13U, 20U}) {
                    expect_equal(scalar.search(query, k), parallel.search(query, k));
                }
            }
        }
    }
}

TEST(MultithreadedBackend, ConcurrentReadOnlyQueriesHaveIndependentState) {
    ScalarSearchBackend scalar{3};
    MultithreadedScalarSearchBackend parallel{3, 4};
    for (int i = 0; i < 29; ++i) {
        const std::vector<float> values{static_cast<float>(i - 14), 1.0F, -2.0F};
        scalar.add(std::to_string(i), values);
        parallel.add(std::to_string(i), values);
    }
    const std::vector<float> a{1.0F, 2.0F, 3.0F}, b{-1.0F, 0.0F, 1.0F};
    auto first = std::async(std::launch::async, [&] { return parallel.search(a, 10); });
    auto second = std::async(std::launch::async, [&] { return parallel.search(b, 10); });
    expect_equal(scalar.search(a, 10), first.get());
    expect_equal(scalar.search(b, 10), second.get());
}

TEST(MultithreadedBackend, ExtremeFiniteValuesAndHugeThreadBudget) {
    for (float scale : {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()}) {
        ScalarSearchBackend scalar{17};
        MultithreadedScalarSearchBackend parallel{17, std::numeric_limits<std::size_t>::max()};
        for (int sign : {-1, 0, 1}) {
            const std::vector<float> values(17, static_cast<float>(sign) * scale);
            scalar.add(std::to_string(sign), values);
            parallel.add(std::to_string(sign), values);
        }
        const std::vector<float> query(17, scale);
        expect_equal(scalar.search(query, 10), parallel.search(query, 10));
    }
}
}  // namespace
}  // namespace vectorpulse

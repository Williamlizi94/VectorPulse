#include "vectorpulse/multithreaded_avx2_search_backend.h"
#include "vectorpulse/avx2_search_backend.h"
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
                  const std::vector<SearchResult>& actual, bool exact_scores = false) {
    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t rank = 0; rank < expected.size(); ++rank) {
        EXPECT_EQ(expected[rank].id, actual[rank].id) << "rank=" << rank;
        EXPECT_NEAR(expected[rank].score, actual[rank].score, 1e-6F) << "rank=" << rank;
        // Thread scheduling must not change an individual kernel's reduction order.
        if (exact_scores) EXPECT_EQ(expected[rank].score, actual[rank].score);
    }
}
TEST(MultithreadedAVX2Support, ConstructionHonorsAvailability) {
    EXPECT_EQ(MultithreadedAVX2SearchBackend::is_supported(), AVX2SearchBackend::is_supported());
    if (!MultithreadedAVX2SearchBackend::is_supported()) {
        EXPECT_THROW((MultithreadedAVX2SearchBackend{8, 4}), std::runtime_error);
        EXPECT_THROW((MultithreadedAVX2SearchBackend{8}), std::runtime_error);
        // The portable scalar backend must remain usable in a kernel-disabled build.
        ScalarSearchBackend scalar{8};
        scalar.add("one", std::vector<float>(8, 1.0F));
        EXPECT_EQ(scalar.search(std::vector<float>(8, 1.0F), 1).front().id, "one");
    } else {
        EXPECT_NO_THROW((MultithreadedAVX2SearchBackend{8, 4}));
        EXPECT_THROW((MultithreadedAVX2SearchBackend{0, 4}), std::invalid_argument);
        EXPECT_THROW((MultithreadedAVX2SearchBackend{8, 0}), std::invalid_argument);
    }
}
class MultithreadedAVX2Comparison : public testing::TestWithParam<std::tuple<std::size_t, std::size_t>> {};
TEST_P(MultithreadedAVX2Comparison, MatchesScalarAndAVX2ForEveryRankAndK) {
    if (!MultithreadedAVX2SearchBackend::is_supported()) GTEST_SKIP() << "AVX2/FMA unavailable";
    const auto [dimension, threads] = GetParam();
    ScalarSearchBackend scalar{dimension};
    AVX2SearchBackend avx{dimension};
    MultithreadedAVX2SearchBackend parallel{dimension, threads};
    std::mt19937 generator{1729};
    auto generate = [&] {
        std::vector<float> values(dimension);
        for (auto& value : values) {
            value = static_cast<float>(static_cast<int>(generator() % 2001) - 1000) / 1000.0F;
        }
        return values;
    };
    // 23 entries exercise uneven partitioning and the 32-worker clamp.
    for (int i = 0; i < 23; ++i) {
        const auto values = i == 22 ? std::vector<float>(dimension) : generate();
        const auto id = std::to_string(i);
        scalar.add(id, values); avx.add(id, values); parallel.add(id, values);
    }
    for (int q = 0; q < 5; ++q) {
        const auto query = q == 0 ? std::vector<float>(dimension) : generate();
        for (std::size_t k : {0U, 1U, 10U, 23U, 40U}) {
            const auto actual = parallel.search(query, k);
            expect_equal(scalar.search(query, k), actual);
            expect_equal(avx.search(query, k), actual, true);
        }
    }
}
INSTANTIATE_TEST_SUITE_P(DimensionsAndThreads, MultithreadedAVX2Comparison,
    testing::Combine(testing::Values(1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U,
        10U, 11U, 12U, 13U, 14U, 15U, 16U, 17U, 31U, 128U, 384U, 767U, 768U, 769U),
        testing::Values(1U, 2U, 3U, 4U, 8U, 16U, 32U)));

TEST(MultithreadedAVX2Backend, ValidationAndStoreContract) {
    if (!MultithreadedAVX2SearchBackend::is_supported()) GTEST_SKIP();
    MultithreadedAVX2SearchBackend default_backend{3};
    EXPECT_EQ(default_backend.thread_count(), 1U);
    auto backend = std::make_unique<MultithreadedAVX2SearchBackend>(3, 4);
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
    EXPECT_THROW(static_cast<void>(store.search(std::vector<float>(2), 10)), std::invalid_argument);
    const auto result = store.search(std::vector<float>{1.0F, 2.0F, 3.0F}, 100);
    ASSERT_EQ(result.size(), 1U);
    EXPECT_EQ(result.front().id, "a");
    EXPECT_NEAR(result.front().score, 1.0F, 1e-6F);
    EXPECT_TRUE(store.search(std::vector<float>(3), 0).empty());
}
TEST(MultithreadedAVX2Backend, TiesAcrossPartitionsAndRepeatedSearches) {
    if (!MultithreadedAVX2SearchBackend::is_supported()) GTEST_SKIP();
    for (std::size_t threads : {1U, 2U, 3U, 4U, 8U, 16U, 32U}) {
        ScalarSearchBackend scalar{128};
        AVX2SearchBackend avx{128};
        MultithreadedAVX2SearchBackend parallel{128, threads};
        for (int i = 12; i >= 0; --i) {
            const std::vector<float> values(128, 1.0F);
            const auto id = std::to_string(i);
            scalar.add(id, values); avx.add(id, values); parallel.add(id, values);
        }
        for (int repeat = 0; repeat < 3; ++repeat) {
            for (const auto& query : {std::vector<float>(128, 1.0F), std::vector<float>(128)}) {
                for (std::size_t k : {1U, 5U, 13U, 20U}) {
                    const auto actual = parallel.search(query, k);
                    expect_equal(scalar.search(query, k), actual);
                    expect_equal(avx.search(query, k), actual, true);
                    // Independent lexical tie oracle, crossing multiple partitions.
                    ASSERT_FALSE(actual.empty());
                    EXPECT_EQ(actual.front().id, "0");
                    if (k == 5) {
                        const std::vector<std::string> ids{"0", "1", "10", "11", "12"};
                        for (std::size_t i = 0; i < ids.size(); ++i) EXPECT_EQ(actual[i].id, ids[i]);
                    }
                }
            }
        }
    }
}
TEST(MultithreadedAVX2Backend, ConcurrentReadOnlyQueriesHaveIndependentState) {
    if (!MultithreadedAVX2SearchBackend::is_supported()) GTEST_SKIP();
    ScalarSearchBackend scalar{17};
    AVX2SearchBackend avx{17};
    MultithreadedAVX2SearchBackend parallel{17, 4};
    for (int i = 0; i < 29; ++i) {
        std::vector<float> values(17, 1.0F);
        values.front() = static_cast<float>(i - 14);
        const auto id = std::to_string(i);
        scalar.add(id, values); avx.add(id, values); parallel.add(id, values);
    }
    const std::vector<float> a(17, 1.0F), b(17, -1.0F);
    auto first = std::async(std::launch::async, [&] { return parallel.search(a, 10); });
    auto second = std::async(std::launch::async, [&] { return parallel.search(b, 10); });
    const auto first_result = first.get(), second_result = second.get();
    expect_equal(scalar.search(a, 10), first_result);
    expect_equal(avx.search(a, 10), first_result, true);
    expect_equal(scalar.search(b, 10), second_result);
    expect_equal(avx.search(b, 10), second_result, true);
}
TEST(MultithreadedAVX2Backend, ExtremeFiniteValuesAndHugeThreadBudget) {
    if (!MultithreadedAVX2SearchBackend::is_supported()) GTEST_SKIP();
    for (float scale : {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()}) {
        ScalarSearchBackend scalar{17};
        AVX2SearchBackend avx{17};
        MultithreadedAVX2SearchBackend parallel{17, std::numeric_limits<std::size_t>::max()};
        for (int sign : {-1, 0, 1}) {
            const std::vector<float> values(17, static_cast<float>(sign) * scale);
            const auto id = std::to_string(sign);
            scalar.add(id, values); avx.add(id, values); parallel.add(id, values);
        }
        const std::vector<float> query(17, scale);
        const auto actual = parallel.search(query, 10);
        expect_equal(scalar.search(query, 10), actual);
        expect_equal(avx.search(query, 10), actual, true);
    }
}
}  // namespace
}  // namespace vectorpulse

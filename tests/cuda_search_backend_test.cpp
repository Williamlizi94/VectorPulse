#include "vectorpulse/cuda_search_backend.h"
#include "vectorpulse/scalar_search_backend.h"
#include "vectorpulse/multithreaded_avx2_search_backend.h"
#include "vectorpulse/vector_store.h"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <tuple>
#include <future>

namespace vectorpulse {
namespace {
void expect_equal(const std::vector<SearchResult>& reference, const std::vector<SearchResult>& actual, float tolerance = 1e-6F) {
    ASSERT_EQ(reference.size(), actual.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_EQ(reference[i].id, actual[i].id) << "rank=" << i;
        EXPECT_TRUE(std::isfinite(actual[i].score));
        EXPECT_NEAR(reference[i].score, actual[i].score, tolerance) << "rank=" << i;
    }
}
TEST(CudaSupport, AvailabilityAndUnsupportedConstruction) {
    const auto info = CudaSearchBackend::device_info();
    EXPECT_EQ(info.compiled, CudaSearchBackend::is_compiled());
    EXPECT_EQ(info.available, CudaSearchBackend::is_supported());
    EXPECT_THROW((CudaSearchBackend{0}), std::invalid_argument);
    if (info.available) {
        EXPECT_TRUE(info.compiled);
        EXPECT_FALSE(info.name.empty());
        EXPECT_GT(info.global_memory_bytes, 0U);
        EXPECT_GT(info.runtime_version, 0);
        EXPECT_NO_THROW((CudaSearchBackend{3}));
    } else {
        EXPECT_FALSE(info.reason.empty());
        EXPECT_THROW((CudaSearchBackend{3}), std::runtime_error);
        ScalarSearchBackend scalar{3};
        scalar.add("one", {1.0F, 0.0F, 0.0F});
        EXPECT_EQ(scalar.search(std::vector<float>{1.0F, 0.0F, 0.0F}, 1).front().id, "one");
    }
}
class CudaComparison : public testing::TestWithParam<std::tuple<std::size_t, std::size_t, int>> {};
TEST_P(CudaComparison, MatchesScalarAcrossBlocksDimensionsAndK) {
    if (!CudaSearchBackend::is_supported()) GTEST_SKIP() << CudaSearchBackend::device_info().reason;
    const auto [dimension, count, mode] = GetParam();
    ScalarSearchBackend scalar{dimension};
    CudaSearchBackend gpu{dimension, mode == 0 ? CudaStorageMode::PerQuery : CudaStorageMode::Persistent,
                          mode == 2 ? CudaKernel::BlockParallel : mode == 3 ? CudaKernel::NaiveFP32 :
                          mode == 4 ? CudaKernel::BlockParallelFP32 : CudaKernel::Naive};
    const float tolerance = mode >= 3 ? CudaSearchBackend::fp32_score_tolerance : 1e-6F;
    CudaSearchBackend naive{dimension};
    std::unique_ptr<MultithreadedAVX2SearchBackend> cpu;
    if (MultithreadedAVX2SearchBackend::is_supported()) {
        cpu = std::make_unique<MultithreadedAVX2SearchBackend>(dimension, 4);
    }
    std::mt19937 generator{1729};
    auto generate = [&] {
        std::vector<float> values(dimension);
        for (auto& value : values) value = static_cast<float>(static_cast<int>(generator() % 2001) - 1000) / 1000.0F;
        return values;
    };
    for (std::size_t i = 0; i < count; ++i) {
        const auto values = i == 0 ? std::vector<float>(dimension) : generate();
        const auto id = std::to_string(i);
        scalar.add(id, values); gpu.add(id, values); naive.add(id, values);
        if (cpu) cpu->add(id, values);
    }
    for (int q = 0; q < 3; ++q) {
        const auto query = q == 0 ? std::vector<float>(dimension) : generate();
        for (const auto k : {std::size_t{0}, std::size_t{1}, std::size_t{10}, count, count + 5}) {
            const auto actual = gpu.search(query, k);
            expect_equal(scalar.search(query, k), actual, tolerance);
            expect_equal(naive.search(query, k), actual, tolerance);
            if (cpu) expect_equal(cpu->search(query, k), actual, tolerance);
        }
    }
}
INSTANTIATE_TEST_SUITE_P(DimensionsAndCounts, CudaComparison,
    testing::Combine(testing::Values(3U, 7U, 128U, 384U, 768U, 769U),
                     testing::Values(1U, 257U, 513U), testing::Values(0, 1, 2, 3, 4)));
TEST(CudaBackend, EmptyStoreValidationAndRetrieval) {
    if (!CudaSearchBackend::is_supported()) GTEST_SKIP();
    VectorStore store{3, std::make_unique<CudaSearchBackend>(3)};
    EXPECT_EQ(store.dimension(), 3U);
    EXPECT_EQ(store.size(), 0U);
    EXPECT_TRUE(store.search(std::vector<float>(3), 10).empty());
    EXPECT_TRUE(store.search(std::vector<float>(3), 0).empty());
    EXPECT_THROW(static_cast<void>(store.search(std::vector<float>(2), 0)), std::invalid_argument);
    EXPECT_THROW(store.add("bad", {1.0F}), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(store.get("missing")), std::out_of_range);
    store.add("a", {1.0F, 0.0F, 0.0F});
    EXPECT_THROW(store.add("a", {0.0F, 1.0F, 0.0F}), std::invalid_argument);
    EXPECT_EQ(store.get("a"), (std::vector<float>{1.0F, 0.0F, 0.0F}));
    const auto actual = store.search(std::vector<float>{1.0F, 0.0F, 0.0F}, 100);
    ASSERT_EQ(actual.size(), 1U);
    EXPECT_EQ(actual.front().id, "a");
    EXPECT_FLOAT_EQ(actual.front().score, 1.0F);
}
TEST(CudaBackend, ExactTiesAcrossMultipleBlocks) {
    if (!CudaSearchBackend::is_supported()) GTEST_SKIP();
    ScalarSearchBackend scalar{7};
    CudaSearchBackend gpu{7};
    for (int i = 512; i >= 0; --i) {
        const auto id = std::to_string(i);
        const std::vector<float> values(7, 1.0F);
        scalar.add(id, values); gpu.add(id, values);
    }
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (float q : {0.0F, 1.0F, -1.0F}) {
            const std::vector<float> query(7, q);
            const auto actual = gpu.search(query, 10);
            expect_equal(scalar.search(query, 10), actual);
            ASSERT_EQ(actual.size(), 10U);
            EXPECT_EQ(actual[0].id, "0");
            EXPECT_EQ(actual[1].id, "1");
            EXPECT_EQ(actual[2].id, "10");
        }
    }
}
TEST(CudaBackend, RejectsNonfiniteInputsWithoutCorruptingStore) {
    if (!CudaSearchBackend::is_supported()) GTEST_SKIP();
    CudaSearchBackend gpu{3};
    for (float bad : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        EXPECT_THROW(gpu.add("bad", {bad, 0.0F, 1.0F}), std::invalid_argument);
        EXPECT_THROW(static_cast<void>(gpu.search(std::vector<float>{bad, 0.0F, 1.0F}, 0)), std::invalid_argument);
    }
    EXPECT_EQ(gpu.size(), 0U);
    gpu.add("good", {1.0F, 0.0F, 0.0F});
    EXPECT_FLOAT_EQ(gpu.search(std::vector<float>{1.0F, 0.0F, 0.0F}, 1)[0].score, 1.0F);
}
TEST(CudaBackend, ExtremeFiniteInputsRetainReferenceRange) {
    if (!CudaSearchBackend::is_supported()) GTEST_SKIP();
    for (float scale : {std::numeric_limits<float>::max(), std::numeric_limits<float>::min(),
                       std::numeric_limits<float>::denorm_min()}) {
        ScalarSearchBackend scalar{7};
        CudaSearchBackend gpu{7};
        for (int sign : {-1, 0, 1}) {
            const std::vector<float> values(7, static_cast<float>(sign) * scale);
            scalar.add(std::to_string(sign), values); gpu.add(std::to_string(sign), values);
        }
        expect_equal(scalar.search(std::vector<float>(7, scale), 10), gpu.search(std::vector<float>(7, scale), 10));
    }
}
TEST(CudaBackend, ProfiledSearchAndInsertionAfterSearch) {
    if (!CudaSearchBackend::is_supported()) GTEST_SKIP();
    CudaSearchBackend gpu{3};
    const std::vector<float> query{1.0F, 0.0F, 0.0F};
    EXPECT_EQ(gpu.search_profiled(query, 10).timings.kernel_ms, 0.0);
    gpu.add("b", {0.0F, 1.0F, 0.0F});
    static_cast<void>(gpu.search(query, 1));
    gpu.add("a", query);
    const auto result = gpu.search_profiled(query, 2);
    expect_equal(gpu.search(query, 2), result.results);
    ASSERT_EQ(result.results.size(), 2U);
    EXPECT_EQ(result.results.front().id, "a");
    for (double ms : {result.timings.flatten_ms, result.timings.h2d_ms, result.timings.kernel_ms,
                      result.timings.d2h_ms, result.timings.top_k_ms, result.timings.total_ms}) {
        EXPECT_TRUE(std::isfinite(ms));
        EXPECT_GE(ms, 0.0);
    }
    EXPECT_EQ(gpu.search_profiled(query, 0).timings.kernel_ms, 0.0);
}
TEST(CudaPersistent, BuildsOnceAndTransfersOnlyQuery) {
    if (!CudaSearchBackend::is_supported()) GTEST_SKIP();
    CudaSearchBackend gpu{3};
    const std::vector<float> query{1, 0, 0};
    EXPECT_FALSE(gpu.build_index().rebuilt);
    gpu.add("b", query);
    EXPECT_FALSE(gpu.search_profiled(query, 0).timings.index_build.rebuilt);
    const auto build = gpu.build_index();
    EXPECT_TRUE(build.rebuilt);
    EXPECT_EQ(build.database_bytes, 3 * sizeof(float));
    EXPECT_FALSE(gpu.build_index().rebuilt);
    for (int i = 0; i < 3; ++i) {
        const auto result = gpu.search_profiled(query, 1);
        EXPECT_FALSE(result.timings.index_build.rebuilt);
        EXPECT_EQ(result.timings.database_h2d_bytes, 0U);
        EXPECT_EQ(result.timings.query_h2d_bytes, 3 * sizeof(float));
        EXPECT_EQ(result.results.front().id, "b");
    }
    EXPECT_THROW(gpu.add("b", query), std::invalid_argument);
    EXPECT_THROW(gpu.add("bad", {1}), std::invalid_argument);
    EXPECT_THROW(gpu.add("nan", {NAN, 0, 0}), std::invalid_argument);
    EXPECT_FALSE(gpu.build_index().rebuilt);
    // Multiple appends are coalesced into a single lazy rebuild; ties use IDs.
    gpu.add("a", query);
    gpu.add("c", {0, 1, 0});
    const auto updated = gpu.search_profiled(query, 10);
    EXPECT_TRUE(updated.timings.index_build.rebuilt);
    EXPECT_EQ(updated.timings.index_build.database_bytes, 9 * sizeof(float));
    ASSERT_EQ(updated.results.size(), 3U);
    EXPECT_EQ(updated.results[0].id, "a");
    EXPECT_EQ(updated.results[1].id, "b");
    EXPECT_FALSE(gpu.build_index().rebuilt);
}
TEST(CudaPersistent, ConcurrentFirstQueriesPublishOneIndex) {
    if (!CudaSearchBackend::is_supported()) GTEST_SKIP();
    CudaSearchBackend gpu{128};
    ScalarSearchBackend scalar{128};
    for (int i = 0; i < 257; ++i) {
        std::vector<float> values(128, static_cast<float>(i % 7 - 3));
        values[i % 128] = 2.0F;
        gpu.add(std::to_string(i), values);
        scalar.add(std::to_string(i), values);
    }
    std::vector<std::future<CudaProfiledSearch>> tasks;
    for (int i = 0; i < 4; ++i) {
        tasks.push_back(std::async(std::launch::async, [&, i] {
            return gpu.search_profiled(std::vector<float>(128, static_cast<float>(i - 2)), 10);
        }));
    }
    int builds = 0;
    for (int i = 0; i < 4; ++i) {
        const auto result = tasks[i].get();
        builds += result.timings.index_build.rebuilt;
        expect_equal(scalar.search(std::vector<float>(128, static_cast<float>(i - 2)), 10), result.results);
        EXPECT_EQ(result.timings.database_h2d_bytes, 0U);
    }
    EXPECT_EQ(builds, 1);
}
TEST(CudaPersistent, RepeatedRebuildsAndOwnershipLifetimes) {
    if (!CudaSearchBackend::is_supported()) GTEST_SKIP();
    const std::vector<float> query{1, 2, 3, 4, 5, 6, 7};
    for (int lifetime = 0; lifetime < 3; ++lifetime) {
        CudaSearchBackend gpu{7};
        ScalarSearchBackend scalar{7};
        for (int i = 0; i < 10; ++i) {
            const auto values = std::vector<float>(7, static_cast<float>(i - 5));
            gpu.add(std::to_string(i), values); scalar.add(std::to_string(i), values);
            const auto result = gpu.search_profiled(query, 20);
            EXPECT_TRUE(result.timings.index_build.rebuilt);
            expect_equal(scalar.search(query, 20), result.results);
            EXPECT_FALSE(gpu.build_index().rebuilt);
        }
    }
}
TEST(CudaPersistent, NaiveModeRemainsFullTransferBaseline) {
    if (!CudaSearchBackend::is_supported()) GTEST_SKIP();
    CudaSearchBackend gpu{3, CudaStorageMode::PerQuery};
    const std::vector<float> query{1, 0, 0};
    gpu.add("a", query); gpu.add("b", {0, 1, 0});
    EXPECT_FALSE(gpu.build_index().rebuilt);
    for (int i = 0; i < 2; ++i) {
        const auto result = gpu.search_profiled(query, 2);
        EXPECT_FALSE(result.timings.index_build.rebuilt);
        EXPECT_EQ(result.timings.database_h2d_bytes, 6 * sizeof(float));
        EXPECT_EQ(result.timings.query_h2d_bytes, 3 * sizeof(float));
        ASSERT_EQ(result.results.size(), 2U);
        EXPECT_EQ(result.results.front().id, "a");
    }
}

TEST(CudaBlockParallel, PersistentLifecycleAndTailDimensions) {
    if (!CudaSearchBackend::is_supported()) GTEST_SKIP();
    EXPECT_THROW((CudaSearchBackend{3, CudaStorageMode::PerQuery, CudaKernel::BlockParallel}), std::invalid_argument);
    for (const std::size_t dimension : {1U, 255U, 256U, 257U, 767U, 769U, 1025U}) {
        CudaSearchBackend block{dimension, CudaStorageMode::Persistent, CudaKernel::BlockParallel};
        CudaSearchBackend naive{dimension};
        ScalarSearchBackend scalar{dimension};
        const std::vector<float> query(dimension, 1.0F);
        EXPECT_TRUE(block.search(query, 10).empty());
        EXPECT_FALSE(block.build_index().rebuilt);
        for (const auto* id : {"b", "a", "c"}) {
            block.add(id, query); naive.add(id, query); scalar.add(id, query);
            const auto result = block.search_profiled(query, 10);
            EXPECT_TRUE(result.timings.index_build.rebuilt);
            expect_equal(scalar.search(query, 10), result.results);
            expect_equal(naive.search(query, 10), result.results);
            EXPECT_FALSE(block.build_index().rebuilt);
            EXPECT_EQ(result.timings.database_h2d_bytes, 0U);
            EXPECT_EQ(result.timings.query_h2d_bytes, dimension * sizeof(float));
        }
        EXPECT_EQ(block.search(query, 1).front().id, "a");
        EXPECT_TRUE(block.search(query, 0).empty());
        EXPECT_THROW(block.add("a", query), std::invalid_argument);
        EXPECT_FALSE(block.build_index().rebuilt);
        EXPECT_THROW(static_cast<void>(block.search(std::vector<float>{}, 0)), std::invalid_argument);
        auto bad = query; bad[0] = std::numeric_limits<float>::infinity();
        EXPECT_THROW(block.add("bad", bad), std::invalid_argument);
        EXPECT_THROW(static_cast<void>(block.search(bad, 0)), std::invalid_argument);
        expect_equal(scalar.search(std::vector<float>(dimension), 10), block.search(std::vector<float>(dimension), 10));
    }
}
TEST(CudaBlockParallel, ExtremeFiniteValuesAndCancellation) {
    if (!CudaSearchBackend::is_supported()) GTEST_SKIP();
    for (float scale : {std::numeric_limits<float>::max(), std::numeric_limits<float>::min(),
                       std::numeric_limits<float>::denorm_min(), 1.0F}) {
        CudaSearchBackend block{769, CudaStorageMode::Persistent, CudaKernel::BlockParallel};
        CudaSearchBackend naive{769};
        ScalarSearchBackend scalar{769};
        for (int row = 0; row < 8; ++row) {
            std::vector<float> values(769);
            for (std::size_t d = 0; d < values.size(); ++d) values[d] = (d % 3 == 0 ? -scale : scale);
            if (row == 0) values.assign(769, 0.0F);
            else values[row] = 0.0F;
            const auto id = std::to_string(row);
            block.add(id, values); naive.add(id, values); scalar.add(id, values);
        }
        const std::vector<float> query(769, scale);
        expect_equal(scalar.search(query, 20), block.search(query, 20));
        expect_equal(naive.search(query, 20), block.search(query, 20));
    }
}
TEST(CudaBlockParallel, ConcurrentFirstQueriesReuseIndex) {
    if (!CudaSearchBackend::is_supported()) GTEST_SKIP();
    CudaSearchBackend block{768, CudaStorageMode::Persistent, CudaKernel::BlockParallel};
    ScalarSearchBackend scalar{768};
    for (int row = 0; row < 257; ++row) {
        std::vector<float> values(768, 1.0F); values[row] = -2.0F;
        block.add(std::to_string(row), values); scalar.add(std::to_string(row), values);
    }
    std::vector<std::future<CudaProfiledSearch>> tasks;
    for (int i = 0; i < 4; ++i) tasks.push_back(std::async(std::launch::async, [&, i] {
        return block.search_profiled(std::vector<float>(768, static_cast<float>(i - 2)), 10);
    }));
    int builds = 0;
    for (int i = 0; i < 4; ++i) {
        const auto result = tasks[i].get(); builds += result.timings.index_build.rebuilt;
        expect_equal(scalar.search(std::vector<float>(768, static_cast<float>(i - 2)), 10), result.results);
    }
    EXPECT_EQ(builds, 1);
}

TEST(CudaFP32, SafeRangeValidationAndIndexUpdates) {
    if (!CudaSearchBackend::is_supported()) GTEST_SKIP();
    for (const auto kernel : {CudaKernel::NaiveFP32, CudaKernel::BlockParallelFP32}) {
        EXPECT_THROW((CudaSearchBackend{768, CudaStorageMode::PerQuery, kernel}), std::invalid_argument);
        CudaSearchBackend gpu{768, CudaStorageMode::Persistent, kernel};
        ScalarSearchBackend scalar{768};
        std::vector<float> query(768, 1.0F);
        EXPECT_TRUE(gpu.search(query, 10).empty());
        for (float bad : {std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min(),
                          std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            auto values = query; values[0] = bad;
            EXPECT_THROW(gpu.add("bad", values), std::invalid_argument);
            EXPECT_THROW(static_cast<void>(gpu.search(values, 0)), std::invalid_argument);
        }
        for (const auto* id : {"b", "a", "c"}) {
            gpu.add(id, query); scalar.add(id, query);
            const auto result = gpu.search_profiled(query, 10);
            EXPECT_TRUE(result.timings.index_build.rebuilt);
            EXPECT_EQ(result.timings.database_h2d_bytes, 0U);
            expect_equal(scalar.search(query, 10), result.results, CudaSearchBackend::fp32_score_tolerance);
            EXPECT_FALSE(gpu.build_index().rebuilt);
        }
        EXPECT_EQ(gpu.search(query, 1).front().id, "a");
        EXPECT_TRUE(gpu.search(query, 0).empty());
    }
}
TEST(CudaFP32, CancellationTailDimensionsAndSafeMagnitudeExtremes) {
    if (!CudaSearchBackend::is_supported()) GTEST_SKIP();
    for (const auto kernel : {CudaKernel::NaiveFP32, CudaKernel::BlockParallelFP32}) {
        for (const std::size_t dimension : {1U, 255U, 256U, 257U, 768U, 769U, 1025U}) {
            for (const float scale : {1e-18F, 1.0F, 1e15F}) {
                CudaSearchBackend gpu{dimension, CudaStorageMode::Persistent, kernel};
                ScalarSearchBackend scalar{dimension};
                for (int row = 0; row < 5; ++row) {
                    std::vector<float> values(dimension, 0.0F);
                    if (row != 0) for (std::size_t d = 0; d < dimension; ++d)
                        values[d] = (d % 5 < static_cast<std::size_t>(row) ? scale : -scale);
                    gpu.add(std::to_string(row), values); scalar.add(std::to_string(row), values);
                }
                for (float q : {0.0F, scale, -scale}) {
                    const std::vector<float> query(dimension, q);
                    expect_equal(scalar.search(query, 20), gpu.search(query, 20), CudaSearchBackend::fp32_score_tolerance);
                }
            }
        }
    }
}

}
}

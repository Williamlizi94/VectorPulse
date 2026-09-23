#include "vectorpulse/vector_index.h"
#include "vectorpulse/avx2_search_backend.h"
#include "vectorpulse/cuda_search_backend.h"
#include "vectorpulse/multithreaded_avx2_search_backend.h"
#include "vectorpulse/multithreaded_scalar_search_backend.h"
#include "vectorpulse/scalar_search_backend.h"

#include <gtest/gtest.h>
#include <stdexcept>

namespace vectorpulse {
namespace {

TEST(VectorIndexTest, PreservesValidationAndEdgeCases) {
    EXPECT_THROW((void)VectorIndex{0}, std::invalid_argument);
    EXPECT_THROW((void)(VectorIndex{3, std::make_unique<ScalarSearchBackend>(2)}),
                 std::invalid_argument);
    VectorIndex index{2};
    const std::vector<float> query{1.0F, 0.0F};
    EXPECT_EQ(index.dimension(), 2U);
    EXPECT_EQ(index.size(), 0U);
    EXPECT_TRUE(index.search(query, 10).empty());
    EXPECT_THROW(index.add("bad", {1.0F}), std::invalid_argument);
    EXPECT_THROW((void)index.search(std::vector<float>{1.0F}, 1), std::invalid_argument);
    index.add("original", query);
    EXPECT_THROW(index.add("original", {0.0F, 1.0F}), std::invalid_argument);
    EXPECT_EQ(index.size(), 1U);
    EXPECT_TRUE(index.search(query, 0).empty());
    const auto results = index.search(query, 10);
    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(results[0].id, "original");
    EXPECT_FLOAT_EQ(results[0].score, 1.0F);
}

class VectorIndexBackendTest : public ::testing::TestWithParam<int> {};

TEST_P(VectorIndexBackendTest, SearchesAndAddsThroughUnifiedApi) {
    std::unique_ptr<SearchBackend> backend;
    switch (GetParam()) {
    case 0: break; // Default scalar.
    case 1: backend = std::make_unique<ScalarSearchBackend>(2); break;
    case 2:
        if (!AVX2SearchBackend::is_supported()) GTEST_SKIP() << "AVX2 unavailable";
        backend = std::make_unique<AVX2SearchBackend>(2); break;
    case 3: backend = std::make_unique<MultithreadedScalarSearchBackend>(2, 2); break;
    case 4:
        if (!MultithreadedAVX2SearchBackend::is_supported()) GTEST_SKIP() << "AVX2 unavailable";
        backend = std::make_unique<MultithreadedAVX2SearchBackend>(2, 2); break;
    default:
        if (!CudaSearchBackend::is_supported()) GTEST_SKIP() << "CUDA unavailable";
        const auto kernel = GetParam() == 5 ? CudaKernel::Naive
                          : GetParam() == 6 ? CudaKernel::BlockParallel
                          : GetParam() == 7 ? CudaKernel::NaiveFP32
                                           : CudaKernel::BlockParallelFP32;
        backend = std::make_unique<CudaSearchBackend>(2,
            GetParam() == 9 ? CudaStorageMode::PerQuery : CudaStorageMode::Persistent,
            GetParam() == 9 ? CudaKernel::Naive : kernel);
    }
    VectorIndex index{2, std::move(backend)};
    index.add("opposite", {-1.0F, 0.0F});
    index.add("zero", {0.0F, 0.0F});
    index.add("b", {1.0F, 1.0F});
    index.add("a", {1.0F, 1.0F});
    const std::vector<float> query{1.0F, 0.0F};
    const VectorIndex& reader = index;
    const auto initial = reader.search(query, 3);
    ASSERT_EQ(initial.size(), 3U);
    EXPECT_EQ(initial[0].id, "a");
    EXPECT_EQ(initial[1].id, "b");
    EXPECT_EQ(initial[2].id, "zero");
    EXPECT_NEAR(initial[0].score, 0.70710678F, 1e-5F);
    EXPECT_FLOAT_EQ(initial[2].score, 0.0F);

    // Also exercises persistent CUDA rebuilds after a first search.
    index.add("exact", query);
    EXPECT_EQ(reader.size(), 5U);
    EXPECT_EQ(reader.dimension(), 2U);
    const auto results = reader.search(query, 10);
    ASSERT_EQ(results.size(), 5U);
    EXPECT_EQ(results.front().id, "exact");
    EXPECT_FLOAT_EQ(results.front().score, 1.0F);
    EXPECT_EQ(results.back().id, "opposite");
    EXPECT_FLOAT_EQ(results.back().score, -1.0F);
    for (std::size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i - 1].score, results[i].score);
    }
}

INSTANTIATE_TEST_SUITE_P(AllBackends, VectorIndexBackendTest, ::testing::Range(0, 10));

}  // namespace
}  // namespace vectorpulse

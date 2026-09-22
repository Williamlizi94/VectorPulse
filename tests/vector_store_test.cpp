#include "vectorpulse/vector_store.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace vectorpulse {
namespace {

TEST(VectorStoreTest, InsertsAndRetrievesVectors) {
    VectorStore store{3};
    store.add("doc1", {1.0F, 2.0F, 3.0F});

    EXPECT_EQ(store.size(), 1U);
    EXPECT_EQ(store.dimension(), 3U);
    EXPECT_EQ(store.get("doc1"), (std::vector<float>{1.0F, 2.0F, 3.0F}));
    EXPECT_THROW((void)store.get("missing"), std::out_of_range);
}

TEST(VectorStoreTest, RejectsInvalidDimensions) {
    EXPECT_THROW((void)VectorStore{0}, std::invalid_argument);

    VectorStore store{3};
    EXPECT_THROW(store.add("short", {1.0F, 2.0F}), std::invalid_argument);
    EXPECT_THROW((void)store.search(std::vector<float>{1.0F, 2.0F}, 1),
                 std::invalid_argument);
    EXPECT_EQ(store.size(), 0U);
}

TEST(VectorStoreTest, RejectsDuplicateIdsWithoutReplacingOriginal) {
    VectorStore store{2};
    store.add("same", {1.0F, 0.0F});

    EXPECT_THROW(store.add("same", {0.0F, 1.0F}), std::invalid_argument);
    EXPECT_EQ(store.size(), 1U);
    EXPECT_EQ(store.get("same"), (std::vector<float>{1.0F, 0.0F}));
}

TEST(VectorStoreTest, FindsTopOneResult) {
    VectorStore store{2};
    store.add("east", {1.0F, 0.0F});
    store.add("north", {0.0F, 1.0F});

    const auto results = store.search(std::vector<float>{0.9F, 0.1F}, 1);

    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(results[0].id, "east");
    EXPECT_GT(results[0].score, 0.99F);
}

TEST(VectorStoreTest, ReturnsTopKInDescendingSimilarityOrder) {
    VectorStore store{2};
    store.add("opposite", {-1.0F, 0.0F});
    store.add("orthogonal", {0.0F, 1.0F});
    store.add("diagonal", {1.0F, 1.0F});
    store.add("exact", {1.0F, 0.0F});

    const auto results = store.search(std::vector<float>{1.0F, 0.0F}, 3);

    ASSERT_EQ(results.size(), 3U);
    EXPECT_EQ(results[0].id, "exact");
    EXPECT_EQ(results[1].id, "diagonal");
    EXPECT_EQ(results[2].id, "orthogonal");
    EXPECT_GE(results[0].score, results[1].score);
    EXPECT_GE(results[1].score, results[2].score);
}

TEST(VectorStoreTest, ReturnsAllVectorsWhenKExceedsDatabaseSize) {
    VectorStore store{2};
    store.add("b", {0.0F, 1.0F});
    store.add("a", {1.0F, 0.0F});

    const auto results = store.search(std::vector<float>{1.0F, 0.0F}, 10);

    ASSERT_EQ(results.size(), 2U);
    EXPECT_EQ(results[0].id, "a");
    EXPECT_EQ(results[1].id, "b");
}

TEST(VectorStoreTest, HandlesEmptyDatabaseAndZeroK) {
    VectorStore store{2};
    EXPECT_TRUE(store.search(std::vector<float>{1.0F, 0.0F}, 5).empty());

    store.add("item", {1.0F, 0.0F});
    EXPECT_TRUE(store.search(std::vector<float>{1.0F, 0.0F}, 0).empty());
}

TEST(VectorStoreTest, SearchesZeroVectorsWithoutNan) {
    VectorStore store{2};
    store.add("zero", {0.0F, 0.0F});

    const auto results = store.search(std::vector<float>{1.0F, 0.0F}, 1);

    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(results[0].id, "zero");
    EXPECT_FLOAT_EQ(results[0].score, 0.0F);
}

}  // namespace
}  // namespace vectorpulse

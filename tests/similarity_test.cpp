#include "vectorpulse/similarity.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace vectorpulse {
namespace {

TEST(CosineSimilarityTest, ComputesKnownSimilarities) {
    EXPECT_NEAR(cosine_similarity(std::vector<float>{1.0F, 0.0F},
                                  std::vector<float>{1.0F, 0.0F}),
                1.0F,
                1.0e-6F);
    EXPECT_NEAR(cosine_similarity(std::vector<float>{1.0F, 0.0F},
                                  std::vector<float>{0.0F, 1.0F}),
                0.0F,
                1.0e-6F);
    EXPECT_NEAR(cosine_similarity(std::vector<float>{1.0F, 0.0F},
                                  std::vector<float>{-1.0F, 0.0F}),
                -1.0F,
                1.0e-6F);
    EXPECT_NEAR(cosine_similarity(std::vector<float>{1.0F, 2.0F, 3.0F},
                                  std::vector<float>{4.0F, 5.0F, 6.0F}),
                0.9746318F,
                1.0e-6F);
}

TEST(CosineSimilarityTest, DefinesZeroMagnitudeSimilarityAsZero) {
    EXPECT_FLOAT_EQ(cosine_similarity(std::vector<float>{0.0F, 0.0F},
                                      std::vector<float>{1.0F, 2.0F}),
                    0.0F);
    EXPECT_FLOAT_EQ(cosine_similarity(std::vector<float>{0.0F, 0.0F},
                                      std::vector<float>{0.0F, 0.0F}),
                    0.0F);
}

TEST(CosineSimilarityTest, RejectsMismatchedDimensions) {
    EXPECT_THROW(
        (void)cosine_similarity(std::vector<float>{1.0F},
                                std::vector<float>{1.0F, 2.0F}),
        std::invalid_argument);
}

TEST(CosineSimilarityTest, RejectsEmptyVectors) {
    EXPECT_THROW(
        (void)cosine_similarity(std::vector<float>{}, std::vector<float>{}),
        std::invalid_argument);
}

}  // namespace
}  // namespace vectorpulse

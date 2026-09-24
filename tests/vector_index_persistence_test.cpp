#include "vectorpulse/vector_index.h"
#include "vectorpulse/avx2_search_backend.h"
#include "vectorpulse/cuda_search_backend.h"
#include "vectorpulse/multithreaded_avx2_search_backend.h"
#include "vectorpulse/multithreaded_scalar_search_backend.h"
#include "vectorpulse/scalar_search_backend.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>

namespace vectorpulse {
namespace {

class PersistenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto root = std::filesystem::temp_directory_path();
        std::random_device random;
        for (int attempt = 0; attempt < 100; ++attempt) {
            const auto candidate = root / ("vectorpulse-persistence-" + std::to_string(random())
                                          + "-" + std::to_string(random()));
            if (std::filesystem::create_directory(candidate)) {
                directory = candidate;
                path = directory / "index.vp";
                return;
            }
        }
        FAIL() << "Unable to create unique test directory";
    }
    void TearDown() override {
        if (!directory.empty()) {
            std::error_code ignored;
            std::filesystem::remove_all(directory, ignored);
        }
    }
    std::vector<char> read() const {
        std::ifstream input{path, std::ios::binary};
        return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    }
    void write(const std::vector<char>& data) const {
        std::ofstream output{path, std::ios::binary | std::ios::trunc};
        output.write(data.data(), static_cast<std::streamsize>(data.size()));
        ASSERT_TRUE(output.good());
    }
    static void set_integer(std::vector<char>& data, std::size_t offset,
                            std::uint64_t value, std::size_t width = 8) {
        for (std::size_t i = 0; i < width; ++i) {
            data.at(offset + i) = static_cast<char>((value >> (8 * i)) & 0xffU);
        }
    }
    // Independent bit-at-a-time CRC implementation for valid malformed fixtures.
    static void fix_checksum(std::vector<char>& data) {
        std::uint32_t crc = 0xffffffffU;
        for (std::size_t i = 0; i < data.size() - 4; ++i) {
            crc ^= static_cast<unsigned char>(data[i]);
            for (int bit = 0; bit < 8; ++bit) {
                crc = (crc >> 1) ^ ((crc & 1U) ? 0xedb88320U : 0U);
            }
        }
        set_integer(data, data.size() - 4, crc ^ 0xffffffffU, 4);
    }
    void expect_same_results(const VectorIndex& first, const VectorIndex& second,
                             const std::vector<float>& query, std::size_t k) {
        const auto expected = first.search(query, k);
        const auto actual = second.search(query, k);
        ASSERT_EQ(expected.size(), actual.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i].id, actual[i].id);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(expected[i].score),
                      std::bit_cast<std::uint32_t>(actual[i].score));
        }
    }
    std::filesystem::path directory;
    std::filesystem::path path;
};

TEST_F(PersistenceTest, ScalarRoundTripPreservesResultsIdsAndRejectedAdds) {
    VectorIndex original{3};
    original.add("z", {1.0F, 1.0F, 0.0F});
    original.add("a", {1.0F, 1.0F, 0.0F});
    original.add("", {0.0F, -0.0F, 0.0F});
    original.add(std::string{"nul\0id", 6}, {-1.0F, 0.0F, 0.0F});
    original.add("\xe5\x90\x91\xe9\x87\x8f", {1.0F, 0.0F, 0.0F});
    EXPECT_THROW(original.add("a", {1.0F, 0.0F, 0.0F}), std::invalid_argument);
    original.save(path);
    auto loaded = VectorIndex::load(path);
    EXPECT_EQ(loaded.dimension(), 3U);
    EXPECT_EQ(loaded.size(), original.size());
    for (const auto& query : {std::vector<float>{1.0F, 0.0F, 0.0F},
                              std::vector<float>{-0.25F, 0.75F, 0.2F},
                              std::vector<float>{0.0F, 0.0F, 0.0F}}) {
        for (const auto k : {0U, 1U, 3U, 20U}) expect_same_results(original, loaded, query, k);
    }
    const auto before = read();
    loaded.save(path);
    EXPECT_EQ(read(), before);
    loaded.add("new", {0.0F, 1.0F, 0.0F});
    EXPECT_EQ(loaded.size(), original.size() + 1);
}

TEST_F(PersistenceTest, EmptyIndexRoundTrips) {
    VectorIndex original{7};
    original.save(path);
    auto loaded = VectorIndex::load(path);
    EXPECT_EQ(loaded.dimension(), 7U);
    EXPECT_EQ(loaded.size(), 0U);
    EXPECT_TRUE(loaded.search(std::vector<float>(7), 10).empty());
}

TEST_F(PersistenceTest, FormatIsLittleEndianAndPreservesFloatBits) {
    VectorIndex original{4};
    original.add("x", {1.0F, -0.0F, std::numeric_limits<float>::denorm_min(),
                       std::numeric_limits<float>::max()});
    original.save(path);
    const auto actual = read();
    std::vector<char> expected(32 + 8 + 1 + 16 + 4, 0);
    const std::array<char, 8> signature{'V', 'P', 'I', 'N', 'D', 'E', 'X', '\0'};
    std::copy(signature.begin(), signature.end(), expected.begin());
    set_integer(expected, 8, 1, 4);
    set_integer(expected, 16, 4);
    set_integer(expected, 24, 1);
    set_integer(expected, 32, 1);
    expected[40] = 'x';
    set_integer(expected, 41, 0x3f800000U, 4);
    set_integer(expected, 45, 0x80000000U, 4);
    set_integer(expected, 49, 0x00000001U, 4);
    set_integer(expected, 53, 0x7f7fffffU, 4);
    fix_checksum(expected);
    EXPECT_EQ(actual, expected);
    VectorIndex::load(path).save(path);
    EXPECT_EQ(read(), actual);
}

TEST_F(PersistenceTest, ChunkedVectorsAndLongIdsRoundTrip) {
    VectorIndex original{2051};
    std::vector<float> values(2051);
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i % 19) / 19.0F;
    original.add(std::string(70000, 'x'), values);
    original.save(path);
    auto loaded = VectorIndex::load(path);
    expect_same_results(original, loaded, values, 1);
    const auto before = read();
    loaded.save(path);
    EXPECT_EQ(read(), before);
}

TEST_F(PersistenceTest, IncludesEntriesAlreadyInInjectedBackend) {
    auto backend = std::make_unique<ScalarSearchBackend>(2);
    backend->add("preexisting", {1.0F, 0.0F});
    VectorIndex original{2, std::move(backend)};
    original.save(path);
    auto loaded = VectorIndex::load(path);
    expect_same_results(original, loaded, {1.0F, 0.0F}, 1);
}

TEST_F(PersistenceTest, RejectsMissingUnreadableAndUnwritablePaths) {
    EXPECT_THROW((void)VectorIndex::load(path), std::runtime_error);
    EXPECT_THROW((void)VectorIndex::load(directory), std::runtime_error);
    VectorIndex index{2};
    EXPECT_THROW(index.save(directory), std::runtime_error);
    EXPECT_THROW(index.save(directory / "missing" / "index.vp"), std::runtime_error);
    write({'n', 'o', 't', ' ', 'a', 'n', ' ', 'i', 'n', 'd', 'e', 'x'});
    EXPECT_THROW((void)VectorIndex::load(path), std::runtime_error);
}

TEST_F(PersistenceTest, RejectsEveryTruncatedPrefixAndTrailingData) {
    VectorIndex index{2};
    index.add("item", {1.0F, 0.0F});
    index.save(path);
    const auto valid = read();
    for (std::size_t size = 0; size < valid.size(); ++size) {
        SCOPED_TRACE(size);
        write(std::vector<char>(valid.begin(), valid.begin() + static_cast<std::ptrdiff_t>(size)));
        EXPECT_THROW((void)VectorIndex::load(path), std::runtime_error);
    }
    auto extra = valid;
    extra.push_back('x');
    write(extra);
    EXPECT_THROW((void)VectorIndex::load(path), std::runtime_error);
}

TEST_F(PersistenceTest, RejectsCorruptedHeadersEvenWithValidChecksum) {
    VectorIndex index{2};
    index.add("item", {1.0F, 0.0F});
    index.save(path);
    const auto valid = read();
    for (const auto offset : {0U, 8U, 12U}) {
        auto bytes = valid;
        bytes[offset] ^= 0x7f;
        fix_checksum(bytes);
        write(bytes);
        EXPECT_THROW((void)VectorIndex::load(path), std::runtime_error);
    }
    for (const auto offset : {16U, 24U}) {
        for (const auto value : {std::uint64_t{0}, std::numeric_limits<std::uint64_t>::max()}) {
            auto bytes = valid;
            set_integer(bytes, offset, value);
            fix_checksum(bytes);
            write(bytes);
            EXPECT_THROW((void)VectorIndex::load(path), std::runtime_error);
        }
    }
}

TEST_F(PersistenceTest, ChecksumDetectsPlausibleHeaderAndPayloadCorruption) {
    VectorIndex empty{2};
    empty.save(path);
    auto bytes = read();
    bytes[16] = 3; // Still a structurally valid empty index, but wrong checksum.
    write(bytes);
    EXPECT_THROW((void)VectorIndex::load(path), std::runtime_error);
    VectorIndex index{2};
    index.add("item", {1.0F, 0.0F});
    index.save(path);
    const auto valid = read();
    for (const auto offset : {std::size_t{40}, std::size_t{44}, valid.size() - 1}) {
        bytes = valid;
        bytes[offset] ^= 1;
        write(bytes);
        EXPECT_THROW((void)VectorIndex::load(path), std::runtime_error);
    }
}

TEST_F(PersistenceTest, RejectsOversizedIdsAndDuplicateRecords) {
    VectorIndex index{2};
    index.add("one", {1.0F, 0.0F});
    index.add("two", {0.0F, 1.0F});
    index.save(path);
    const auto valid = read();
    auto bytes = valid;
    set_integer(bytes, 32, std::numeric_limits<std::uint64_t>::max());
    fix_checksum(bytes);
    write(bytes);
    EXPECT_THROW((void)VectorIndex::load(path), std::runtime_error);
    bytes = valid;
    std::copy_n(bytes.begin() + 40, 3, bytes.begin() + 59);
    fix_checksum(bytes);
    write(bytes);
    EXPECT_THROW((void)VectorIndex::load(path), std::runtime_error);
}

TEST_F(PersistenceTest, RejectsIncompatibleOrNonemptyLoadBackend) {
    VectorIndex index{2};
    index.save(path);
    EXPECT_THROW((void)VectorIndex::load(path, std::make_unique<ScalarSearchBackend>(3)),
                 std::invalid_argument);
    auto backend = std::make_unique<ScalarSearchBackend>(2);
    backend->add("item", {1.0F, 0.0F});
    EXPECT_THROW((void)VectorIndex::load(path, std::move(backend)), std::invalid_argument);
}

class PersistenceBackendTest : public PersistenceTest, public ::testing::WithParamInterface<int> {
protected:
    std::unique_ptr<SearchBackend> make_backend() const {
        switch (GetParam()) {
        case 0: return std::make_unique<ScalarSearchBackend>(7);
        case 1: return std::make_unique<AVX2SearchBackend>(7);
        case 2: return std::make_unique<MultithreadedScalarSearchBackend>(7, 2);
        case 3: return std::make_unique<MultithreadedAVX2SearchBackend>(7, 2);
        default:
            const auto kernel = GetParam() == 4 ? CudaKernel::Naive
                              : GetParam() == 5 ? CudaKernel::BlockParallel
                              : GetParam() == 6 ? CudaKernel::NaiveFP32
                                               : CudaKernel::BlockParallelFP32;
            return std::make_unique<CudaSearchBackend>(7,
                GetParam() == 8 ? CudaStorageMode::PerQuery : CudaStorageMode::Persistent,
                GetParam() == 8 ? CudaKernel::Naive : kernel);
        }
    }
};

TEST_P(PersistenceBackendTest, ExactRoundTripAndBackendIndependentFiles) {
    if ((GetParam() == 1 || GetParam() == 3) && !AVX2SearchBackend::is_supported()) {
        GTEST_SKIP() << "AVX2 unavailable";
    }
    if (GetParam() >= 4 && !CudaSearchBackend::is_supported()) GTEST_SKIP() << "CUDA unavailable";
    VectorIndex original{7, make_backend()};
    std::mt19937 generator{42};
    std::uniform_real_distribution<float> distribution{-1.0F, 1.0F};
    for (int i = 0; i < 31; ++i) {
        std::vector<float> values(7);
        for (auto& value : values) value = distribution(generator);
        original.add(std::to_string(i), values);
    }
    original.add("zero", std::vector<float>(7));
    // Build the original CUDA cache before saving; the file must not retain it.
    (void)original.search(std::vector<float>(7, 1.0F), 3);
    original.save(path);
    const auto original_bytes = read();
    auto target_backend = make_backend();
    auto* cuda = dynamic_cast<CudaSearchBackend*>(target_backend.get());
    auto loaded = VectorIndex::load(path, std::move(target_backend));
    if (cuda && GetParam() != 8) {
        const auto first_query = cuda->search_profiled(std::vector<float>(7, 1.0F), 3);
        EXPECT_TRUE(first_query.timings.index_build.rebuilt);
        EXPECT_EQ(first_query.timings.database_h2d_bytes, 0U);
        EXPECT_GT(first_query.timings.index_build.database_bytes, 0U);
    }
    for (int i = 0; i < 5; ++i) {
        std::vector<float> query(7);
        for (auto& value : query) value = distribution(generator);
        for (const auto k : {0U, 1U, 10U, 100U}) expect_same_results(original, loaded, query, k);
    }
    loaded.save(path);
    EXPECT_EQ(read(), original_bytes);
    auto scalar = VectorIndex::load(path);
    scalar.save(path);
    EXPECT_EQ(read(), original_bytes); // No backend choice, GPU cache or timing state in file.
    auto scalar_reference = VectorIndex::load(path);
    auto cross_backend = VectorIndex::load(path, make_backend());
    const auto actual = cross_backend.search(std::vector<float>(7, 1.0F), 10);
    const auto expected = scalar_reference.search(std::vector<float>(7, 1.0F), 10);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_EQ(actual[i].id, expected[i].id);
        EXPECT_NEAR(actual[i].score, expected[i].score, 1e-5F);
    }
    loaded.add("after-load", std::vector<float>(7, 1.0F));
    EXPECT_EQ(loaded.search(std::vector<float>(7, 1.0F), 1)[0].id, "after-load");
}

INSTANTIATE_TEST_SUITE_P(AllBackends, PersistenceBackendTest, ::testing::Range(0, 9));

}  // namespace
}  // namespace vectorpulse

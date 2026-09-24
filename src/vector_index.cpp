#include "vectorpulse/vector_index.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace vectorpulse {
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "VectorPulse persistence requires IEEE-754 binary32 floats");
static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
constexpr std::array<char, 8> magic{'V', 'P', 'I', 'N', 'D', 'E', 'X', '\0'};
constexpr std::uint32_t format_version = 1;
constexpr std::uint64_t header_bytes = 32;
constexpr std::uint64_t checksum_bytes = 4;

[[noreturn]] void invalid_file(const char* reason) {
    throw std::runtime_error(std::string{"invalid VectorPulse file: "} + reason);
}

// CRC-32/ISO-HDLC, covering every byte except the final checksum itself.
constexpr auto crc_table = [] {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t i = 0; i < table.size(); ++i) {
        auto value = i;
        for (int bit = 0; bit < 8; ++bit) {
            value = (value >> 1) ^ ((value & 1U) ? 0xedb88320U : 0U);
        }
        table[i] = value;
    }
    return table;
}();

class Checksum {
public:
    void update(std::span<const char> bytes) {
        for (const char byte : bytes) {
            value_ = crc_table[(value_ ^ static_cast<unsigned char>(byte)) & 0xffU]
                     ^ (value_ >> 8);
        }
    }
    [[nodiscard]] std::uint32_t value() const { return value_ ^ 0xffffffffU; }
private:
    std::uint32_t value_ = 0xffffffffU;
};

template <typename UInt>
std::array<char, sizeof(UInt)> encode(UInt value) {
    std::array<char, sizeof(UInt)> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<char>((value >> (8 * i)) & 0xffU);
    }
    return bytes;
}

template <typename UInt>
UInt decode(std::span<const char> bytes) {
    UInt value = 0;
    for (std::size_t i = 0; i < sizeof(UInt); ++i) {
        value |= static_cast<UInt>(static_cast<unsigned char>(bytes[i])) << (8 * i);
    }
    return value;
}

class Writer {
public:
    explicit Writer(std::ofstream& output) : output_(output) {}
    void bytes(std::span<const char> data) {
        while (!data.empty()) {
            const auto chunk = data.first(std::min<std::size_t>(data.size(), 65536));
            output_.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
            if (!output_) throw std::runtime_error("cannot write VectorPulse file");
            checksum_.update(chunk);
            data = data.subspan(chunk.size());
        }
    }
    template <typename UInt> void integer(UInt value) { bytes(encode(value)); }
    [[nodiscard]] std::uint32_t checksum() const { return checksum_.value(); }
private:
    std::ofstream& output_;
    Checksum checksum_;
};

class Reader {
public:
    Reader(std::ifstream& input, std::uint64_t payload_size)
        : input_(input), remaining_(payload_size) {}
    void bytes(std::span<char> data) {
        if (data.size() > remaining_) invalid_file("truncated data");
        remaining_ -= data.size();
        while (!data.empty()) {
            const auto chunk = data.first(std::min<std::size_t>(data.size(), 65536));
            input_.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
            if (!input_) invalid_file("read failure or truncated data");
            checksum_.update(chunk);
            data = data.subspan(chunk.size());
        }
    }
    template <typename UInt> UInt integer() {
        std::array<char, sizeof(UInt)> data{};
        bytes(data);
        return decode<UInt>(data);
    }
    [[nodiscard]] std::uint64_t remaining() const { return remaining_; }
    [[nodiscard]] std::uint32_t checksum() const { return checksum_.value(); }
private:
    std::ifstream& input_;
    std::uint64_t remaining_;
    Checksum checksum_;
};

void write_vector(Writer& writer, std::span<const float> values) {
    std::array<char, 4096> buffer{};
    while (!values.empty()) {
        const auto count = std::min(values.size(), buffer.size() / sizeof(float));
        for (std::size_t i = 0; i < count; ++i) {
            const auto bits = encode(std::bit_cast<std::uint32_t>(values[i]));
            std::copy(bits.begin(), bits.end(), buffer.data() + i * sizeof(float));
        }
        writer.bytes(std::span{buffer}.first(count * sizeof(float)));
        values = values.subspan(count);
    }
}

void read_vector(Reader& reader, std::span<float> values) {
    std::array<char, 4096> buffer{};
    while (!values.empty()) {
        const auto count = std::min(values.size(), buffer.size() / sizeof(float));
        reader.bytes(std::span{buffer}.first(count * sizeof(float)));
        for (std::size_t i = 0; i < count; ++i) {
            values[i] = std::bit_cast<float>(decode<std::uint32_t>(
                std::span{buffer}.subspan(i * sizeof(float), sizeof(float))));
        }
        values = values.subspan(count);
    }
}

}  // namespace

void VectorIndex::save(const std::filesystem::path& path) const {
    std::ofstream output{path, std::ios::binary | std::ios::trunc};
    if (!output) throw std::runtime_error("cannot open VectorPulse file for writing");
    Writer writer{output};
    writer.bytes(magic);
    writer.integer(format_version);
    writer.integer(std::uint32_t{0}); // Reserved flags; must be zero in version 1.
    writer.integer(static_cast<std::uint64_t>(dimension()));
    writer.integer(static_cast<std::uint64_t>(size()));
    std::size_t visited = 0;
    store_.for_each_vector([&](std::string_view id, std::span<const float> values) {
        if (visited == size() || values.size() != dimension()) {
            throw std::runtime_error("inconsistent backend entries during save");
        }
        writer.integer(static_cast<std::uint64_t>(id.size()));
        writer.bytes({id.data(), id.size()});
        write_vector(writer, values);
        ++visited;
    });
    if (visited != size()) throw std::runtime_error("incomplete backend enumeration during save");
    writer.integer(writer.checksum());
    output.close();
    if (!output) throw std::runtime_error("cannot finish writing VectorPulse file");
}

VectorIndex VectorIndex::load(const std::filesystem::path& path,
                              std::unique_ptr<SearchBackend> backend) {
    std::ifstream input{path, std::ios::binary | std::ios::ate};
    if (!input) throw std::runtime_error("cannot open VectorPulse file for reading");
    const auto length = input.tellg();
    if (length < 0 || static_cast<std::uint64_t>(length) < header_bytes + checksum_bytes) {
        invalid_file("missing header or checksum");
    }
    input.seekg(0);
    Reader reader{input, static_cast<std::uint64_t>(length) - checksum_bytes};
    std::array<char, magic.size()> file_magic{};
    reader.bytes(file_magic);
    if (file_magic != magic) invalid_file("bad magic");
    if (reader.integer<std::uint32_t>() != format_version) invalid_file("unsupported version");
    if (reader.integer<std::uint32_t>() != 0) invalid_file("unknown flags");
    const auto dimension = reader.integer<std::uint64_t>();
    const auto count = reader.integer<std::uint64_t>();
    if (dimension == 0 || dimension > std::vector<float>{}.max_size() ||
        dimension > (std::numeric_limits<std::uint64_t>::max() - 8) / sizeof(float)) {
        invalid_file("invalid dimension");
    }
    const auto vector_bytes = dimension * sizeof(float);
    const auto minimum_record_bytes = 8 + vector_bytes;
    if (count > std::numeric_limits<std::size_t>::max() ||
        count > reader.remaining() / minimum_record_bytes) {
        invalid_file("invalid vector count or truncated records");
    }
    if (backend && (backend->size() != 0 || backend->dimension() != dimension)) {
        throw std::invalid_argument("load backend must be empty and match the file dimension");
    }
    VectorIndex index{static_cast<std::size_t>(dimension), std::move(backend)};
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto id_length = reader.integer<std::uint64_t>();
        // Leave enough bytes for this vector and every remaining record before
        // allocating. The multiplication is bounded by the count check above.
        const auto required = vector_bytes + (count - i - 1) * minimum_record_bytes;
        if (reader.remaining() < required || id_length > reader.remaining() - required ||
            id_length > std::string{}.max_size()) {
            invalid_file("invalid ID length or truncated record");
        }
        std::string id(static_cast<std::size_t>(id_length), '\0');
        reader.bytes({id.data(), id.size()});
        std::vector<float> values(static_cast<std::size_t>(dimension));
        read_vector(reader, values);
        try {
            index.add(std::move(id), std::move(values));
        } catch (const std::invalid_argument& error) {
            throw std::runtime_error(std::string{"invalid VectorPulse record: "} + error.what());
        }
    }
    if (reader.remaining() != 0) invalid_file("unexpected trailing data");
    std::array<char, sizeof(std::uint32_t)> footer{};
    input.read(footer.data(), static_cast<std::streamsize>(footer.size()));
    if (!input || decode<std::uint32_t>(footer) != reader.checksum()) {
        invalid_file("checksum mismatch");
    }
    if (input.peek() != std::char_traits<char>::eof()) invalid_file("unexpected trailing data");
    return index;
}

}  // namespace vectorpulse

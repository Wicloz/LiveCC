#include "test_support.hpp"

#include <atomic>
#include <fstream>
#include <string>

#include "engine/chunk.hpp"

namespace ccmfplayer::testing_support {

std::vector<std::byte> MakeBytes(std::initializer_list<unsigned char> values) {
    std::vector<std::byte> out;
    out.reserve(values.size());
    for (unsigned char v : values) {
        out.push_back(static_cast<std::byte>(v));
    }
    return out;
}

std::vector<std::byte> BuildChunk(std::uint64_t pts, std::uint8_t type,
                                   std::span<const std::byte> payload,
                                   std::uint8_t compression) {
    std::vector<std::byte> out;
    out.reserve(kChunkHeaderSize + payload.size());

    out.push_back(static_cast<std::byte>(kChunkMarker));
    for (int i = 0; i < 6; ++i) {
        out.push_back(static_cast<std::byte>((pts >> (8 * i)) & 0xFF));
    }
    const auto length = static_cast<std::uint32_t>(payload.size());
    for (int i = 0; i < 3; ++i) {
        out.push_back(static_cast<std::byte>((length >> (8 * i)) & 0xFF));
    }
    out.push_back(static_cast<std::byte>(type));
    out.push_back(static_cast<std::byte>(compression));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

namespace {
std::atomic<unsigned> gTempFileCounter{0};
}  // namespace

TempFile::TempFile(std::span<const std::byte> bytes) {
    const unsigned id = gTempFileCounter.fetch_add(1, std::memory_order_relaxed);
    path_ = std::filesystem::temp_directory_path()
            / ("player_test_" + std::to_string(id) + ".ccmf");
    std::ofstream out(path_, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

TempFile::~TempFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
}

}  // namespace ccmfplayer::testing_support

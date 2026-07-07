#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <span>
#include <vector>

namespace ccmfplayer::testing_support {

// Builds a byte vector from a brace list of small (0-255) integer literals,
// so tests can spell out wire-format bytes (spec hex values) without a wall
// of static_cast<std::byte>(...).
[[nodiscard]] std::vector<std::byte> MakeBytes(std::initializer_list<unsigned char> values);

// Builds one full chunk (header + payload), mirroring server/ccmf.py's
// chunk(): marker, 6-byte LE pts, 3-byte LE length, type, compression, then
// the payload bytes verbatim. For hand-built synthetic test fixtures.
[[nodiscard]] std::vector<std::byte> BuildChunk(std::uint64_t pts, std::uint8_t type,
                                                 std::span<const std::byte> payload,
                                                 std::uint8_t compression = 0);

// RAII temp file: writes `bytes` to a fresh path under the OS temp directory
// at construction, deletes it at destruction. Needed because CcmfFile only
// opens paths, not in-memory buffers.
class TempFile {
public:
    explicit TempFile(std::span<const std::byte> bytes);
    ~TempFile();

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    [[nodiscard]] const std::filesystem::path& Path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace ccmfplayer::testing_support

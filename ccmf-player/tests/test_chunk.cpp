#include "engine/chunk.hpp"

#include <gtest/gtest.h>

#include "test_support.hpp"

namespace ccmfplayer {
namespace {

using testing_support::BuildChunk;
using testing_support::MakeBytes;

TEST(ParseChunkHeader, ValidHeaderRoundTrips) {
    // marker=0x43, pts=0x123456 (LE 6 bytes), length=10 (LE 3 bytes), type=0 (video),
    // compression=0.
    const auto bytes = MakeBytes({0x43, 0x56, 0x34, 0x12, 0x00, 0x00, 0x00,
                                   10, 0, 0, kChunkTypeVideo, 0});
    const ChunkHeader header = ParseChunkHeader(bytes);

    EXPECT_EQ(header.pts, 0x123456u);
    EXPECT_EQ(header.length, 10u);
    EXPECT_EQ(header.type, kChunkTypeVideo);
    EXPECT_EQ(header.compression, 0u);
}

TEST(ParseChunkHeader, MaxWidthFieldsRoundTrip) {
    // pts at the 48-bit max, length at the 24-bit max.
    const std::uint64_t maxPts = (std::uint64_t{1} << 48) - 1;
    const std::uint32_t maxLength = (std::uint32_t{1} << 24) - 1;
    auto bytes = BuildChunk(maxPts, kChunkTypeAudio, std::span<const std::byte>{});
    // BuildChunk always encodes payload.size() as length, so overwrite the
    // length field directly to exercise the true 24-bit max without
    // allocating a 16 MiB payload.
    bytes[7] = static_cast<std::byte>(maxLength & 0xFF);
    bytes[8] = static_cast<std::byte>((maxLength >> 8) & 0xFF);
    bytes[9] = static_cast<std::byte>((maxLength >> 16) & 0xFF);

    const ChunkHeader header = ParseChunkHeader(bytes);
    EXPECT_EQ(header.pts, maxPts);
    EXPECT_EQ(header.length, maxLength);
}

TEST(ParseChunkHeader, TruncatedHeaderThrows) {
    const auto bytes = MakeBytes({0x43, 1, 2, 3, 4, 5});  // 6 bytes, needs 12
    EXPECT_THROW((void)ParseChunkHeader(bytes), CcmfError);
}

TEST(ParseChunkHeader, EmptyBufferThrows) {
    EXPECT_THROW((void)ParseChunkHeader(std::span<const std::byte>{}), CcmfError);
}

TEST(ParseChunkHeader, BadMarkerThrows) {
    auto bytes = BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3}));
    bytes[0] = static_cast<std::byte>(0xFF);
    EXPECT_THROW((void)ParseChunkHeader(bytes), CcmfError);
}

TEST(ParseChunkHeader, UnsupportedCompressionThrows) {
    auto bytes = BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3}), /*compression=*/1);
    EXPECT_THROW((void)ParseChunkHeader(bytes), CcmfError);
}

}  // namespace
}  // namespace ccmfplayer

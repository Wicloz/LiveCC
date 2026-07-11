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

TEST(ParseChunkHeader, RecordsCompressionWithoutRejecting) {
    // Compression is recorded, not validated, so indexing can still walk a file
    // using an algorithm this build can't decode (spec 4.1). Rejection happens
    // later, in DecompressPayload.
    auto bytes = BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3}), /*compression=*/1);
    EXPECT_EQ(ParseChunkHeader(bytes).compression, 1u);
}

TEST(DecompressPayload, NoneIsPassthrough) {
    const auto payload = MakeBytes({1, 2, 3, 4, 5});
    EXPECT_EQ(DecompressPayload(payload, kCompressionNone), payload);
}

TEST(DecompressPayload, Lz4RoundTripsServerVector) {
    // Wire bytes = server lz4.block.compress + our u32 size prefix (spec 4.1.2);
    // must inflate to the original. Exercises extended literal/match lengths and
    // an overlapping back-reference.
    const auto wire = MakeBytes({62, 0, 0, 0, 255, 6, 84, 104, 101, 32, 113, 117,
                                 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102,
                                 111, 120, 46, 32, 21, 0, 17, 80, 32, 102, 111, 120,
                                 46});
    const auto expected = MakeBytes({84, 104, 101, 32, 113, 117, 105, 99, 107, 32,
                                     98, 114, 111, 119, 110, 32, 102, 111, 120, 46,
                                     32, 84, 104, 101, 32, 113, 117, 105, 99, 107,
                                     32, 98, 114, 111, 119, 110, 32, 102, 111, 120,
                                     46, 32, 84, 104, 101, 32, 113, 117, 105, 99,
                                     107, 32, 98, 114, 111, 119, 110, 32, 102, 111,
                                     120, 46});
    EXPECT_EQ(DecompressPayload(wire, kCompressionLz4), expected);
}

namespace {
// "The quick brown fox jumps over the lazy dog. " x6 (270 bytes).
const std::vector<std::byte> kFoxPayload = MakeBytes({
    84,104,101,32,113,117,105,99,107,32,98,114,111,119,110,32,102,111,120,32,106,
    117,109,112,115,32,111,118,101,114,32,116,104,101,32,108,97,122,121,32,100,111,
    103,46,32,84,104,101,32,113,117,105,99,107,32,98,114,111,119,110,32,102,111,120,
    32,106,117,109,112,115,32,111,118,101,114,32,116,104,101,32,108,97,122,121,32,
    100,111,103,46,32,84,104,101,32,113,117,105,99,107,32,98,114,111,119,110,32,102,
    111,120,32,106,117,109,112,115,32,111,118,101,114,32,116,104,101,32,108,97,122,
    121,32,100,111,103,46,32,84,104,101,32,113,117,105,99,107,32,98,114,111,119,110,
    32,102,111,120,32,106,117,109,112,115,32,111,118,101,114,32,116,104,101,32,108,
    97,122,121,32,100,111,103,46,32,84,104,101,32,113,117,105,99,107,32,98,114,111,
    119,110,32,102,111,120,32,106,117,109,112,115,32,111,118,101,114,32,116,104,101,
    32,108,97,122,121,32,100,111,103,46,32,84,104,101,32,113,117,105,99,107,32,98,
    114,111,119,110,32,102,111,120,32,106,117,109,112,115,32,111,118,101,114,32,116,
    104,101,32,108,97,122,121,32,100,111,103,46,32});
}  // namespace

TEST(DecompressPayload, BrotliRoundTripsServerVector) {
    // server ccmf.compress_payload(payload, COMPRESSION_BROTLI): [u32 size][stream].
    const auto wire = MakeBytes({
        14,1,0,0,27,13,1,136,44,14,120,211,208,149,93,151,16,187,23,43,169,202,208,
        146,204,140,173,65,92,230,242,54,200,25,158,158,10,123,131,13,56,112,72,32,
        111,36,188,65,167,21,206,28,30,39,170,41,56,194,217,236,211,96});
    EXPECT_EQ(DecompressPayload(wire, kCompressionBrotli), kFoxPayload);
}

TEST(DecompressPayload, Bzip2RoundTripsServerVector) {
    const auto wire = MakeBytes({
        14,1,0,0,66,90,104,57,49,65,89,38,83,89,87,64,232,21,0,0,32,147,128,64,1,4,
        0,63,255,255,240,32,0,112,80,0,52,0,0,34,148,212,211,70,140,38,134,212,219,
        82,64,228,50,62,134,71,192,212,96,106,58,15,113,246,48,59,12,14,192,216,110,
        29,199,129,212,108,61,7,224,192,220,50,59,141,7,33,168,240,52,29,71,240,216,
        104,50,50,58,8,114,60,143,241,119,36,83,133,9,5,116,14,129,80});
    EXPECT_EQ(DecompressPayload(wire, kCompressionBzip2), kFoxPayload);
}

TEST(DecompressPayload, UnsupportedAlgorithmThrows) {
    EXPECT_THROW((void)DecompressPayload(MakeBytes({1, 2, 3}), /*deflate=*/1), CcmfError);
}

}  // namespace
}  // namespace ccmfplayer

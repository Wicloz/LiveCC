#include "engine/resync.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "engine/chunk.hpp"
#include "test_support.hpp"

namespace ccmfplayer {
namespace {

using testing_support::BuildChunk;
using testing_support::MakeBytes;
using testing_support::TempFile;

std::vector<std::byte> Concat(std::initializer_list<std::vector<std::byte>> chunks) {
    std::vector<std::byte> out;
    for (const auto& chunk : chunks) {
        out.insert(out.end(), chunk.begin(), chunk.end());
    }
    return out;
}

// A file opened for the resync primitives, with its size precomputed.
struct Opened {
    explicit Opened(const std::filesystem::path& path) : in(path, std::ios::binary) {
        in.seekg(0, std::ios::end);
        size = static_cast<std::uint64_t>(in.tellg());
        in.seekg(0, std::ios::beg);
    }
    std::ifstream in;
    std::uint64_t size = 0;
};

// --------------------------------------------------------------------------
// TryReadHeaderAt.
// --------------------------------------------------------------------------

TEST(TryReadHeaderAt, ReadsAValidBoundary) {
    const TempFile file(BuildChunk(4242, kChunkTypeVideo, MakeBytes({1, 2, 3})));
    Opened f(file.Path());

    ChunkEntry out;
    ASSERT_TRUE(TryReadHeaderAt(f.in, 0, f.size, out));
    EXPECT_EQ(out.offset, 0u);
    EXPECT_EQ(out.pts, 4242u);
    EXPECT_EQ(out.length, 3u);
    EXPECT_EQ(out.type, kChunkTypeVideo);
    EXPECT_EQ(out.End(), f.size);
}

TEST(TryReadHeaderAt, RejectsBadMarker) {
    auto bytes = BuildChunk(0, kChunkTypeVideo, MakeBytes({1}));
    bytes[0] = std::byte{0xFF};
    const TempFile file(bytes);
    Opened f(file.Path());

    ChunkEntry out;
    EXPECT_FALSE(TryReadHeaderAt(f.in, 0, f.size, out));
}

TEST(TryReadHeaderAt, RejectsPayloadThatOverrunsTheFile) {
    auto bytes = BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3, 4}));
    bytes.resize(bytes.size() - 2);  // header still says length=4, file is short
    const TempFile file(bytes);
    Opened f(file.Path());

    ChunkEntry out;
    EXPECT_FALSE(TryReadHeaderAt(f.in, 0, f.size, out));
}

TEST(TryReadHeaderAt, RejectsTruncatedHeader) {
    const TempFile file(MakeBytes({0x43, 1, 2, 3}));  // 4 bytes, header needs 12
    Opened f(file.Path());

    ChunkEntry out;
    EXPECT_FALSE(TryReadHeaderAt(f.in, 0, f.size, out));
}

// --------------------------------------------------------------------------
// ChainFrom.
// --------------------------------------------------------------------------

TEST(ChainFrom, ChainsEveryChunkToEof) {
    const TempFile file(Concat({
        BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3})),
        BuildChunk(1000, kChunkTypeAudio, MakeBytes({4})),
        BuildChunk(2000, kChunkTypeVideo, MakeBytes({5, 6})),
    }));
    Opened f(file.Path());

    std::vector<ChunkEntry> out;
    const ChainResult r = ChainFrom(f.in, 0, f.size, 100, out);
    EXPECT_TRUE(r.reachedEof);
    EXPECT_EQ(r.links, 3u);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out.back().pts, 2000u);
}

TEST(ChainFrom, StopsAtDesync) {
    auto bytes = Concat({
        BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3})),
        BuildChunk(1000, kChunkTypeVideo, MakeBytes({4})),
    });
    bytes.insert(bytes.end(), 20, std::byte{0x00});  // junk trailer, not a chunk
    const TempFile file(bytes);
    Opened f(file.Path());

    std::vector<ChunkEntry> out;
    const ChainResult r = ChainFrom(f.in, 0, f.size, 100, out);
    EXPECT_FALSE(r.reachedEof);
    EXPECT_EQ(r.links, 2u);  // both real chunks, then it hits the junk and stops
}

// --------------------------------------------------------------------------
// Resync: land anywhere, find the next real boundary, ignore false markers.
// --------------------------------------------------------------------------

TEST(Resync, SkipsFalseMarkerInPayloadAndFindsRealBoundary) {
    // chunk0's payload contains a stray marker byte (0x43) that must NOT be
    // mistaken for a boundary -- it doesn't chain.
    const auto payload0 = MakeBytes({0x00, kChunkMarker, 0x00, 0x00});
    const auto chunk0 = BuildChunk(0, kChunkTypeVideo, payload0);
    const std::uint64_t chunk1Offset = chunk0.size();
    const TempFile file(Concat({
        chunk0,
        BuildChunk(1000, kChunkTypeVideo, MakeBytes({1})),
        BuildChunk(2000, kChunkTypeVideo, MakeBytes({2})),
        BuildChunk(3000, kChunkTypeVideo, MakeBytes({3})),
    }));
    Opened f(file.Path());

    std::vector<ChunkEntry> collected;
    const auto m = Resync(f.in, /*from=*/1, /*limit=*/f.size, f.size, /*requiredLinks=*/2, collected);
    ASSERT_TRUE(m.has_value());
    EXPECT_EQ(*m, chunk1Offset);
    ASSERT_FALSE(collected.empty());
    EXPECT_EQ(collected.front().pts, 1000u);
}

// --------------------------------------------------------------------------
// FindTailChunks: read backwards to the duration anchor.
// --------------------------------------------------------------------------

TEST(FindTailChunks, RecoversTrailingChainReachingEof) {
    const auto bytes = Concat({
        BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3})),
        BuildChunk(1000, kChunkTypeAudio, MakeBytes({4})),
        BuildChunk(2000, kChunkTypeVideo, MakeBytes({5, 6})),
        BuildChunk(3000, kChunkTypeAudio, MakeBytes({7, 8, 9})),
    });
    const TempFile file(bytes);
    Opened f(file.Path());

    const auto tail = FindTailChunks(f.in, f.size);
    ASSERT_TRUE(tail.has_value());
    ASSERT_FALSE(tail->empty());
    EXPECT_EQ(tail->back().pts, 3000u);
    EXPECT_EQ(tail->back().End(), f.size);
}

TEST(FindTailChunks, HandlesASingleChunkFile) {
    const TempFile file(BuildChunk(500, kChunkTypeVideo, MakeBytes({1, 2})));
    Opened f(file.Path());

    const auto tail = FindTailChunks(f.in, f.size);
    ASSERT_TRUE(tail.has_value());
    ASSERT_EQ(tail->size(), 1u);
    EXPECT_EQ(tail->back().pts, 500u);
    EXPECT_EQ(tail->back().End(), f.size);
}

TEST(FindTailChunks, EmptyFileHasNoTail) {
    const TempFile file(std::vector<std::byte>{});
    Opened f(file.Path());
    EXPECT_FALSE(FindTailChunks(f.in, f.size).has_value());
}

}  // namespace
}  // namespace ccmfplayer

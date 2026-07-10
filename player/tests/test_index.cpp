#include "engine/index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
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

bool HasPts(const std::vector<ChunkEntry>& run, std::uint64_t pts) {
    return std::any_of(run.begin(), run.end(),
                       [pts](const ChunkEntry& e) { return e.pts == pts; });
}

// --------------------------------------------------------------------------
// Lazy discovery: nothing is read up front; chunks enter the index only when
// something asks for them.
// --------------------------------------------------------------------------

TEST(CcmfFile, ConstructionIsLazyAndIndexesNothing) {
    const TempFile file(Concat({
        BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3})),
        BuildChunk(1000, kChunkTypeAudio, MakeBytes({4, 5})),
        BuildChunk(2000, kChunkTypeVideo, MakeBytes({6})),
    }));

    CcmfFile ccmf(file.Path());
    EXPECT_GT(ccmf.FileSize(), 0u);
    EXPECT_TRUE(ccmf.KnownChunks().empty());  // opened, but read nothing yet
}

TEST(CcmfFile, FirstChunkAndNextChunkWalkForwardAndRecord) {
    const TempFile file(Concat({
        BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3})),
        BuildChunk(1000, kChunkTypeVideo, MakeBytes({4})),
        BuildChunk(2000, kChunkTypeVideo, MakeBytes({5, 6})),
    }));
    CcmfFile ccmf(file.Path());

    const auto first = ccmf.FirstChunk();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->offset, 0u);
    EXPECT_EQ(first->pts, 0u);

    const auto second = ccmf.NextChunkAfter(*first);
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->pts, 1000u);
    EXPECT_EQ(second->offset, kChunkHeaderSize + 3);

    const auto third = ccmf.NextChunkAfter(*second);
    ASSERT_TRUE(third.has_value());
    EXPECT_EQ(third->pts, 2000u);

    const auto past = ccmf.NextChunkAfter(*third);
    EXPECT_FALSE(past.has_value());  // last chunk ends exactly at EOF

    EXPECT_EQ(ccmf.KnownChunks().size(), 3u);  // exactly what was walked
}

// --------------------------------------------------------------------------
// Backread: the duration anchor is recovered by reading from the tail.
// --------------------------------------------------------------------------

TEST(CcmfFile, TailChunksRecoversTrailingChain) {
    const TempFile file(Concat({
        BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3})),
        BuildChunk(1000, kChunkTypeAudio, MakeBytes({4, 5})),
        BuildChunk(2000, kChunkTypeVideo, MakeBytes({6})),
        BuildChunk(3000, kChunkTypeAudio, MakeBytes({7, 8, 9})),
    }));
    CcmfFile ccmf(file.Path());

    const std::vector<ChunkEntry>& tail = ccmf.TailChunks();
    ASSERT_FALSE(tail.empty());
    EXPECT_EQ(tail.back().pts, 3000u);
    EXPECT_EQ(tail.back().End(), ccmf.FileSize());  // the chain reaches EOF
    // Everything the backread touched is recorded.
    EXPECT_FALSE(ccmf.KnownChunks().empty());
}

// --------------------------------------------------------------------------
// Region homing: RegionAround brackets a target pts and records what it finds.
// --------------------------------------------------------------------------

TEST(CcmfFile, RegionAroundBracketsTheTargetPts) {
    const TempFile file(Concat({
        BuildChunk(0, kChunkTypeVideo, MakeBytes({1})),
        BuildChunk(1000, kChunkTypeVideo, MakeBytes({2})),
        BuildChunk(2000, kChunkTypeVideo, MakeBytes({3})),
        BuildChunk(3000, kChunkTypeVideo, MakeBytes({4})),
    }));
    CcmfFile ccmf(file.Path());

    const std::vector<ChunkEntry> run = ccmf.RegionAround(1500);
    ASSERT_FALSE(run.empty());
    EXPECT_TRUE(HasPts(run, 1000u));  // the chunk active at 1500
    EXPECT_TRUE(HasPts(run, 2000u));  // and the one bracketing it above
    // The run is contiguous and ascending in offset.
    for (std::size_t i = 1; i < run.size(); ++i) {
        EXPECT_EQ(run[i].offset, run[i - 1].End());
    }
    EXPECT_FALSE(ccmf.KnownChunks().empty());
}

TEST(CcmfFile, RegionAroundHomesWithoutWalkingFromTheStart) {
    // A file spanning well over the region guard (5 s), so seeking deep into it
    // exercises the interpolation-search homing rather than a walk from offset 0.
    std::vector<std::byte> bytes;
    for (int i = 0; i < 40; ++i) {
        const auto chunk = BuildChunk(static_cast<std::uint64_t>(i) * 48000, kChunkTypeVideo,
                                      MakeBytes({static_cast<unsigned char>(i)}));
        bytes.insert(bytes.end(), chunk.begin(), chunk.end());
    }
    const TempFile file(bytes);
    CcmfFile ccmf(file.Path());

    const std::vector<ChunkEntry> run = ccmf.RegionAround(500000);  // ~10.4 s in
    ASSERT_FALSE(run.empty());
    EXPECT_TRUE(HasPts(run, 480000u));  // active GOP (greatest pts <= 500000)
    EXPECT_TRUE(HasPts(run, 528000u));  // and the bracketing GOP above it
    // Homing jumped ahead instead of walking from the head.
    EXPECT_FALSE(HasPts(run, 0u));
    EXPECT_GT(run.front().pts, 0u);
    EXPECT_LE(run.front().pts, 500000u - 5u * 48000u);  // started before the guard window
}

// --------------------------------------------------------------------------
// Payload reads (unchanged behaviour, now off lazily-discovered entries).
// --------------------------------------------------------------------------

TEST(CcmfFile, ReadChunkPayloadReturnsExactBytes) {
    const auto payload = MakeBytes({0xDE, 0xAD, 0xBE, 0xEF});
    const TempFile file(BuildChunk(1234, kChunkTypeVideo, payload));
    CcmfFile ccmf(file.Path());

    const auto first = ccmf.FirstChunk();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(ccmf.ReadChunkPayload(*first), payload);
}

TEST(CcmfFile, ReadChunkPayloadOfSecondChunkSkipsTheFirst) {
    const auto firstPayload = MakeBytes({1, 1, 1});
    const auto secondPayload = MakeBytes({2, 2});
    const TempFile file(Concat({
        BuildChunk(0, kChunkTypeVideo, firstPayload),
        BuildChunk(500, kChunkTypeVideo, secondPayload),
    }));
    CcmfFile ccmf(file.Path());

    const auto first = ccmf.FirstChunk();
    ASSERT_TRUE(first.has_value());
    const auto second = ccmf.NextChunkAfter(*first);
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(ccmf.ReadChunkPayload(*second), secondPayload);
}

TEST(CcmfFile, ReadChunkPayloadRangeReadsASlice) {
    const auto payload = MakeBytes({0xAA, 0xBB, 0xCC, 0xDD, 0xEE});
    const TempFile file(BuildChunk(0, kChunkTypeAudio, payload));
    CcmfFile ccmf(file.Path());

    const auto entry = ccmf.FirstChunk();
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(ccmf.ReadChunkPayloadRange(*entry, 0, 1), MakeBytes({0xAA}));
    EXPECT_EQ(ccmf.ReadChunkPayloadRange(*entry, 2, 2), MakeBytes({0xCC, 0xDD}));
    EXPECT_EQ(ccmf.ReadChunkPayloadRange(*entry, 0, 5), payload);
}

TEST(CcmfFile, ReadChunkPayloadRangeOutOfBoundsThrows) {
    const auto payload = MakeBytes({1, 2, 3});
    const TempFile file(BuildChunk(0, kChunkTypeAudio, payload));
    CcmfFile ccmf(file.Path());

    const auto entry = ccmf.FirstChunk();
    ASSERT_TRUE(entry.has_value());
    EXPECT_THROW((void)ccmf.ReadChunkPayloadRange(*entry, 2, 5), CcmfError);
}

// --------------------------------------------------------------------------
// IndexAll: the eager reference/verification path (tests + escape hatch).
// --------------------------------------------------------------------------

TEST(CcmfFile, IndexAllPartitionsByType) {
    const TempFile file(Concat({
        BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3})),
        BuildChunk(0, kChunkTypeAudio, MakeBytes({4, 5})),
        BuildChunk(2000, kChunkTypeVideo, MakeBytes({6})),
        BuildChunk(2000, kChunkTypeAudio, MakeBytes({})),
    }));

    CcmfFile ccmf(file.Path());
    const CcmfFile::FullIndex idx = ccmf.IndexAll();

    ASSERT_EQ(idx.chunks.size(), 4u);
    ASSERT_EQ(idx.video.size(), 2u);
    ASSERT_EQ(idx.audio.size(), 2u);
    EXPECT_EQ(idx.video[0].pts, 0u);
    EXPECT_EQ(idx.video[1].pts, 2000u);
    EXPECT_EQ(idx.audio[0].length, 2u);
    EXPECT_EQ(idx.audio[1].length, 0u);
}

TEST(CcmfFile, IndexAllChunkOffsetsAdvanceByHeaderPlusLength) {
    const TempFile file(Concat({
        BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3})),
        BuildChunk(100, kChunkTypeVideo, MakeBytes({4})),
    }));

    CcmfFile ccmf(file.Path());
    const CcmfFile::FullIndex idx = ccmf.IndexAll();
    ASSERT_EQ(idx.chunks.size(), 2u);
    EXPECT_EQ(idx.chunks[0].offset, 0u);
    EXPECT_EQ(idx.chunks[1].offset, kChunkHeaderSize + 3);
}

TEST(CcmfFile, IndexAllSkipsUnknownType) {
    const TempFile file(Concat({
        BuildChunk(0, /*type=*/2, MakeBytes({9, 9})),  // subtitle: deferred, must be skipped
        BuildChunk(100, kChunkTypeVideo, MakeBytes({1})),
    }));

    CcmfFile ccmf(file.Path());
    const CcmfFile::FullIndex idx = ccmf.IndexAll();
    EXPECT_EQ(idx.chunks.size(), 2u);       // still counted in the full list
    EXPECT_EQ(idx.video.size(), 1u);        // but not in the video partition
    EXPECT_EQ(idx.audio.size(), 0u);
}

TEST(CcmfFile, IndexAllAllowsEqualPts) {
    // Equal PTS across chunks is legitimate (a GOP and an audio chunk starting
    // together); the lazy reader no longer enforces any monotonicity at load.
    const TempFile file(Concat({
        BuildChunk(1000, kChunkTypeVideo, MakeBytes({1})),
        BuildChunk(1000, kChunkTypeVideo, MakeBytes({2})),
    }));
    CcmfFile ccmf(file.Path());
    EXPECT_EQ(ccmf.IndexAll().video.size(), 2u);
}

TEST(CcmfFile, IndexAllThrowsOnTruncatedPayload) {
    auto chunk = BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3, 4}));
    chunk.resize(chunk.size() - 2);  // declared length=4, but only 2 bytes follow
    const TempFile file(chunk);
    CcmfFile ccmf(file.Path());
    EXPECT_THROW((void)ccmf.IndexAll(), CcmfError);
}

TEST(CcmfFile, IndexAllThrowsOnBadMarker) {
    auto chunk = BuildChunk(0, kChunkTypeVideo, MakeBytes({1}));
    chunk[0] = static_cast<std::byte>(0xFF);
    const TempFile file(chunk);
    CcmfFile ccmf(file.Path());
    EXPECT_THROW((void)ccmf.IndexAll(), CcmfError);
}

// --------------------------------------------------------------------------
// Degenerate files.
// --------------------------------------------------------------------------

TEST(CcmfFile, EmptyFileHasNoChunks) {
    const TempFile file(std::vector<std::byte>{});
    CcmfFile ccmf(file.Path());
    EXPECT_EQ(ccmf.FileSize(), 0u);
    EXPECT_FALSE(ccmf.FirstChunk().has_value());
    EXPECT_TRUE(ccmf.TailChunks().empty());
    EXPECT_TRUE(ccmf.IndexAll().chunks.empty());
    EXPECT_TRUE(ccmf.KnownChunks().empty());
}

TEST(CcmfFile, MissingFileThrows) {
    // A path under the (real) OS temp dir that we make sure doesn't exist,
    // rather than a fictitious drive letter -- an unmapped drive letter can
    // make Windows spend several seconds probing it before failing.
    const auto missing =
        std::filesystem::temp_directory_path() / "player_missing_file_test.ccmf";
    std::filesystem::remove(missing);
    EXPECT_THROW(CcmfFile{missing}, CcmfError);
}

// Cross-check IndexAll against the Python reference decoder (server/ccmf.py).
// See the original note: this fixture is one video GOP + two stereo audio
// chunks (front-left, front-right) sharing PTS 0 (spec 4.6).
TEST(CcmfFile, IndexAllMatchesPythonReferenceDecoderForRealFixture) {
    CcmfFile ccmf(std::filesystem::path(CCMF_TEST_FIXTURES_DIR) / "small_stereo.ccmf");
    const CcmfFile::FullIndex idx = ccmf.IndexAll();

    ASSERT_EQ(idx.chunks.size(), 3u);
    ASSERT_EQ(idx.video.size(), 1u);
    ASSERT_EQ(idx.audio.size(), 2u);

    EXPECT_EQ(idx.video[0].pts, 0u);
    EXPECT_EQ(idx.video[0].length, 6414u);

    EXPECT_EQ(idx.audio[0].pts, 0u);
    EXPECT_EQ(idx.audio[0].length, 72001u);
    EXPECT_EQ(idx.audio[1].pts, 0u);
    EXPECT_EQ(idx.audio[1].length, 72001u);
}

// The lazy tail read must agree with the eager scan about where the file ends.
TEST(CcmfFile, TailChunkMatchesIndexAllForRealFixture) {
    CcmfFile lazy(std::filesystem::path(CCMF_TEST_FIXTURES_DIR) / "multi_gop_mono.ccmf");
    const std::vector<ChunkEntry>& tail = lazy.TailChunks();
    ASSERT_FALSE(tail.empty());

    CcmfFile eager(std::filesystem::path(CCMF_TEST_FIXTURES_DIR) / "multi_gop_mono.ccmf");
    const CcmfFile::FullIndex idx = eager.IndexAll();
    ASSERT_FALSE(idx.chunks.empty());

    EXPECT_EQ(tail.back().offset, idx.chunks.back().offset);
    EXPECT_EQ(tail.back().pts, idx.chunks.back().pts);
    EXPECT_EQ(tail.back().End(), lazy.FileSize());
}

}  // namespace
}  // namespace ccmfplayer

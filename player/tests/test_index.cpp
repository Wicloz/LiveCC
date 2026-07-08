#include "engine/index.hpp"

#include <gtest/gtest.h>

#include <vector>

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

TEST(CcmfFile, IndexesAndPartitionsByType) {
    const TempFile file(Concat({
        BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3})),
        BuildChunk(0, kChunkTypeAudio, MakeBytes({4, 5})),
        BuildChunk(2000, kChunkTypeVideo, MakeBytes({6})),
        BuildChunk(2000, kChunkTypeAudio, MakeBytes({})),
    }));

    const CcmfFile ccmf(file.Path());

    ASSERT_EQ(ccmf.Chunks().size(), 4u);
    ASSERT_EQ(ccmf.VideoChunks().size(), 2u);
    ASSERT_EQ(ccmf.AudioChunks().size(), 2u);
    EXPECT_EQ(ccmf.VideoChunks()[0].pts, 0u);
    EXPECT_EQ(ccmf.VideoChunks()[1].pts, 2000u);
    EXPECT_EQ(ccmf.AudioChunks()[0].length, 2u);
    EXPECT_EQ(ccmf.AudioChunks()[1].length, 0u);
}

TEST(CcmfFile, ChunkOffsetsAdvanceByHeaderPlusLength) {
    const TempFile file(Concat({
        BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3})),
        BuildChunk(100, kChunkTypeVideo, MakeBytes({4})),
    }));

    const CcmfFile ccmf(file.Path());
    ASSERT_EQ(ccmf.Chunks().size(), 2u);
    EXPECT_EQ(ccmf.Chunks()[0].offset, 0u);
    EXPECT_EQ(ccmf.Chunks()[1].offset, kChunkHeaderSize + 3);
}

TEST(CcmfFile, ReadChunkPayloadReturnsExactBytes) {
    const auto payload = MakeBytes({0xDE, 0xAD, 0xBE, 0xEF});
    const TempFile file(BuildChunk(1234, kChunkTypeVideo, payload));
    const CcmfFile ccmf(file.Path());

    ASSERT_EQ(ccmf.VideoChunks().size(), 1u);
    const std::vector<std::byte> read = ccmf.ReadChunkPayload(ccmf.VideoChunks()[0]);
    EXPECT_EQ(read, payload);
}

TEST(CcmfFile, ReadChunkPayloadOfSecondChunkSkipsTheFirst) {
    const auto firstPayload = MakeBytes({1, 1, 1});
    const auto secondPayload = MakeBytes({2, 2});
    const TempFile file(Concat({
        BuildChunk(0, kChunkTypeVideo, firstPayload),
        BuildChunk(500, kChunkTypeVideo, secondPayload),
    }));
    const CcmfFile ccmf(file.Path());

    ASSERT_EQ(ccmf.VideoChunks().size(), 2u);
    EXPECT_EQ(ccmf.ReadChunkPayload(ccmf.VideoChunks()[1]), secondPayload);
}

TEST(CcmfFile, ReadChunkPayloadRangeReadsASlice) {
    const auto payload = MakeBytes({0xAA, 0xBB, 0xCC, 0xDD, 0xEE});
    const TempFile file(BuildChunk(0, kChunkTypeAudio, payload));
    const CcmfFile ccmf(file.Path());

    ASSERT_EQ(ccmf.AudioChunks().size(), 1u);
    const auto& entry = ccmf.AudioChunks()[0];
    EXPECT_EQ(ccmf.ReadChunkPayloadRange(entry, 0, 1), MakeBytes({0xAA}));
    EXPECT_EQ(ccmf.ReadChunkPayloadRange(entry, 2, 2), MakeBytes({0xCC, 0xDD}));
    EXPECT_EQ(ccmf.ReadChunkPayloadRange(entry, 0, 5), payload);
}

TEST(CcmfFile, ReadChunkPayloadRangeOutOfBoundsThrows) {
    const auto payload = MakeBytes({1, 2, 3});
    const TempFile file(BuildChunk(0, kChunkTypeAudio, payload));
    const CcmfFile ccmf(file.Path());

    const auto& entry = ccmf.AudioChunks()[0];
    EXPECT_THROW((void)ccmf.ReadChunkPayloadRange(entry, 2, 5), CcmfError);
}

TEST(CcmfFile, UnknownChunkTypeIsSkippedNotRejected) {
    const TempFile file(Concat({
        BuildChunk(0, /*type=*/2, MakeBytes({9, 9})),  // subtitle: deferred, must be skipped
        BuildChunk(100, kChunkTypeVideo, MakeBytes({1})),
    }));

    const CcmfFile ccmf(file.Path());
    EXPECT_EQ(ccmf.Chunks().size(), 2u);       // still counted in the full list
    EXPECT_EQ(ccmf.VideoChunks().size(), 1u);  // but not in the video partition
    EXPECT_EQ(ccmf.AudioChunks().size(), 0u);
}

TEST(CcmfFile, TruncatedPayloadThrows) {
    auto chunk = BuildChunk(0, kChunkTypeVideo, MakeBytes({1, 2, 3, 4}));
    chunk.resize(chunk.size() - 2);  // declared length=4, but only 2 bytes follow
    const TempFile file(chunk);

    EXPECT_THROW(CcmfFile{file.Path()}, CcmfError);
}

TEST(CcmfFile, TruncatedHeaderThrows) {
    const TempFile file(MakeBytes({0x43, 1, 2, 3, 4, 5}));  // 6 bytes, header needs 12
    EXPECT_THROW(CcmfFile{file.Path()}, CcmfError);
}

TEST(CcmfFile, BadMarkerThrows) {
    auto chunk = BuildChunk(0, kChunkTypeVideo, MakeBytes({1}));
    chunk[0] = static_cast<std::byte>(0xFF);
    const TempFile file(chunk);

    EXPECT_THROW(CcmfFile{file.Path()}, CcmfError);
}

TEST(CcmfFile, NonMonotonicPtsThrows) {
    const TempFile file(Concat({
        BuildChunk(2000, kChunkTypeVideo, MakeBytes({1})),
        BuildChunk(1000, kChunkTypeVideo, MakeBytes({2})),  // pts goes backwards
    }));

    EXPECT_THROW(CcmfFile{file.Path()}, CcmfError);
}

TEST(CcmfFile, EqualPtsIsAllowedNotRejected) {
    // Video and audio chunks legitimately share a PTS (e.g. a GOP and an
    // audio chunk starting at the same instant); only a *decrease* is an error.
    const TempFile file(Concat({
        BuildChunk(1000, kChunkTypeVideo, MakeBytes({1})),
        BuildChunk(1000, kChunkTypeVideo, MakeBytes({2})),
    }));

    const CcmfFile ccmf(file.Path());
    EXPECT_EQ(ccmf.VideoChunks().size(), 2u);
}

TEST(CcmfFile, EmptyFileHasNoChunks) {
    const TempFile file(std::vector<std::byte>{});
    const CcmfFile ccmf(file.Path());
    EXPECT_TRUE(ccmf.Chunks().empty());
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

// Cross-check against the Python reference decoder (server/ccmf.py). This
// fixture was rendered with `python tools/render_cc.py big_jungus.mp4
// --grid pocket --fps 10 --duration 1.5 --channels stereo`, and the expected
// values below are exactly what
//   python -c "import ccmf; [print(pts, ctype, len(payload)) for pts, ctype,
//               payload in ccmf.iter_chunks(open('small_stereo.ccmf','rb').read())]"
// printed for the same bytes: one video GOP (the whole clip fits one GOP at
// this duration) and two stereo audio chunks (front-left, front-right)
// sharing PTS 0, matching spec 4.6's "stereo is two audio chunks... sharing
// PTS".
TEST(CcmfFile, MatchesPythonReferenceDecoderForRealFixture) {
    const CcmfFile ccmf(std::filesystem::path(CCMF_TEST_FIXTURES_DIR) / "small_stereo.ccmf");

    ASSERT_EQ(ccmf.Chunks().size(), 3u);
    ASSERT_EQ(ccmf.VideoChunks().size(), 1u);
    ASSERT_EQ(ccmf.AudioChunks().size(), 2u);

    EXPECT_EQ(ccmf.VideoChunks()[0].pts, 0u);
    EXPECT_EQ(ccmf.VideoChunks()[0].length, 6414u);

    EXPECT_EQ(ccmf.AudioChunks()[0].pts, 0u);
    EXPECT_EQ(ccmf.AudioChunks()[0].length, 72001u);
    EXPECT_EQ(ccmf.AudioChunks()[1].pts, 0u);
    EXPECT_EQ(ccmf.AudioChunks()[1].length, 72001u);
}

}  // namespace
}  // namespace ccmfplayer

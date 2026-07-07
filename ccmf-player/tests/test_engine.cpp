#include "engine/engine.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <vector>

#include "test_support.hpp"

namespace ccmfplayer {
namespace {

using testing_support::TempFile;

std::filesystem::path Fixture(const char* name) {
    return std::filesystem::path(CCMF_TEST_FIXTURES_DIR) / name;
}

// --------------------------------------------------------------------------
// FindChunkIndexForPts: pure, exhaustively tested at every boundary.
// --------------------------------------------------------------------------

TEST(FindChunkIndexForPts, EmptyChunksReturnsZero) {
    EXPECT_EQ(FindChunkIndexForPts({}, 12345), 0u);
}

TEST(FindChunkIndexForPts, SingleChunkAlwaysReturnsZero) {
    const std::vector<ChunkEntry> chunks = {ChunkEntry{0, 500, 10, kChunkTypeVideo, 0}};
    EXPECT_EQ(FindChunkIndexForPts(chunks, 0), 0u);
    EXPECT_EQ(FindChunkIndexForPts(chunks, 500), 0u);
    EXPECT_EQ(FindChunkIndexForPts(chunks, 999999), 0u);
}

TEST(FindChunkIndexForPts, MultiChunkBoundaries) {
    const std::vector<ChunkEntry> chunks = {
        ChunkEntry{0, 0, 1, kChunkTypeVideo, 0},
        ChunkEntry{1, 100, 1, kChunkTypeVideo, 0},
        ChunkEntry{2, 250, 1, kChunkTypeVideo, 0},
    };
    EXPECT_EQ(FindChunkIndexForPts(chunks, 0), 0u);
    EXPECT_EQ(FindChunkIndexForPts(chunks, 50), 0u);
    EXPECT_EQ(FindChunkIndexForPts(chunks, 99), 0u);
    EXPECT_EQ(FindChunkIndexForPts(chunks, 100), 1u);
    EXPECT_EQ(FindChunkIndexForPts(chunks, 200), 1u);
    EXPECT_EQ(FindChunkIndexForPts(chunks, 250), 2u);
    EXPECT_EQ(FindChunkIndexForPts(chunks, 100000), 2u);  // clamps to last
}

// --------------------------------------------------------------------------
// FindFrameIndexForPts: pure, exhaustively tested at every boundary.
// --------------------------------------------------------------------------

TEST(FindFrameIndexForPts, AccumulatesDurationsAndClamps) {
    std::vector<Frame> frames(3);
    frames[0].duration = 10;
    frames[1].duration = 20;
    frames[2].duration = 5;
    // Boundaries: [0,10) -> 0, [10,30) -> 1, [30,35) -> 2, >=35 clamps to 2.
    EXPECT_EQ(FindFrameIndexForPts(frames, 0), 0u);
    EXPECT_EQ(FindFrameIndexForPts(frames, 9), 0u);
    EXPECT_EQ(FindFrameIndexForPts(frames, 10), 1u);
    EXPECT_EQ(FindFrameIndexForPts(frames, 29), 1u);
    EXPECT_EQ(FindFrameIndexForPts(frames, 30), 2u);
    EXPECT_EQ(FindFrameIndexForPts(frames, 34), 2u);
    EXPECT_EQ(FindFrameIndexForPts(frames, 35), 2u);
    EXPECT_EQ(FindFrameIndexForPts(frames, 1000000), 2u);
}

TEST(FindFrameIndexForPts, SingleFrameAlwaysZero) {
    std::vector<Frame> frames(1);
    frames[0].duration = 4800;
    EXPECT_EQ(FindFrameIndexForPts(frames, 0), 0u);
    EXPECT_EQ(FindFrameIndexForPts(frames, 4799), 0u);
    EXPECT_EQ(FindFrameIndexForPts(frames, 999999), 0u);
}

// --------------------------------------------------------------------------
// PlaybackEngine: construction and basic properties, against real fixtures.
// multi_gop_mono.ccmf: 5 video GOPs + 5 mono audio chunks, pts boundaries
// 0/48000/96000/144000/192000, total duration 216000 (4.5s @ 48kHz), pocket
// grid (26x20). Verified once with the Python reference when the fixture was
// generated (see the project plan); here the engine's own seek/frame
// resolution is cross-checked against DecodeVideoPayload/DecodeAudioPayload
// applied directly to the same chunks, since those are already proven
// byte-exact against Python in test_video.cpp/test_audio.cpp.
// --------------------------------------------------------------------------

TEST(PlaybackEngine, ConstructionReportsFileProperties) {
    PlaybackEngine engine(Fixture("multi_gop_mono.ccmf"));
    EXPECT_TRUE(engine.HasVideo());
    EXPECT_TRUE(engine.HasAudio());
    EXPECT_EQ(engine.ChannelCount(), 1u);
    EXPECT_EQ(engine.Width(), 26);
    EXPECT_EQ(engine.Height(), 20);
    EXPECT_EQ(engine.Duration(), 216000u);
    EXPECT_FALSE(engine.IsPlaying());
    EXPECT_EQ(engine.CurrentPts(), 0u);
    EXPECT_FALSE(engine.HasError());
}

TEST(PlaybackEngine, VideoOnlyFixtureHasNoAudio) {
    PlaybackEngine engine(Fixture("video_only.ccmf"));
    EXPECT_TRUE(engine.HasVideo());
    EXPECT_FALSE(engine.HasAudio());
    EXPECT_EQ(engine.ChannelCount(), 0u);

    std::vector<std::int16_t> buf(100, 1234);
    EXPECT_EQ(engine.PullAudio(buf), 0u);
    EXPECT_EQ(buf[0], 1234);  // untouched, per PullAudio's documented contract
}

TEST(PlaybackEngine, AudioOnlyFixtureHasNoVideo) {
    PlaybackEngine engine(Fixture("audio_only.ccmf"));
    EXPECT_FALSE(engine.HasVideo());
    EXPECT_TRUE(engine.HasAudio());
    EXPECT_EQ(engine.CurrentFrame(), nullptr);
}

// --------------------------------------------------------------------------
// Seeking: cross-checked against direct decode of the same chunks.
// --------------------------------------------------------------------------

TEST(PlaybackEngine, SeekToStartShowsFirstFrame) {
    PlaybackEngine engine(Fixture("multi_gop_mono.ccmf"));
    CcmfFile raw(Fixture("multi_gop_mono.ccmf"));
    const DecodedGop expectedGop0 = DecodeVideoPayload(raw.ReadChunkPayload(raw.VideoChunks()[0]));

    engine.Seek(0);
    ASSERT_NE(engine.CurrentFrame(), nullptr);
    EXPECT_EQ(engine.CurrentFrame()->glyph, expectedGop0.frames.front().glyph);
    EXPECT_EQ(engine.CurrentFrame()->fg, expectedGop0.frames.front().fg);
    EXPECT_EQ(engine.CurrentFrame()->bg, expectedGop0.frames.front().bg);
}

TEST(PlaybackEngine, SeekIntoSecondGopDecodesTheRightChunk) {
    PlaybackEngine engine(Fixture("multi_gop_mono.ccmf"));
    CcmfFile raw(Fixture("multi_gop_mono.ccmf"));
    const DecodedGop expectedGop1 = DecodeVideoPayload(raw.ReadChunkPayload(raw.VideoChunks()[1]));

    engine.Seek(50000);  // GOP1 starts at 48000; 2000 samples into it
    ASSERT_NE(engine.CurrentFrame(), nullptr);
    const std::size_t expectedFrameIdx = FindFrameIndexForPts(expectedGop1.frames, 2000);
    EXPECT_EQ(engine.CurrentFrame()->glyph, expectedGop1.frames[expectedFrameIdx].glyph);
}

TEST(PlaybackEngine, SeekToExactDurationShowsLastFrameOfLastGop) {
    PlaybackEngine engine(Fixture("multi_gop_mono.ccmf"));
    CcmfFile raw(Fixture("multi_gop_mono.ccmf"));
    const DecodedGop expectedLastGop =
        DecodeVideoPayload(raw.ReadChunkPayload(raw.VideoChunks().back()));

    engine.Seek(216000);
    EXPECT_EQ(engine.CurrentPts(), 216000u);
    ASSERT_NE(engine.CurrentFrame(), nullptr);
    EXPECT_EQ(engine.CurrentFrame()->glyph, expectedLastGop.frames.back().glyph);
}

TEST(PlaybackEngine, SeekPastDurationClamps) {
    PlaybackEngine engine(Fixture("multi_gop_mono.ccmf"));
    engine.Seek(999999999);
    EXPECT_EQ(engine.CurrentPts(), engine.Duration());
}

TEST(PlaybackEngine, RepeatedSeeksWithinTheSameGopAreCheap) {
    // Not a timing test (that would be flaky); just confirms seeking
    // backward and forward within one already-decoded GOP keeps returning
    // sensible, distinct frames rather than getting stuck.
    PlaybackEngine engine(Fixture("multi_gop_mono.ccmf"));
    engine.Seek(1000);
    const auto first = engine.CurrentFrame()->glyph;
    engine.Seek(40000);  // still within GOP0 (ends at 48000)
    const auto second = engine.CurrentFrame()->glyph;
    engine.Seek(1000);
    const auto third = engine.CurrentFrame()->glyph;
    EXPECT_EQ(first, third);
    // (second may or may not differ from first/third depending on content;
    // the meaningful assertion is that re-seeking back reproduces the same
    // frame deterministically, which the above checks.)
    (void)second;
}

// --------------------------------------------------------------------------
// Audio pulling: cross-checked against direct decode of the same chunks.
// --------------------------------------------------------------------------

TEST(PlaybackEngine, PullAudioFromStartMatchesDirectDecode) {
    PlaybackEngine engine(Fixture("multi_gop_mono.ccmf"));
    CcmfFile raw(Fixture("multi_gop_mono.ccmf"));
    const DecodedAudio expected = DecodeAudioPayload(raw.ReadChunkPayload(raw.AudioChunks()[0]));

    engine.Seek(0);
    std::vector<std::int16_t> buf(100);
    const std::size_t written = engine.PullAudio(buf);
    ASSERT_EQ(written, 100u);
    for (std::size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(buf[i], expected.pcm[i]) << "sample " << i;
    }
}

TEST(PlaybackEngine, PullAudioAcrossChunkBoundaryIsContinuous) {
    PlaybackEngine engine(Fixture("multi_gop_mono.ccmf"));
    CcmfFile raw(Fixture("multi_gop_mono.ccmf"));
    const DecodedAudio chunk0 = DecodeAudioPayload(raw.ReadChunkPayload(raw.AudioChunks()[0]));
    const DecodedAudio chunk1 = DecodeAudioPayload(raw.ReadChunkPayload(raw.AudioChunks()[1]));
    ASSERT_EQ(chunk0.pcm.size(), 48000u);

    engine.Seek(0);
    std::vector<std::int16_t> buf(48000 + 100);
    const std::size_t written = engine.PullAudio(buf);
    ASSERT_EQ(written, buf.size());

    EXPECT_TRUE(std::equal(chunk0.pcm.begin(), chunk0.pcm.end(), buf.begin()));
    for (std::size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(buf[48000 + i], chunk1.pcm[i]) << "sample " << i << " past the boundary";
    }
}

TEST(PlaybackEngine, PullAudioNearEndOfTrackReturnsOnlyWhatRemains) {
    PlaybackEngine engine(Fixture("multi_gop_mono.ccmf"));
    engine.Seek(216000 - 50);
    std::vector<std::int16_t> buf(1000, -1);
    const std::size_t written = engine.PullAudio(buf);
    EXPECT_EQ(written, 50u);
    EXPECT_EQ(buf[50], -1);  // untouched past what's available
}

TEST(PlaybackEngine, SeekResetsTheAudioCursor) {
    PlaybackEngine engine(Fixture("multi_gop_mono.ccmf"));
    std::vector<std::int16_t> buf(1000);
    engine.PullAudio(buf);  // advance the cursor away from 0

    engine.Seek(0);
    CcmfFile raw(Fixture("multi_gop_mono.ccmf"));
    const DecodedAudio expected = DecodeAudioPayload(raw.ReadChunkPayload(raw.AudioChunks()[0]));
    std::vector<std::int16_t> after(100);
    engine.PullAudio(after);
    for (std::size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(after[i], expected.pcm[i]) << "sample " << i;
    }
}

TEST(PlaybackEngine, StereoAudioInterleavesLeftAndRight) {
    PlaybackEngine engine(Fixture("small_stereo.ccmf"));
    ASSERT_TRUE(engine.HasAudio());
    ASSERT_EQ(engine.ChannelCount(), 2u);

    CcmfFile raw(Fixture("small_stereo.ccmf"));
    ASSERT_EQ(raw.AudioChunks().size(), 2u);
    const DecodedAudio left = DecodeAudioPayload(raw.ReadChunkPayload(raw.AudioChunks()[0]));
    const DecodedAudio right = DecodeAudioPayload(raw.ReadChunkPayload(raw.AudioChunks()[1]));
    ASSERT_EQ(left.channel, kChannelFrontLeft);
    ASSERT_EQ(right.channel, kChannelFrontRight);

    engine.Seek(0);
    std::vector<std::int16_t> buf(20);  // 10 frames x 2 channels
    const std::size_t written = engine.PullAudio(buf);
    ASSERT_EQ(written, 20u);
    for (std::size_t frame = 0; frame < 10; ++frame) {
        EXPECT_EQ(buf[frame * 2 + 0], left.pcm[frame]) << "frame " << frame << " left";
        EXPECT_EQ(buf[frame * 2 + 1], right.pcm[frame]) << "frame " << frame << " right";
    }
}

// --------------------------------------------------------------------------
// Play / Pause / Advance.
// --------------------------------------------------------------------------

TEST(PlaybackEngine, PlayPauseToggle) {
    PlaybackEngine engine(Fixture("multi_gop_mono.ccmf"));
    EXPECT_FALSE(engine.IsPlaying());
    engine.Play();
    EXPECT_TRUE(engine.IsPlaying());
    engine.Pause();
    EXPECT_FALSE(engine.IsPlaying());
    engine.TogglePlayPause();
    EXPECT_TRUE(engine.IsPlaying());
    engine.TogglePlayPause();
    EXPECT_FALSE(engine.IsPlaying());
}

TEST(PlaybackEngine, AdvanceWhilePausedDoesNothing) {
    PlaybackEngine engine(Fixture("multi_gop_mono.ccmf"));
    engine.Seek(1000);
    engine.Advance(1.0);
    EXPECT_EQ(engine.CurrentPts(), 1000u);
}

TEST(PlaybackEngine, AdvanceFollowsTheAudioCursorWhenAudioIsPresent) {
    PlaybackEngine engine(Fixture("multi_gop_mono.ccmf"));
    engine.Seek(0);
    engine.Play();

    // Simulate the app layer's audio device consuming samples: pulling
    // audio is what actually advances the master clock in this design (see
    // engine.hpp's sync-model doc), not the dt passed to Advance().
    std::vector<std::int16_t> buf(5000);
    engine.PullAudio(buf);

    engine.Advance(0.001);  // a tiny/irrelevant dt: audio, not dt, drives the clock
    EXPECT_EQ(engine.CurrentPts(), 5000u);
}

TEST(PlaybackEngine, AdvanceIsDtDrivenForVideoOnlyFiles) {
    PlaybackEngine engine(Fixture("video_only.ccmf"));
    ASSERT_FALSE(engine.HasAudio());
    engine.Play();
    engine.Advance(0.1);  // 0.1s @ 48kHz = 4800 samples
    EXPECT_EQ(engine.CurrentPts(), 4800u);
}

TEST(PlaybackEngine, AdvancePastEndClampsAndAutoPauses) {
    PlaybackEngine engine(Fixture("video_only.ccmf"));
    engine.Play();
    engine.Advance(1000.0);  // way past the 1s clip
    EXPECT_EQ(engine.CurrentPts(), engine.Duration());
    EXPECT_FALSE(engine.IsPlaying());
}

TEST(PlaybackEngine, PlayAfterReachingEndReplaysFromStart) {
    PlaybackEngine engine(Fixture("video_only.ccmf"));
    engine.Seek(engine.Duration());
    engine.Play();  // at/past the end when Play() is called
    EXPECT_EQ(engine.CurrentPts(), 0u);
    EXPECT_TRUE(engine.IsPlaying());
}

// --------------------------------------------------------------------------
// Looping.
// --------------------------------------------------------------------------

TEST(PlaybackEngine, LoopingIsOffByDefault) {
    PlaybackEngine engine(Fixture("video_only.ccmf"));
    EXPECT_FALSE(engine.IsLooping());
}

TEST(PlaybackEngine, SetAndToggleLooping) {
    PlaybackEngine engine(Fixture("video_only.ccmf"));
    engine.SetLooping(true);
    EXPECT_TRUE(engine.IsLooping());
    engine.ToggleLooping();
    EXPECT_FALSE(engine.IsLooping());
    engine.ToggleLooping();
    EXPECT_TRUE(engine.IsLooping());
}

TEST(PlaybackEngine, AdvancePastEndRestartsWhenLooping) {
    PlaybackEngine engine(Fixture("video_only.ccmf"));
    engine.SetLooping(true);
    engine.Play();
    engine.Advance(1000.0);  // way past the 1s clip
    EXPECT_EQ(engine.CurrentPts(), 0u);
    EXPECT_TRUE(engine.IsPlaying());  // keeps playing, doesn't auto-pause
    ASSERT_NE(engine.CurrentFrame(), nullptr);
}

TEST(PlaybackEngine, LoopingContinuesPlayingAcrossMultipleWraps) {
    PlaybackEngine engine(Fixture("video_only.ccmf"));
    engine.SetLooping(true);
    engine.Play();
    // Several small steps that individually stay under the 1s duration, but
    // whose sum crosses it multiple times.
    for (int i = 0; i < 20; ++i) {
        engine.Advance(0.3);
    }
    EXPECT_TRUE(engine.IsPlaying());
    EXPECT_LT(engine.CurrentPts(), engine.Duration());
}

TEST(PlaybackEngine, AudioFollowsLoopRestartToo) {
    PlaybackEngine engine(Fixture("multi_gop_mono.ccmf"));
    engine.SetLooping(true);
    engine.Play();

    std::vector<std::int16_t> buf(engine.Duration());  // pull the whole track at once
    engine.PullAudio(buf);
    engine.Advance(0.001);  // audio is the master clock; this just syncs currentPts_

    EXPECT_EQ(engine.CurrentPts(), 0u);
    EXPECT_TRUE(engine.IsPlaying());

    // The audio cursor was reset by the loop restart too, so pulling again
    // from here reproduces the start of the track, not silence/EOF.
    CcmfFile raw(Fixture("multi_gop_mono.ccmf"));
    const DecodedAudio expected = DecodeAudioPayload(raw.ReadChunkPayload(raw.AudioChunks()[0]));
    std::vector<std::int16_t> after(100);
    const std::size_t written = engine.PullAudio(after);
    ASSERT_EQ(written, 100u);
    for (std::size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(after[i], expected.pcm[i]) << "sample " << i;
    }
}

// --------------------------------------------------------------------------
// Resilience: a corrupt chunk degrades gracefully rather than crashing.
// --------------------------------------------------------------------------

std::vector<std::byte> ReadWholeFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(size);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    return bytes;
}

TEST(PlaybackEngine, CorruptFirstVideoChunkThrowsFromConstructor) {
    const auto original = Fixture("multi_gop_mono.ccmf");
    std::vector<std::byte> bytes = ReadWholeFile(original);

    CcmfFile raw(original);
    const ChunkEntry& first = raw.VideoChunks()[0];
    // Stomp the whole payload with a byte pattern DecodeVideoPayload can't
    // parse as a valid unit stream (0xFF -> flags bit7 set, enc=7, unknown).
    for (std::uint64_t i = 0; i < first.length; ++i) {
        bytes[first.offset + kChunkHeaderSize + i] = std::byte{0xFF};
    }

    const TempFile corrupted(bytes);
    EXPECT_THROW(PlaybackEngine{corrupted.Path()}, CcmfError);
}

TEST(PlaybackEngine, CorruptMiddleVideoChunkDegradesGracefully) {
    const auto original = Fixture("multi_gop_mono.ccmf");
    std::vector<std::byte> bytes = ReadWholeFile(original);

    CcmfFile raw(original);
    const ChunkEntry& secondGop = raw.VideoChunks()[2];  // corrupt GOP index 2, not first/last
    for (std::uint64_t i = 0; i < secondGop.length; ++i) {
        bytes[secondGop.offset + kChunkHeaderSize + i] = std::byte{0xFF};
    }
    const DecodedGop expectedGop0 = DecodeVideoPayload(raw.ReadChunkPayload(raw.VideoChunks()[0]));

    const TempFile corrupted(bytes);
    PlaybackEngine engine(corrupted.Path());  // must NOT throw: only chunk 2 is bad
    EXPECT_FALSE(engine.HasError());

    engine.Seek(0);  // caches GOP0 (good)
    ASSERT_NE(engine.CurrentFrame(), nullptr);

    engine.Seek(secondGop.pts + 1000);  // lands in the corrupted GOP
    EXPECT_TRUE(engine.HasError());
    EXPECT_FALSE(engine.LastError().empty());
    // Still shows GOP0's content (its last frame, per ResolveVideoFrame's
    // documented degrade-in-place behavior) rather than crashing or
    // blanking.
    ASSERT_NE(engine.CurrentFrame(), nullptr);
    EXPECT_EQ(engine.CurrentFrame()->glyph, expectedGop0.frames.back().glyph);
}

}  // namespace
}  // namespace ccmfplayer

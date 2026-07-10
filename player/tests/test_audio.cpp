#include "engine/audio.hpp"

#include <gtest/gtest.h>

#include <filesystem>

#include "engine/index.hpp"
#include "test_support.hpp"

namespace ccmfplayer {
namespace {

using testing_support::MakeBytes;

TEST(DecodePcm8ToI16, KnownValues) {
    // amplitude = byte - 128, scaled to int16 by *256.
    const auto data = MakeBytes({0, 128, 255, 1, 127});
    const auto pcm = DecodePcm8ToI16(data);

    const std::vector<std::int16_t> expected = {
        static_cast<std::int16_t>(-32768),  // (0-128)*256
        0,                                  // (128-128)*256
        static_cast<std::int16_t>(32512),   // (255-128)*256
        static_cast<std::int16_t>(-32512),  // (1-128)*256
        static_cast<std::int16_t>(-256),    // (127-128)*256
    };
    EXPECT_EQ(pcm, expected);
}

TEST(DecodePcm8ToI16, EmptyInputIsEmptyOutput) {
    EXPECT_TRUE(DecodePcm8ToI16(std::span<const std::byte>{}).empty());
}

TEST(ScaleU8ToI16, MatchesDecodePcm8ForTheSameBytes) {
    const std::vector<std::uint8_t> u8 = {0, 64, 128, 192, 255};
    const auto viaScale = ScaleU8ToI16(u8);
    const auto viaPcm8 = DecodePcm8ToI16(MakeBytes({0, 64, 128, 192, 255}));
    EXPECT_EQ(viaScale, viaPcm8);
}

// Cross-checked against server/dfpwm.py's decode() (a fresh-state call, same
// contract as this function): dfpwm.decode(bytes([0x00]*4)).
TEST(DecodeDfpwmToU8, AllZeroBytesMatchesPythonReference) {
    const auto data = MakeBytes({0x00, 0x00, 0x00, 0x00});
    const auto out = DecodeDfpwmToU8(data);

    const std::vector<std::uint8_t> expected = {
        127, 126, 125, 124, 123, 122, 121, 119, 117, 115, 113, 111,
        109, 107, 105, 103, 101, 99,  97,  95,  93,  91,  88,  85,
        82,  80,  77,  75,  73,  71,  69,  67,
    };
    EXPECT_EQ(out, expected);
}

// dfpwm.decode(bytes([0xFF]*4)).
TEST(DecodeDfpwmToU8, AllOnesBytesMatchesPythonReference) {
    const auto data = MakeBytes({0xFF, 0xFF, 0xFF, 0xFF});
    const auto out = DecodeDfpwmToU8(data);

    const std::vector<std::uint8_t> expected = {
        129, 130, 131, 132, 133, 134, 135, 137, 139, 141, 143, 145,
        147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 170,
        173, 176, 178, 180, 182, 184, 186, 188,
    };
    EXPECT_EQ(out, expected);
}

// dfpwm.decode(bytes([0xAA]*4)) -- an alternating bit pattern.
TEST(DecodeDfpwmToU8, AlternatingBitsMatchesPythonReference) {
    const auto data = MakeBytes({0xAA, 0xAA, 0xAA, 0xAA});
    const auto out = DecodeDfpwmToU8(data);

    const std::vector<std::uint8_t> expected = {
        127, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
        128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
        128, 128, 128, 128, 128, 128, 128, 128,
    };
    EXPECT_EQ(out, expected);
}

// dfpwm.decode(dfpwm.encode(bytes((i*3) % 256 for i in range(32)))) -- a
// decode(encode(ramp)) round trip through the (lossy) codec; not expected to
// reproduce the original ramp, only to match what Python's decode() of the
// SAME encoded bytes produces.
TEST(DecodeDfpwmToU8, RoundTripMatchesPythonReference) {
    const auto encoded = MakeBytes({0, 0, 0, 216});
    const auto out = DecodeDfpwmToU8(encoded);

    const std::vector<std::uint8_t> expected = {
        127, 126, 125, 124, 123, 122, 121, 119, 117, 115, 113, 111,
        109, 107, 105, 103, 101, 99,  97,  95,  93,  91,  88,  85,
        82,  80,  77,  78,  83,  85,  86,  91,
    };
    EXPECT_EQ(out, expected);
}

TEST(DecodeDfpwmToU8, EmptyInputIsEmptyOutput) {
    EXPECT_TRUE(DecodeDfpwmToU8(std::span<const std::byte>{}).empty());
}

TEST(DecodeAudioPayload, DispatchesPcm8ByHighNibble) {
    // a-hdr = codec(0=pcm8)<<4 | channel(0=mono) = 0x00, then 2 PCM8 samples.
    const auto payload = MakeBytes({0x00, 128, 255});
    const DecodedAudio audio = DecodeAudioPayload(payload);

    EXPECT_EQ(audio.codec, kAudioCodecPcm8);
    EXPECT_EQ(audio.channel, kChannelMono);
    EXPECT_EQ(audio.pcm, DecodePcm8ToI16(MakeBytes({128, 255})));
}

TEST(DecodeAudioPayload, DispatchesDfpwmByHighNibbleAndExtractsChannel) {
    // a-hdr = codec(1=dfpwm)<<4 | channel(2=front_right) = 0x12.
    const auto payload = MakeBytes({0x12, 0xAA, 0xAA, 0xAA, 0xAA});
    const DecodedAudio audio = DecodeAudioPayload(payload);

    EXPECT_EQ(audio.codec, kAudioCodecDfpwm);
    EXPECT_EQ(audio.channel, kChannelFrontRight);
    EXPECT_EQ(audio.pcm, ScaleU8ToI16(DecodeDfpwmToU8(MakeBytes({0xAA, 0xAA, 0xAA, 0xAA}))));
}

TEST(DecodeAudioPayload, EmptyPayloadThrows) {
    EXPECT_THROW((void)DecodeAudioPayload(std::span<const std::byte>{}), CcmfError);
}

TEST(DecodeAudioPayload, UnsupportedCodecThrows) {
    // high nibble 0xF is not a defined codec.
    const auto payload = MakeBytes({0xF0, 1, 2, 3});
    EXPECT_THROW((void)DecodeAudioPayload(payload), CcmfError);
}

TEST(DecodeAudioPayload, HeaderOnlyPayloadIsEmptyPcm) {
    const auto payload = MakeBytes({0x00});  // pcm8, mono, zero samples
    const DecodedAudio audio = DecodeAudioPayload(payload);
    EXPECT_TRUE(audio.pcm.empty());
}

// Cross-check against the real fixture (rendered with --channels stereo,
// the default --audio-codec pcm): server/ccmf.py's parse_audio_payload on
// the same bytes reported codec=0 (pcm8), channel=1/2 (front-left/right),
// 72000 samples each, first 5 raw bytes all 128 (silence).
TEST(DecodeAudioPayload, MatchesPythonReferenceForRealFixture) {
    const std::filesystem::path fixturesDir(CCMF_TEST_FIXTURES_DIR);
    CcmfFile ccmf(fixturesDir / "small_stereo.ccmf");
    const CcmfFile::FullIndex idx = ccmf.IndexAll();
    ASSERT_EQ(idx.audio.size(), 2u);

    const DecodedAudio left = DecodeAudioPayload(ccmf.ReadChunkPayload(idx.audio[0]));
    EXPECT_EQ(left.codec, kAudioCodecPcm8);
    EXPECT_EQ(left.channel, kChannelFrontLeft);
    ASSERT_EQ(left.pcm.size(), 72000u);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(left.pcm[i], 0) << "sample " << i;  // byte 128 -> amplitude 0
    }

    const DecodedAudio right = DecodeAudioPayload(ccmf.ReadChunkPayload(idx.audio[1]));
    EXPECT_EQ(right.codec, kAudioCodecPcm8);
    EXPECT_EQ(right.channel, kChannelFrontRight);
    EXPECT_EQ(right.pcm.size(), 72000u);
}

// --------------------------------------------------------------------------
// Output layout + role remapping (the fixed-output / dynamic up-down-mix path).
// --------------------------------------------------------------------------

TEST(OutputRolesForChannelCount, MonoAndStereo) {
    EXPECT_EQ(OutputRolesForChannelCount(1), (std::vector<std::uint8_t>{kChannelMono}));
    EXPECT_EQ(OutputRolesForChannelCount(2),
              (std::vector<std::uint8_t>{kChannelFrontLeft, kChannelFrontRight}));
    // Anything else falls back to stereo.
    EXPECT_EQ(OutputRolesForChannelCount(6),
              (std::vector<std::uint8_t>{kChannelFrontLeft, kChannelFrontRight}));
    EXPECT_EQ(OutputRolesForChannelCount(0), (std::vector<std::uint8_t>{kChannelMono}));
}

TEST(ResolveOutputSources, ExactMatchPassesThrough) {
    EXPECT_EQ(ResolveOutputSources(kChannelFrontLeft, {kChannelFrontLeft, kChannelFrontRight}),
              (std::vector<std::uint8_t>{kChannelFrontLeft}));
    EXPECT_EQ(ResolveOutputSources(kChannelMono, {kChannelMono}),
              (std::vector<std::uint8_t>{kChannelMono}));
}

TEST(ResolveOutputSources, MonoSourceUpmixesToBothStereoChannels) {
    EXPECT_EQ(ResolveOutputSources(kChannelFrontLeft, {kChannelMono}),
              (std::vector<std::uint8_t>{kChannelMono}));
    EXPECT_EQ(ResolveOutputSources(kChannelFrontRight, {kChannelMono}),
              (std::vector<std::uint8_t>{kChannelMono}));
}

TEST(ResolveOutputSources, StereoSourceDownmixesToMonoOutput) {
    // Mono output with no mono source but L+R present -> average of the two.
    EXPECT_EQ(ResolveOutputSources(kChannelMono, {kChannelFrontLeft, kChannelFrontRight}),
              (std::vector<std::uint8_t>{kChannelFrontLeft, kChannelFrontRight}));
}

TEST(ResolveOutputSources, PrefersNearestNeighbourThenAnything) {
    // FL output, only FR present -> use FR rather than go silent.
    EXPECT_EQ(ResolveOutputSources(kChannelFrontLeft, {kChannelFrontRight}),
              (std::vector<std::uint8_t>{kChannelFrontRight}));
    // FR output, mono present -> prefer mono.
    EXPECT_EQ(ResolveOutputSources(kChannelFrontRight, {kChannelMono, kChannelFrontLeft}),
              (std::vector<std::uint8_t>{kChannelMono}));
}

TEST(ResolveOutputSources, EmptyPresentIsSilence) {
    EXPECT_TRUE(ResolveOutputSources(kChannelMono, {}).empty());
}

}  // namespace
}  // namespace ccmfplayer

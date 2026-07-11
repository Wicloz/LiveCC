#include "engine/video.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "engine/index.hpp"
#include "test_support.hpp"

namespace ccmfplayer {
namespace {

using testing_support::MakeBytes;

// A 48-byte palette unit body: 16 copies of one solid {r,g,b}, for tests that
// only care about which palette ends up in effect, not its exact contents.
std::vector<std::byte> SolidPaletteBody(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    std::vector<std::byte> body;
    body.reserve(48);
    for (int i = 0; i < 16; ++i) {
        body.push_back(std::byte{r});
        body.push_back(std::byte{g});
        body.push_back(std::byte{b});
    }
    return body;
}

// Cross-checked against server/ccmf.py's pack_chars for codes [0..7] (one
// full group of 8, no padding):
//   >>> ccmf.pack_chars(np.arange(8, dtype=np.uint8) + 0x80)
//   bytes([0, 68, 50, 20, 199])
TEST(UnpackChars, FullGroupMatchesPythonReference) {
    const auto packed = MakeBytes({0, 68, 50, 20, 199});
    const auto glyphs = UnpackChars(packed, 8);

    const std::vector<std::uint8_t> expected = {128, 129, 130, 131, 132, 133, 134, 135};
    EXPECT_EQ(glyphs, expected);
}

// Cross-checked against Python for codes [31, 0, 17] with n=3 (padded group):
//   >>> ccmf.pack_chars(np.array([31,0,17], dtype=np.uint8) + 0x80)
//   bytes([248, 34, 0, 0, 0])
TEST(UnpackChars, PaddedGroupMatchesPythonReference) {
    const auto packed = MakeBytes({248, 34, 0, 0, 0});
    const auto glyphs = UnpackChars(packed, 3);

    const std::vector<std::uint8_t> expected = {0x9F, 0x80, 0x91};
    EXPECT_EQ(glyphs, expected);
}

TEST(UnpackChars, TooShortThrows) {
    const auto packed = MakeBytes({0, 68, 50, 20});  // one byte short of a full group
    EXPECT_THROW((void)UnpackChars(packed, 8), CcmfError);
}

// Cross-checked against Python for indices [15, 0, 7, 8, 1] with n=5 (odd,
// padded):
//   >>> ccmf.pack_nibbles(np.array([15,0,7,8,1], dtype=np.uint8))
//   bytes([240, 120, 16])
TEST(UnpackNibbles, PaddedGroupMatchesPythonReference) {
    const auto packed = MakeBytes({240, 120, 16});
    const auto indices = UnpackNibbles(packed, 5);

    const std::vector<std::uint8_t> expected = {15, 0, 7, 8, 1};
    EXPECT_EQ(indices, expected);
}

TEST(UnpackNibbles, TooShortThrows) {
    const auto packed = MakeBytes({240});  // n=5 needs 3 bytes
    EXPECT_THROW((void)UnpackNibbles(packed, 5), CcmfError);
}

// A full synthetic GOP (palette + raw keyframe + delta + repeat) on a 2x2
// grid, built and decoded with server/ccmf.py to get ground truth:
//   palette = ((arange(48, dtype=int64) * 5) % 256).astype(uint8)
//   glyph0 = [[0x80, 0x81], [0x9F, 0x90]]; fg0 = [[0,1],[2,3]]; bg0 = [[4,5],[6,7]]
//   frame 1 changes cell (0,1) to glyph=0x85, fg=9, bg=10
//   frame 2 repeats frame 1
// `ccmf.video_payload(2, 2, ...)` for that sequence produced exactly the
// bytes below; `ccmf.parse_video_payload` on those bytes produced the
// expected frames asserted after.
TEST(DecodeVideoPayload, MatchesPythonReferenceForSyntheticGop) {
    const auto payload = MakeBytes({
        2,   0,   2,   0,   0,   0,   5,   10,  15,  20,  25,  30,  35,  40,  45,  50,
        55,  60,  65,  70,  75,  80,  85,  90,  95,  100, 105, 110, 115, 120, 125, 130,
        135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210,
        215, 220, 225, 230, 235, 128, 208, 7,   0,   127, 0,   0,   0,   1,   35,  69,
        103, 144, 232, 3,   1,   0,   1,   0,   1,   133, 169, 160, 244, 1,
    });

    const DecodedGop gop = DecodeVideoPayload(payload);

    ASSERT_EQ(gop.width, 2);
    ASSERT_EQ(gop.height, 2);
    ASSERT_EQ(gop.frames.size(), 3u);

    // Every frame shares the one palette unit at the start of the GOP.
    for (const Frame& frame : gop.frames) {
        EXPECT_EQ(frame.palette.colors[0], (std::array<std::uint8_t, 3>{0, 5, 10}));
        EXPECT_EQ(frame.palette.colors[1], (std::array<std::uint8_t, 3>{15, 20, 25}));
        EXPECT_EQ(frame.palette.colors[2], (std::array<std::uint8_t, 3>{30, 35, 40}));
    }

    const Frame& raw = gop.frames[0];
    EXPECT_EQ(raw.duration, 2000);
    EXPECT_EQ(raw.glyph, (std::vector<std::uint8_t>{0x80, 0x81, 0x9F, 0x90}));
    EXPECT_EQ(raw.fg, (std::vector<std::uint8_t>{0, 1, 2, 3}));
    EXPECT_EQ(raw.bg, (std::vector<std::uint8_t>{4, 5, 6, 7}));

    const Frame& delta = gop.frames[1];
    EXPECT_EQ(delta.duration, 1000);
    EXPECT_EQ(delta.glyph, (std::vector<std::uint8_t>{0x80, 0x85, 0x9F, 0x90}));
    EXPECT_EQ(delta.fg, (std::vector<std::uint8_t>{0, 9, 2, 3}));
    EXPECT_EQ(delta.bg, (std::vector<std::uint8_t>{4, 10, 6, 7}));

    // repeat holds whatever the delta frame left in place.
    const Frame& repeat = gop.frames[2];
    EXPECT_EQ(repeat.duration, 500);
    EXPECT_EQ(repeat.glyph, delta.glyph);
    EXPECT_EQ(repeat.fg, delta.fg);
    EXPECT_EQ(repeat.bg, delta.bg);
}

// Cross-checked against server/ccmf.py's ans_frame_unit for a 2x2 GOP:
//   palette = bytes(range(48)); glyph = [[0x80,0x81],[0x9F,0x90]];
//   fg = [[0,1],[2,3]]; bg = [[4,5],[6,7]] -> the byte list below.
// Exercises the rANS+RLE (spec 4.5.3) decoder against the reference encoder.
TEST(DecodeVideoPayload, AnsKeyframeMatchesPythonReference) {
    const auto payload = MakeBytes({
        2,0,2,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
        24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,
        176,208,7,66,0,0,1,4,0,0,4,1,0,4,16,0,4,31,0,4,5,0,0,0,1,0,180,0,1,4,0,0,
        4,1,0,4,2,0,4,3,0,4,5,0,0,0,1,0,228,0,1,4,4,0,4,5,0,4,6,0,4,7,0,4,5,0,0,
        0,1,0,228,0,
    });

    const DecodedGop gop = DecodeVideoPayload(payload);
    ASSERT_EQ(gop.width, 2);
    ASSERT_EQ(gop.height, 2);
    ASSERT_EQ(gop.frames.size(), 1u);

    const Frame& ans = gop.frames[0];
    EXPECT_EQ(ans.duration, 2000);
    EXPECT_EQ(ans.glyph, (std::vector<std::uint8_t>{0x80, 0x81, 0x9F, 0x90}));
    EXPECT_EQ(ans.fg, (std::vector<std::uint8_t>{0, 1, 2, 3}));
    EXPECT_EQ(ans.bg, (std::vector<std::uint8_t>{4, 5, 6, 7}));
    // An ANS keyframe is a RAP: a following delta must apply against it.
}

TEST(DecodeVideoPayload, TooShortHeaderThrows) {
    const auto payload = MakeBytes({0, 0, 1});  // needs 4 bytes for width/height
    EXPECT_THROW((void)DecodeVideoPayload(payload), CcmfError);
}

TEST(DecodeVideoPayload, NoFramesThrows) {
    // Just a width/height header and nothing else -- not even a palette.
    const auto payload = MakeBytes({1, 0, 1, 0});
    EXPECT_THROW((void)DecodeVideoPayload(payload), CcmfError);
}

TEST(DecodeVideoPayload, DeltaBeforeKeyframeThrows) {
    // width=1,height=1, palette unit(48 zero bytes), then a delta unit with no
    // prior raw keyframe.
    std::vector<std::byte> payload = MakeBytes({1, 0, 1, 0});
    std::vector<std::byte> paletteUnit(49, std::byte{0});  // flags(0) + 48 colour bytes
    payload.insert(payload.end(), paletteUnit.begin(), paletteUnit.end());
    // delta unit: flags=0x80|(1<<4)=0x90, duration=0, span count=0
    const auto deltaUnit = MakeBytes({0x90, 0, 0, 0, 0});
    payload.insert(payload.end(), deltaUnit.begin(), deltaUnit.end());

    EXPECT_THROW((void)DecodeVideoPayload(payload), CcmfError);
}

TEST(DecodeVideoPayload, RepeatBeforeKeyframeThrows) {
    std::vector<std::byte> payload = MakeBytes({1, 0, 1, 0});
    std::vector<std::byte> paletteUnit(49, std::byte{0});
    payload.insert(payload.end(), paletteUnit.begin(), paletteUnit.end());
    // repeat unit: flags=0x80|(2<<4)=0xA0, duration=0
    const auto repeatUnit = MakeBytes({0xA0, 0, 0});
    payload.insert(payload.end(), repeatUnit.begin(), repeatUnit.end());

    EXPECT_THROW((void)DecodeVideoPayload(payload), CcmfError);
}

TEST(DecodeVideoPayload, FrameBeforePaletteThrows) {
    // width=1,height=1, straight to a raw frame with no palette unit first.
    std::vector<std::byte> payload = MakeBytes({1, 0, 1, 0});
    // raw unit: flags=0x80|(0<<4)=0x80, duration=0, then RawPlanesSize(1,1) bytes.
    std::vector<std::byte> rawUnit = MakeBytes({0x80, 0, 0});
    std::vector<std::byte> planes(RawPlanesSize(1, 1), std::byte{0x80});
    rawUnit.insert(rawUnit.end(), planes.begin(), planes.end());
    payload.insert(payload.end(), rawUnit.begin(), rawUnit.end());

    EXPECT_THROW((void)DecodeVideoPayload(payload), CcmfError);
}

// Spec (docs/cc-media-format.md §4.4): a palette unit MUST NOT be immediately
// followed by another one -- but this reference player is deliberately
// permissive about a malformed stream rather than crashing on it, matching
// player.lua and reproducing this decoder's own long-standing behaviour for
// the "ends with a palette" case below. The first palette is silently
// superseded; only the second (the one actually in effect when the frame
// renders) is ever visible.
TEST(DecodeVideoPayload, ConsecutivePalettesArePermissiveAndLastOneWins) {
    std::vector<std::byte> payload = MakeBytes({1, 0, 1, 0});   // width=1, height=1

    std::vector<std::byte> pal1Unit{std::byte{0}};              // flags(0) + body
    const auto pal1Body = SolidPaletteBody(10, 20, 30);
    pal1Unit.insert(pal1Unit.end(), pal1Body.begin(), pal1Body.end());
    payload.insert(payload.end(), pal1Unit.begin(), pal1Unit.end());

    std::vector<std::byte> pal2Unit{std::byte{0}};
    const auto pal2Body = SolidPaletteBody(200, 210, 220);
    pal2Unit.insert(pal2Unit.end(), pal2Body.begin(), pal2Body.end());
    payload.insert(payload.end(), pal2Unit.begin(), pal2Unit.end());

    std::vector<std::byte> rawUnit = MakeBytes({0x80, 0, 0});   // raw, duration=0
    std::vector<std::byte> planes(RawPlanesSize(1, 1), std::byte{0x80});
    rawUnit.insert(rawUnit.end(), planes.begin(), planes.end());
    payload.insert(payload.end(), rawUnit.begin(), rawUnit.end());

    DecodedGop gop;
    EXPECT_NO_THROW(gop = DecodeVideoPayload(payload));
    ASSERT_EQ(gop.frames.size(), 1u);
    EXPECT_EQ(gop.frames[0].palette.colors[0], (std::array<std::uint8_t, 3>{200, 210, 220}));
}

// Spec §4.4: a palette unit MUST NOT be the last unit in a video payload --
// again, tolerated here rather than rejected: the dangling palette is parsed
// but never attached to any frame (nothing follows it to give it an effective
// time), so it's silently dropped.
TEST(DecodeVideoPayload, TrailingPaletteAfterAFrameIsPermissivelyIgnored) {
    std::vector<std::byte> payload = MakeBytes({1, 0, 1, 0});   // width=1, height=1

    std::vector<std::byte> palUnit{std::byte{0}};
    const auto palBody = SolidPaletteBody(0, 0, 0);
    palUnit.insert(palUnit.end(), palBody.begin(), palBody.end());
    payload.insert(payload.end(), palUnit.begin(), palUnit.end());

    std::vector<std::byte> rawUnit = MakeBytes({0x80, 0, 0});   // raw, duration=0
    std::vector<std::byte> planes(RawPlanesSize(1, 1), std::byte{0x80});
    rawUnit.insert(rawUnit.end(), planes.begin(), planes.end());
    payload.insert(payload.end(), rawUnit.begin(), rawUnit.end());

    std::vector<std::byte> danglingPalUnit{std::byte{0}};       // nothing follows this
    const auto danglingBody = SolidPaletteBody(99, 99, 99);
    danglingPalUnit.insert(danglingPalUnit.end(), danglingBody.begin(), danglingBody.end());
    payload.insert(payload.end(), danglingPalUnit.begin(), danglingPalUnit.end());

    DecodedGop gop;
    EXPECT_NO_THROW(gop = DecodeVideoPayload(payload));
    ASSERT_EQ(gop.frames.size(), 1u);
    EXPECT_EQ(gop.frames[0].palette.colors[0], (std::array<std::uint8_t, 3>{0, 0, 0}));
}

// --------------------------------------------------------------------------
// Cross-check against the real fixture small_stereo.ccmf (rendered with
// convert_to_ccmf.py) and small_stereo.frames.bin, a small binary dump this
// module's tests read directly: [u16 w][u16 h][u16 numFrames] then, for the
// first and last decoded frame, [u16 duration][u8 encoding][glyph w*h]
// [fg w*h][bg w*h][palette 48], produced by actually running
// server/ccmf.py's parse_video_payload on the fixture's video chunk (see the
// generating script referenced in the project plan).
struct ExpectedFrame {
    std::uint16_t duration = 0;
    std::uint8_t encoding = 0;
    std::vector<std::uint8_t> glyph, fg, bg;
    std::array<std::array<std::uint8_t, 3>, 16> palette{};
};

struct ExpectedFixture {
    std::uint16_t width = 0;
    std::uint16_t height = 0;
    std::uint16_t numFrames = 0;
    ExpectedFrame first;
    ExpectedFrame last;
};

ExpectedFixture ReadExpectedFixture(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    EXPECT_TRUE(static_cast<bool>(in)) << "couldn't open " << path;

    ExpectedFixture out;
    auto readU16 = [&in]() {
        unsigned char b[2];
        in.read(reinterpret_cast<char*>(b), 2);
        return static_cast<std::uint16_t>(b[0] | (b[1] << 8));
    };
    out.width = readU16();
    out.height = readU16();
    out.numFrames = readU16();

    const std::size_t n = static_cast<std::size_t>(out.width) * out.height;
    auto readFrame = [&]() {
        ExpectedFrame frame;
        frame.duration = readU16();
        unsigned char enc;
        in.read(reinterpret_cast<char*>(&enc), 1);
        frame.encoding = enc;
        frame.glyph.resize(n);
        in.read(reinterpret_cast<char*>(frame.glyph.data()),
                static_cast<std::streamsize>(n));
        frame.fg.resize(n);
        in.read(reinterpret_cast<char*>(frame.fg.data()), static_cast<std::streamsize>(n));
        frame.bg.resize(n);
        in.read(reinterpret_cast<char*>(frame.bg.data()), static_cast<std::streamsize>(n));
        in.read(reinterpret_cast<char*>(frame.palette.data()), 48);
        return frame;
    };
    out.first = readFrame();
    out.last = readFrame();
    return out;
}

TEST(DecodeVideoPayload, MatchesPythonReferenceForRealFixture) {
    const std::filesystem::path fixturesDir(CCMF_TEST_FIXTURES_DIR);
    CcmfFile ccmf(fixturesDir / "small_stereo.ccmf");
    const CcmfFile::FullIndex idx = ccmf.IndexAll();
    ASSERT_EQ(idx.video.size(), 1u);

    const DecodedGop gop = DecodeVideoPayload(ccmf.ReadChunkPayload(idx.video[0]));
    const ExpectedFixture expected = ReadExpectedFixture(fixturesDir / "small_stereo.frames.bin");

    ASSERT_EQ(gop.width, expected.width);
    ASSERT_EQ(gop.height, expected.height);
    ASSERT_EQ(gop.frames.size(), expected.numFrames);

    const Frame& first = gop.frames.front();
    EXPECT_EQ(first.duration, expected.first.duration);
    EXPECT_EQ(first.glyph, expected.first.glyph);
    EXPECT_EQ(first.fg, expected.first.fg);
    EXPECT_EQ(first.bg, expected.first.bg);
    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(first.palette.colors[i], expected.first.palette[i]) << "palette entry " << i;
    }

    const Frame& last = gop.frames.back();
    EXPECT_EQ(last.duration, expected.last.duration);
    EXPECT_EQ(last.glyph, expected.last.glyph);
    EXPECT_EQ(last.fg, expected.last.fg);
    EXPECT_EQ(last.bg, expected.last.bg);
}

}  // namespace
}  // namespace ccmfplayer

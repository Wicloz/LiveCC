#include "engine/render.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

namespace ccmfplayer {
namespace {

Palette MakePalette() {
    Palette p;
    for (int i = 0; i < 16; ++i) {
        p.colors[i] = {static_cast<std::uint8_t>(i), static_cast<std::uint8_t>((i * 2) % 256),
                        static_cast<std::uint8_t>((i * 3) % 256)};
    }
    return p;
}

// Cross-checked against server/cc_media.py's render_cells for a 3x2 cell
// grid exercising every controllable sub-pixel bit (glyph masks 0,1,2,4,8)
// plus the all-set case (mask 31) and the always-background 6th sub-pixel:
//
//   glyph = [[0x80,0x81,0x82],[0x84,0x88,0x9F]]
//   fg = [[0,1,2],[3,4,5]]; bg = [[10,11,12],[13,14,15]]
//   palette[i] = [i, i*2 % 256, i*3 % 256]
//   render_cells(glyph, fg, bg, palette)  ->  the 6x6x3 buffer flattened below.
TEST(RenderCellsToRgb, MatchesPythonReferenceForFullBitCoverage) {
    Frame frame;
    frame.glyph = {0x80, 0x81, 0x82, 0x84, 0x88, 0x9F};
    frame.fg = {0, 1, 2, 3, 4, 5};
    frame.bg = {10, 11, 12, 13, 14, 15};
    frame.palette = MakePalette();

    std::vector<std::uint8_t> out(RenderedPixelBufferSize(3, 2));
    RenderCellsToRgb(frame, 3, 2, out);

    const std::vector<std::uint8_t> expected = {
        // row 0
        10, 20, 30, 10, 20, 30, 1, 2, 3, 11, 22, 33, 12, 24, 36, 2, 4, 6,
        // row 1
        10, 20, 30, 10, 20, 30, 11, 22, 33, 11, 22, 33, 12, 24, 36, 12, 24, 36,
        // row 2
        10, 20, 30, 10, 20, 30, 11, 22, 33, 11, 22, 33, 12, 24, 36, 12, 24, 36,
        // row 3
        13, 26, 39, 13, 26, 39, 14, 28, 42, 14, 28, 42, 5, 10, 15, 5, 10, 15,
        // row 4
        3, 6, 9, 13, 26, 39, 14, 28, 42, 4, 8, 12, 5, 10, 15, 5, 10, 15,
        // row 5
        13, 26, 39, 13, 26, 39, 14, 28, 42, 14, 28, 42, 5, 10, 15, 15, 30, 45,
    };

    EXPECT_EQ(out, expected);
}

TEST(RenderCellsToRgb, WrongFrameSizeThrows) {
    Frame frame;
    frame.glyph = {0x80};
    frame.fg = {0};
    frame.bg = {0};
    frame.palette = MakePalette();

    // frame is a 1x1 grid; asking for a 2x2 buffer is a size mismatch.
    std::vector<std::uint8_t> out(RenderedPixelBufferSize(2, 2));
    EXPECT_THROW(RenderCellsToRgb(frame, 2, 2, out), CcmfError);
}

TEST(RenderCellsToRgb, WrongBufferSizeThrows) {
    Frame frame;
    frame.glyph = {0x80};
    frame.fg = {0};
    frame.bg = {0};
    frame.palette = MakePalette();

    std::vector<std::uint8_t> out(RenderedPixelBufferSize(1, 1) - 1);  // one byte short
    EXPECT_THROW(RenderCellsToRgb(frame, 1, 1, out), CcmfError);
}

TEST(RenderCellsToRgb, ZeroMaskIsAllBackgroundIncludingSubpixelFive) {
    Frame frame;
    frame.glyph = {0x80};  // mask 0: no bit set
    frame.fg = {1};
    frame.bg = {2};
    frame.palette = MakePalette();

    std::vector<std::uint8_t> out(RenderedPixelBufferSize(1, 1));
    RenderCellsToRgb(frame, 1, 1, out);

    const std::array<std::uint8_t, 3> bg = frame.palette.colors[2];
    for (int px = 0; px < 6; ++px) {
        EXPECT_EQ(out[px * 3 + 0], bg[0]) << "sub-pixel " << px;
        EXPECT_EQ(out[px * 3 + 1], bg[1]) << "sub-pixel " << px;
        EXPECT_EQ(out[px * 3 + 2], bg[2]) << "sub-pixel " << px;
    }
}

TEST(RenderCellsToRgb, FullMaskLeavesOnlyBottomRightAsBackground) {
    Frame frame;
    frame.glyph = {0x9F};  // mask 31: every controllable bit set
    frame.fg = {1};
    frame.bg = {2};
    frame.palette = MakePalette();

    std::vector<std::uint8_t> out(RenderedPixelBufferSize(1, 1));
    RenderCellsToRgb(frame, 1, 1, out);

    const std::array<std::uint8_t, 3> fg = frame.palette.colors[1];
    const std::array<std::uint8_t, 3> bg = frame.palette.colors[2];
    // Sub-pixels are laid out row-major over the cell's 3x2 block: indices
    // 0..4 are s=0..4 (fg-or-bg by mask bit), index 5 is s=5 (always bg).
    for (int s = 0; s < 5; ++s) {
        EXPECT_EQ(out[s * 3 + 0], fg[0]) << "sub-pixel " << s;
    }
    EXPECT_EQ(out[5 * 3 + 0], bg[0]) << "sub-pixel 5 (bottom-right) must stay background";
}

}  // namespace
}  // namespace ccmfplayer

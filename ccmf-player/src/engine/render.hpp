#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "engine/video.hpp"

namespace ccmfplayer {

// Byte size of the RGB pixel buffer RenderCellsToRgb produces for a
// width x height cell grid: (height*3) rows of (width*2) RGB pixels.
[[nodiscard]] constexpr std::size_t RenderedPixelBufferSize(std::size_t width,
                                                              std::size_t height) noexcept {
    return width * 2 * height * 3 * 3;  // (W*2) x (H*3) pixels, 3 bytes (RGB) each
}

// Renders one decoded frame to an RGB pixel buffer -- exactly what a CC
// monitor paints. A blit char (spec 4.5.1) is NOT a text glyph: its 5-bit
// index just selects, per sub-pixel of a 2x3 cell, foreground or background.
// Sub-pixel s (0..5) maps to pixel position (row = s/2, col = s%2) within the
// cell; s < 5 uses fg if bit s of (glyph - 0x80) is set, else bg; s == 5
// (bottom-right) is *always* bg (mirrors server/cc_media.py's render_cells).
//
// `out` must be exactly RenderedPixelBufferSize(width, height) bytes, laid
// out row-major with no padding: out[(row*width*2 + col) * 3 + channel] for
// row in [0, height*3), col in [0, width*2). `frame`'s glyph/fg/bg must each
// have width*height entries (true for anything DecodeVideoPayload produced
// for that width/height). Throws CcmfError if either precondition is
// violated.
void RenderCellsToRgb(const Frame& frame, std::uint16_t width, std::uint16_t height,
                      std::span<std::uint8_t> out);

}  // namespace ccmfplayer

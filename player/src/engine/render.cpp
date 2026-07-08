#include "engine/render.hpp"

namespace ccmfplayer {

void RenderCellsToRgb(const Frame& frame, std::uint16_t width, std::uint16_t height,
                      std::span<std::uint8_t> out) {
    const std::size_t n = static_cast<std::size_t>(width) * height;
    if (frame.glyph.size() != n || frame.fg.size() != n || frame.bg.size() != n) {
        throw CcmfError("frame grid size doesn't match width*height");
    }
    if (out.size() != RenderedPixelBufferSize(width, height)) {
        throw CcmfError("output buffer is the wrong size for this grid");
    }

    const std::size_t pixelWidth = static_cast<std::size_t>(width) * 2;

    for (std::uint16_t cellRow = 0; cellRow < height; ++cellRow) {
        for (std::uint16_t cellCol = 0; cellCol < width; ++cellCol) {
            const std::size_t cellIndex = static_cast<std::size_t>(cellRow) * width + cellCol;
            const auto mask =
                static_cast<std::uint8_t>(frame.glyph[cellIndex] - kBlitCharMin);
            const std::uint8_t fgIdx = frame.fg[cellIndex];
            const std::uint8_t bgIdx = frame.bg[cellIndex];

            for (int s = 0; s < 6; ++s) {
                // s==5 (bottom-right sub-pixel) is always background; s<5
                // picks fg/bg by that bit of the glyph's dither mask.
                const bool useFg = (s < 5) && ((mask >> s) & 1);
                const auto& colour =
                    useFg ? frame.palette.colors[fgIdx] : frame.palette.colors[bgIdx];

                const std::size_t subRow = static_cast<std::size_t>(s / 2);
                const std::size_t subCol = static_cast<std::size_t>(s % 2);
                const std::size_t pixelRow = static_cast<std::size_t>(cellRow) * 3 + subRow;
                const std::size_t pixelCol = static_cast<std::size_t>(cellCol) * 2 + subCol;
                const std::size_t base = (pixelRow * pixelWidth + pixelCol) * 3;

                out[base + 0] = colour[0];
                out[base + 1] = colour[1];
                out[base + 2] = colour[2];
            }
        }
    }
}

}  // namespace ccmfplayer

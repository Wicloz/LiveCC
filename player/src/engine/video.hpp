#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "engine/chunk.hpp"

namespace ccmfplayer {

// Frame-unit encodings (spec 4.5, flags bits 6-4).
inline constexpr std::uint8_t kFrameEncRaw = 0;
inline constexpr std::uint8_t kFrameEncDelta = 1;
inline constexpr std::uint8_t kFrameEncRepeat = 2;

// A blit char is 0x80 plus a 5-bit dither index (spec 4.5.1); valid values
// are exactly [kBlitCharMin, kBlitCharMax].
inline constexpr std::uint8_t kBlitCharMin = 0x80;
inline constexpr std::uint8_t kBlitCharMax = 0x9F;

// A 16-entry RGB palette (spec 4.5's palette unit): index 0-15 -> {r,g,b}.
struct Palette {
    std::array<std::array<std::uint8_t, 3>, 16> colors{};
};

// One decoded, fully-materialized frame: the complete glyph/fg/bg grids
// (row-major, width*height entries each) after applying its unit against the
// running GOP state, plus the palette in effect and how long to hold it
// (spec 4.5's `duration`, in 48 kHz samples).
struct Frame {
    std::uint16_t duration = 0;
    std::vector<std::uint8_t> glyph;  // blit chars: 0x80 + a 5-bit dither index
    std::vector<std::uint8_t> fg;     // palette indices 0-15
    std::vector<std::uint8_t> bg;     // palette indices 0-15
    Palette palette;
};

// A fully-decoded video chunk payload: its grid size and every frame in
// presentation order (spec 4.4: a video chunk is one self-contained GOP).
struct DecodedGop {
    std::uint16_t width = 0;
    std::uint16_t height = 0;
    std::vector<Frame> frames;
};

// Byte size of the packed raw planes (chars + fg + bg) for a width x height
// grid -- spec 4.5.1.
[[nodiscard]] constexpr std::size_t RawPlanesSize(std::size_t width, std::size_t height) noexcept {
    const std::size_t n = width * height;
    return ((n + 7) / 8) * 5 + ((n + 1) / 2) * 2;
}

// Unpacks `n` 5-bit blit-char indices (spec 4.5.1: 8 cells packed MSB-first
// into 5 bytes) into full blit chars (0x80 + index). `data` must hold at
// least ((n+7)/8)*5 bytes; throws CcmfError otherwise.
[[nodiscard]] std::vector<std::uint8_t> UnpackChars(std::span<const std::byte> data,
                                                      std::size_t n);

// Unpacks `n` 4-bit palette indices (spec 4.5.1: 2 cells/byte, high nibble
// first). `data` must hold at least (n+1)/2 bytes; throws CcmfError otherwise.
[[nodiscard]] std::vector<std::uint8_t> UnpackNibbles(std::span<const std::byte> data,
                                                        std::size_t n);

// Decodes a full video chunk payload (spec 4.4/4.5): the width/height header
// followed by a palette unit, a raw keyframe, and any number of further
// palette/raw/delta/repeat units. Throws CcmfError on any structural
// violation (truncation, a delta/repeat before the first keyframe, a delta
// span crossing a row boundary, an out-of-range blit char, or a chunk with
// no frames) -- spec 7 requires bound-checking before drawing.
[[nodiscard]] DecodedGop DecodeVideoPayload(std::span<const std::byte> payload);

}  // namespace ccmfplayer

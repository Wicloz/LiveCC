#include "engine/video.hpp"

#include <string>

#include "engine/byteio.hpp"

namespace ccmfplayer {

namespace {

void CheckAvailable(std::span<const std::byte> payload, std::size_t pos, std::size_t need,
                     const char* what) {
    if (pos + need > payload.size()) {
        throw CcmfError(std::string("truncated video payload: ") + what);
    }
}

Palette ReadPalette(std::span<const std::byte> payload, std::size_t pos) {
    Palette palette;
    for (std::size_t entry = 0; entry < 16; ++entry) {
        for (std::size_t ch = 0; ch < 3; ++ch) {
            palette.colors[entry][ch] =
                std::to_integer<std::uint8_t>(payload[pos + entry * 3 + ch]);
        }
    }
    return palette;
}

}  // namespace

std::vector<std::uint8_t> UnpackChars(std::span<const std::byte> data, std::size_t n) {
    const std::size_t numGroups = (n + 7) / 8;
    if (data.size() < numGroups * 5) {
        throw CcmfError("truncated chars plane");
    }
    std::vector<std::uint8_t> out;
    out.reserve(n);
    for (std::size_t g = 0; g < numGroups && out.size() < n; ++g) {
        std::uint64_t val = 0;
        for (std::size_t b = 0; b < 5; ++b) {
            val = (val << 8) | std::to_integer<std::uint8_t>(data[g * 5 + b]);
        }
        for (int j = 0; j < 8 && out.size() < n; ++j) {
            const int shift = 5 * (7 - j);
            const auto code = static_cast<std::uint8_t>((val >> shift) & 0x1F);
            out.push_back(static_cast<std::uint8_t>(kBlitCharMin + code));
        }
    }
    return out;
}

std::vector<std::uint8_t> UnpackNibbles(std::span<const std::byte> data, std::size_t n) {
    const std::size_t numBytes = (n + 1) / 2;
    if (data.size() < numBytes) {
        throw CcmfError("truncated nibble plane");
    }
    std::vector<std::uint8_t> out;
    out.reserve(n);
    for (std::size_t i = 0; i < numBytes && out.size() < n; ++i) {
        const auto byteVal = std::to_integer<std::uint8_t>(data[i]);
        out.push_back(static_cast<std::uint8_t>(byteVal >> 4));
        if (out.size() < n) {
            out.push_back(static_cast<std::uint8_t>(byteVal & 0x0F));
        }
    }
    return out;
}

DecodedGop DecodeVideoPayload(std::span<const std::byte> payload) {
    if (payload.size() < 4) {
        throw CcmfError("truncated video payload header");
    }
    const auto width = static_cast<std::uint16_t>(ReadLittleEndian<2>(payload.subspan(0, 2)));
    const auto height = static_cast<std::uint16_t>(ReadLittleEndian<2>(payload.subspan(2, 2)));
    const std::size_t n = static_cast<std::size_t>(width) * height;

    std::size_t pos = 4;
    bool havePalette = false;
    Palette palette;
    std::vector<std::uint8_t> glyph;  // running GOP state, mutated by delta units
    std::vector<std::uint8_t> fg;
    std::vector<std::uint8_t> bg;

    DecodedGop gop;
    gop.width = width;
    gop.height = height;

    while (pos < payload.size()) {
        const auto flags = std::to_integer<std::uint8_t>(payload[pos]);
        ++pos;

        if ((flags & 0x80) == 0) {  // palette unit
            CheckAvailable(payload, pos, 48, "palette unit");
            palette = ReadPalette(payload, pos);
            havePalette = true;
            pos += 48;
            continue;
        }

        const std::uint8_t enc = (flags >> 4) & 0x07;
        CheckAvailable(payload, pos, 2, "frame duration");
        const auto duration =
            static_cast<std::uint16_t>(ReadLittleEndian<2>(payload.subspan(pos, 2)));
        pos += 2;

        if (enc == kFrameEncRaw) {
            const std::size_t nc = ((n + 7) / 8) * 5;
            const std::size_t nn = (n + 1) / 2;
            CheckAvailable(payload, pos, nc + nn + nn, "raw frame planes");
            glyph = UnpackChars(payload.subspan(pos, nc), n);
            pos += nc;
            fg = UnpackNibbles(payload.subspan(pos, nn), n);
            pos += nn;
            bg = UnpackNibbles(payload.subspan(pos, nn), n);
            pos += nn;
        } else if (enc == kFrameEncDelta) {
            if (glyph.empty()) {
                throw CcmfError("delta frame before any raw keyframe");
            }
            CheckAvailable(payload, pos, 2, "delta span count");
            const auto count =
                static_cast<std::uint16_t>(ReadLittleEndian<2>(payload.subspan(pos, 2)));
            pos += 2;
            for (std::uint16_t s = 0; s < count; ++s) {
                CheckAvailable(payload, pos, 3, "delta span header");
                const auto start =
                    static_cast<std::uint16_t>(ReadLittleEndian<2>(payload.subspan(pos, 2)));
                const auto length = std::to_integer<std::uint8_t>(payload[pos + 2]);
                pos += 3;
                if (width == 0 || start % width + length > width
                    || static_cast<std::size_t>(start) + length > n) {
                    throw CcmfError("delta span crosses a row/grid boundary");
                }
                CheckAvailable(payload, pos, static_cast<std::size_t>(length) * 2, "delta cells");
                for (std::uint8_t i = 0; i < length; ++i) {
                    const auto cellChar = std::to_integer<std::uint8_t>(payload[pos + i * 2]);
                    const auto colour = std::to_integer<std::uint8_t>(payload[pos + i * 2 + 1]);
                    if (cellChar < kBlitCharMin || cellChar > kBlitCharMax) {
                        throw CcmfError("delta span cell has an out-of-range blit char");
                    }
                    glyph[start + i] = cellChar;
                    fg[start + i] = static_cast<std::uint8_t>(colour & 0x0F);
                    bg[start + i] = static_cast<std::uint8_t>(colour >> 4);
                }
                pos += static_cast<std::size_t>(length) * 2;
            }
        } else if (enc == kFrameEncRepeat) {
            if (glyph.empty()) {
                throw CcmfError("repeat frame before any raw keyframe");
            }
        } else {
            throw CcmfError("unknown frame encoding " + std::to_string(static_cast<int>(enc)));
        }

        if (!havePalette) {
            throw CcmfError("frame before any palette unit");
        }
        gop.frames.push_back(Frame{duration, glyph, fg, bg, palette});
    }

    if (gop.frames.empty()) {
        throw CcmfError("video chunk carries no frame");
    }
    return gop;
}

}  // namespace ccmfplayer

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

// rANS + RLE plane decoder (spec 4.5.3) -- mirrors server/rans.py and the Lua
// client exactly (byte-renormalized rANS, M = 2^12; a linear scan over the
// frequency-descending cumulative table; run lengths as byte tokens).  Writes
// `n` symbols into `out` and returns the offset just past the plane blob.
constexpr std::uint32_t kRansL = 1u << 16;
constexpr std::uint32_t kRansM = 1u << 12;

std::size_t DecodeAnsPlane(std::span<const std::byte> payload, std::size_t pos,
                           std::size_t n, std::uint8_t nsym, std::size_t width,
                           std::vector<std::uint8_t>& out) {
    auto at = [&](std::size_t k) -> std::uint8_t {
        if (k >= payload.size()) throw CcmfError("truncated ANS plane");
        return std::to_integer<std::uint8_t>(payload[k]);
    };
    // A leading `filter` byte selects an optional spatial predictor (0 none,
    // 1 sub/left, 2 up), reversed after the entropy decode.  mode 0 = RLE+rANS
    // (run values + length tokens); mode 1 = plain rANS (one symbol per cell) --
    // whichever the encoder found smaller.
    const std::uint8_t filter = at(pos);
    ++pos;
    const std::uint8_t mode = at(pos);
    ++pos;
    const std::size_t k = at(pos);
    ++pos;
    if (k == 0) throw CcmfError("ANS plane has no symbols");
    std::vector<std::uint8_t> syms(k);
    std::vector<std::uint32_t> freqs(k), cums(k);
    std::uint32_t acc = 0;
    for (std::size_t i = 0; i < k; ++i) {
        syms[i] = at(pos);
        freqs[i] = static_cast<std::uint32_t>(at(pos + 1)) | (static_cast<std::uint32_t>(at(pos + 2)) << 8);
        cums[i] = acc;
        acc += freqs[i];
        pos += 3;
    }
    const std::size_t ransLen =
        static_cast<std::size_t>(at(pos)) | (static_cast<std::size_t>(at(pos + 1)) << 8)
        | (static_cast<std::size_t>(at(pos + 2)) << 16);
    pos += 3;
    std::size_t rp = pos;             // rANS byte cursor
    const std::size_t lpEnd = pos + ransLen;
    std::size_t lp = lpEnd;           // length-token cursor (mode 0)

    std::uint32_t x = (static_cast<std::uint32_t>(at(rp)) << 24)
                    | (static_cast<std::uint32_t>(at(rp + 1)) << 16)
                    | (static_cast<std::uint32_t>(at(rp + 2)) << 8)
                    | static_cast<std::uint32_t>(at(rp + 3));
    rp += 4;

    out.resize(n);
    auto nextSymbol = [&]() -> std::size_t {
        const std::uint32_t slot = x & (kRansM - 1);
        std::size_t i = 0;
        while (i + 1 < k && cums[i + 1] <= slot) ++i;
        x = freqs[i] * (x >> 12) + slot - cums[i];
        while (x < kRansL) x = (x << 8) | at(rp++);
        return i;
    };

    std::size_t endpos;
    if (mode == 1) {                  // plain rANS: one symbol per cell
        for (std::size_t cell = 0; cell < n; ++cell) out[cell] = syms[nextSymbol()];
        endpos = lpEnd;               // rANS stream end
    } else {
        std::size_t cells = 0;        // mode 0: RLE + rANS run values
        while (cells < n) {
            const std::size_t i = nextSymbol();
            const std::uint8_t b = at(lp++);
            std::size_t length = b;
            if (b == 255) {
                length = static_cast<std::size_t>(at(lp)) | (static_cast<std::size_t>(at(lp + 1)) << 8);
                lp += 2;
            }
            if (cells + length > n) throw CcmfError("ANS plane run overruns the grid");
            for (std::size_t c = 0; c < length; ++c) out[cells++] = syms[i];
        }
        endpos = lp;
    }

    // Reverse the spatial predictor: a MATCH token (value nsym) copies its
    // already-reconstructed left (sub) or upper (up) neighbour.  Row-start cells
    // are never MATCH tokens, so a plain forward scan needs no boundary test.
    if (filter == 1) {                                    // sub (left)
        for (std::size_t idx = 1; idx < n; ++idx)
            if (out[idx] == nsym) out[idx] = out[idx - 1];
    } else if (filter == 2 && width > 0) {                // up
        for (std::size_t idx = width; idx < n; ++idx)
            if (out[idx] == nsym) out[idx] = out[idx - width];
    }
    return endpos;
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
        } else if (enc == kFrameEncRawAns) {
            CheckAvailable(payload, pos, 3, "ANS frame body length");
            const std::size_t bodyLen =
                static_cast<std::size_t>(ReadLittleEndian<3>(payload.subspan(pos, 3)));
            pos += 3;
            const std::size_t bodyEnd = pos + bodyLen;
            if (bodyEnd > payload.size()) {
                throw CcmfError("truncated ANS frame body");
            }
            std::vector<std::uint8_t> gcode;
            pos = DecodeAnsPlane(payload, pos, n, 32, width, gcode);
            pos = DecodeAnsPlane(payload, pos, n, 16, width, fg);
            pos = DecodeAnsPlane(payload, pos, n, 16, width, bg);
            if (pos != bodyEnd) {
                throw CcmfError("ANS frame body length mismatch");
            }
            glyph.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                if (gcode[i] > 0x1F) {
                    throw CcmfError("ANS glyph index out of range");
                }
                glyph[i] = static_cast<std::uint8_t>(kBlitCharMin + gcode[i]);
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

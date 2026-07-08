#include "engine/chunk.hpp"

#include "engine/byteio.hpp"

namespace ccmfplayer {

ChunkHeader ParseChunkHeader(std::span<const std::byte> buf) {
    if (buf.size() < kChunkHeaderSize) {
        throw CcmfError("truncated chunk header");
    }
    if (std::to_integer<std::uint8_t>(buf[0]) != kChunkMarker) {
        throw CcmfError("bad chunk marker byte");
    }

    ChunkHeader header;
    header.pts = ReadLittleEndian<6>(buf.subspan(1, 6));
    header.length = static_cast<std::uint32_t>(ReadLittleEndian<3>(buf.subspan(7, 3)));
    header.type = std::to_integer<std::uint8_t>(buf[10]);
    header.compression = std::to_integer<std::uint8_t>(buf[11]);

    // `compression` is recorded but NOT validated here: an unknown/undecodable
    // algorithm must still let indexing proceed (spec 4.1's skip-what-you-can't-
    // decode). DecompressPayload() rejects the ones this player can't inflate.
    return header;
}

namespace {

// LZ4 block-format decompressor (spec 4.1.2): byte-oriented sequences of a
// token, a literal run, and a back-reference match. `outSize` is the known
// decompressed length (from our wrapper), so the loop can bound-check every
// write and never reallocate past it.
std::vector<std::byte> Lz4DecompressBlock(std::span<const std::byte> in,
                                          std::size_t outSize) {
    std::vector<std::byte> out;
    out.reserve(outSize);
    const std::size_t n = in.size();
    std::size_t i = 0;
    auto at = [&](std::size_t k) { return std::to_integer<std::uint8_t>(in[k]); };

    while (i < n) {
        const std::uint8_t token = at(i++);
        std::size_t lit = token >> 4;                         // literal length
        if (lit == 15) {
            std::uint8_t b;
            do {
                if (i >= n) throw CcmfError("LZ4: truncated literal length");
                b = at(i++);
                lit += b;
            } while (b == 255);
        }
        if (i + lit > n) throw CcmfError("LZ4: literal run overruns input");
        if (out.size() + lit > outSize) throw CcmfError("LZ4: output overflow");
        for (std::size_t k = 0; k < lit; ++k) out.push_back(in[i + k]);
        i += lit;
        if (i == n) break;                                    // last: literals only

        if (i + 2 > n) throw CcmfError("LZ4: truncated match offset");
        const std::size_t offset = at(i) | (static_cast<std::size_t>(at(i + 1)) << 8);
        i += 2;
        if (offset == 0 || offset > out.size()) throw CcmfError("LZ4: bad match offset");
        std::size_t mlen = static_cast<std::size_t>(token & 0x0F);   // + MINMATCH
        if (mlen == 15) {
            std::uint8_t b;
            do {
                if (i >= n) throw CcmfError("LZ4: truncated match length");
                b = at(i++);
                mlen += b;
            } while (b == 255);
        }
        mlen += 4;
        if (out.size() + mlen > outSize) throw CcmfError("LZ4: output overflow");
        const std::size_t src = out.size() - offset;
        for (std::size_t k = 0; k < mlen; ++k) out.push_back(out[src + k]);  // overlap-safe
    }
    if (out.size() != outSize) throw CcmfError("LZ4: decompressed size mismatch");
    return out;
}

}  // namespace

std::vector<std::byte> DecompressPayload(std::span<const std::byte> payload,
                                         std::uint8_t compression) {
    if (compression == kCompressionNone) {
        return std::vector<std::byte>(payload.begin(), payload.end());
    }
    if (compression == kCompressionLz4) {
        if (payload.size() < 4) throw CcmfError("LZ4: truncated payload");
        const auto outSize =
            static_cast<std::size_t>(ReadLittleEndian<4>(payload.subspan(0, 4)));
        return Lz4DecompressBlock(payload.subspan(4), outSize);
    }
    throw CcmfError("unsupported compression " + std::to_string(compression));
}

}  // namespace ccmfplayer

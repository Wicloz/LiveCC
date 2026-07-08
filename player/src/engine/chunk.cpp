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

    // Compression is a per-chunk field covering every future type (spec
    // 4.1.2), but only "none" is defined/decodable today -- mirrors
    // server/ccmf.py's parse_chunk, which rejects anything else the same way.
    if (header.compression != 0) {
        throw CcmfError("unsupported compression " + std::to_string(header.compression));
    }

    return header;
}

}  // namespace ccmfplayer

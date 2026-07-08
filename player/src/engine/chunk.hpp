#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ccmfplayer {

// PTS unit (spec 4.2): 48 kHz samples, shared by every chunk type and by the
// playback clock/UI timecode formatting built on top of it.
inline constexpr std::uint32_t kSampleRate = 48000;

// Container chunk marker byte (spec 4.1): fixed resync byte, ASCII 'C'.
inline constexpr std::uint8_t kChunkMarker = 0x43;

// Fixed chunk header size: marker(1) + PTS(6) + length(3) + type(1) +
// compression(1), spec 4.1.
inline constexpr std::size_t kChunkHeaderSize = 12;

// Chunk `type` values (spec 4.3). Not an enum class: the wire byte can be
// any reserved/future value too, and a decoder MUST skip those rather than
// reject them (spec 4.1), so callers compare the raw byte against these
// constants instead of switching over a closed set.
inline constexpr std::uint8_t kChunkTypeVideo = 0;
inline constexpr std::uint8_t kChunkTypeAudio = 1;

// Compression values (spec 4.1.2). Only `none` and `lz4` are decodable here;
// `deflate`/`zstd` are reserved for other native clients and rejected on decode.
inline constexpr std::uint8_t kCompressionNone = 0;
inline constexpr std::uint8_t kCompressionLz4 = 2;

// Thrown for anything that makes a chunk (or a whole file) unreadable: a bad
// marker, a truncated header/payload, unsupported compression, or a
// structural invariant the rest of the engine depends on (e.g. ascending
// PTS) being violated. Spec 7: decoders MUST bound-check before trusting a
// chunk's declared length.
class CcmfError : public std::runtime_error {
public:
    explicit CcmfError(const std::string& message) : std::runtime_error(message) {}
};

// One parsed chunk header (spec 4.1).
struct ChunkHeader {
    std::uint64_t pts = 0;       // absolute PTS in 48 kHz samples (48-bit range)
    std::uint32_t length = 0;    // on-wire payload length in bytes (24-bit range)
    std::uint8_t type = 0;
    std::uint8_t compression = 0;
};

// Parses exactly the first kChunkHeaderSize bytes of `buf` as a chunk header.
// `buf` may be longer (only the header is read); it must be at least
// kChunkHeaderSize bytes. This function knows nothing about a payload's
// actual availability in a file -- it only decodes the 12 header bytes and
// validates the marker and (for now) that compression is "none". Throws
// CcmfError on a truncated buffer, a bad marker, or unsupported compression.
[[nodiscard]] ChunkHeader ParseChunkHeader(std::span<const std::byte> buf);

// Decompresses a chunk payload per its `compression` byte (spec 4.1.2). `none`
// is a straight copy; `lz4` expects [uncompressed size u32 LE][raw LZ4 block]
// and inflates it. Throws CcmfError on an unsupported algorithm or malformed
// data. Callers then interpret the result per the chunk `type`.
[[nodiscard]] std::vector<std::byte> DecompressPayload(
    std::span<const std::byte> payload, std::uint8_t compression);

}  // namespace ccmfplayer

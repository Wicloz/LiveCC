#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <optional>
#include <vector>

#include "engine/chunk.hpp"

namespace ccmfplayer {

// Low-level self-synchronization primitives (spec 4.1.1). CCMF is a
// self-describing byte stream: a fixed 1-byte marker (kChunkMarker) leads every
// chunk, and `length` lets a reader jump to the next one. These functions let a
// reader land at an arbitrary offset, find the next real chunk boundary, and
// *validate* it by chaining (a false marker fails to chain forward). They know
// nothing about the sparse index built on top of them (index.hpp) -- they only
// read a seekable byte stream and report what they find, so they're trivially
// testable against hand-built byte buffers.

// Result of chaining chunk-to-chunk from a candidate offset.
struct ChainResult {
    std::size_t links = 0;    // count of valid chunks successfully chained
    bool reachedEof = false;  // the chain ended exactly at end-of-file
};

// Reads the 12-byte header at `offset` and validates it as a chunk boundary:
// marker == kChunkMarker AND offset + kChunkHeaderSize + length <= fileSize
// (a boundary whose payload would overrun the file is not a boundary). On
// success fills `out` and returns true. Never throws; a short read, bad marker,
// or overrunning length all just return false.
[[nodiscard]] bool TryReadHeaderAt(std::istream& in, std::uint64_t offset,
                                   std::uint64_t fileSize, ChunkEntry& out);

// Follows the chunk chain from `start`: read a header, append it to `out`, jump
// to its End(), repeat. Stops when the next position is EOF (reachedEof = true),
// a header fails to validate (desync), or `maxLinks` chunks have been collected.
// `out` receives every chunk successfully read along the way. This is the core
// correctness primitive: a real boundary chains cleanly, a false marker desyncs
// within about one link (P(random byte == marker) = 1/256, spec 4.1.1).
ChainResult ChainFrom(std::istream& in, std::uint64_t start, std::uint64_t fileSize,
                      std::size_t maxLinks, std::vector<ChunkEntry>& out);

// Scans bytes forward in [from, limit) for the first candidate marker that
// begins a chain of at least `requiredLinks` links OR one that reaches EOF.
// Returns that chunk's offset and fills `collected` with the chain it began;
// std::nullopt if no such boundary exists before `limit`. Used by seek homing
// to resync at an interpolated byte guess.
[[nodiscard]] std::optional<std::uint64_t> Resync(std::istream& in, std::uint64_t from,
                                                  std::uint64_t limit, std::uint64_t fileSize,
                                                  std::size_t requiredLinks,
                                                  std::vector<ChunkEntry>& collected);

// Reads backwards from EOF to recover the file's trailing chunk chain (spec
// 4.1.1: "read backwards ... to determine its duration"). Grows a tail window
// from the end until it finds the marker offset that begins the chain reaching
// exactly EOF with the most links (the earliest true boundary in the window),
// and returns that suffix of chunks -- the last element is the duration anchor.
// std::nullopt for an empty or unparseable-from-the-tail file. Chunks may exceed
// the window: ChainFrom reads headers via file seeks, not from a window buffer.
[[nodiscard]] std::optional<std::vector<ChunkEntry>> FindTailChunks(std::istream& in,
                                                                    std::uint64_t fileSize);

}  // namespace ccmfplayer

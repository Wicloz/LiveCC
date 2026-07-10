#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <vector>

#include "engine/chunk.hpp"

namespace ccmfplayer {

// Opens a .ccmf file and provides a *lazy, self-synchronizing* view of its
// chunk sequence. Unlike a naive reader, it never scans the whole file up
// front (spec 4.1.1 designs CCMF to be seekable like MPEG-TS: a marker byte
// plus chainable `length` fields). Instead it builds a **sparse index** of
// chunks discovered on demand -- when reading backwards for the duration, when
// walking forward during playback, and when homing in on a seek target -- and
// records every chunk it ever identifies so the work is never repeated. Opening
// even a multi-gigabyte file is therefore O(tail), not O(file).
//
// The index is offset-keyed and always consistent: a chunk is recorded exactly
// once, and every discovery path funnels through Record(). Callers resolve a
// presentation timestamp to the chunk(s) active there via RegionAround(), which
// guarantees the relevant contiguous run of chunks has been discovered.
//
// Not copyable (owns a file handle); movable.
class CcmfFile {
public:
    // Opens `path`. Cheap: stats the size and reads nothing else (the index is
    // populated lazily). Throws CcmfError only if the file can't be opened.
    explicit CcmfFile(const std::filesystem::path& path);

    CcmfFile(const CcmfFile&) = delete;
    CcmfFile& operator=(const CcmfFile&) = delete;
    CcmfFile(CcmfFile&&) = default;
    CcmfFile& operator=(CcmfFile&&) = default;

    [[nodiscard]] std::uint64_t FileSize() const noexcept { return fileSize_; }

    // The chunk at offset 0 (a stored file is a bare sequence of chunks, so it
    // begins with one -- spec 1). std::nullopt for an empty file or one whose
    // first bytes aren't a valid header. Records what it finds.
    [[nodiscard]] std::optional<ChunkEntry> FirstChunk();

    // The chain-adjacent chunk immediately after `entry` (at entry.End()), or
    // std::nullopt at end-of-file / on a malformed boundary. The cheap forward
    // step used by normal playback; consults the index before touching disk.
    [[nodiscard]] std::optional<ChunkEntry> NextChunkAfter(const ChunkEntry& entry);

    // The file's trailing chunk chain, recovered by reading backwards from EOF
    // (spec 4.1.1). Computed once and cached; every chunk is recorded. Empty
    // only for a file with no recoverable tail chunk.
    [[nodiscard]] const std::vector<ChunkEntry>& TailChunks();

    // Ensures the index contains the contiguous run of chunks needed to resolve
    // any track (video GOP or audio role) active at `pts`, and returns that run
    // in ascending file order. The run starts far enough before `pts` to include
    // a long GOP or audio chunk already in progress, and ends at the first chunk
    // whose pts exceeds `pts` (or EOF). This is the seek homer: it interpolates
    // a byte guess from the sparse index, resyncs there, and narrows until the
    // target region is pinned -- all discovered chunks are recorded.
    [[nodiscard]] std::vector<ChunkEntry> RegionAround(std::uint64_t pts);

    // A snapshot of every chunk currently in the sparse index, ascending by
    // offset. Reflects only what has been discovered so far (for diagnostics
    // and tests that assert the index grew opportunistically rather than all at
    // once).
    [[nodiscard]] std::vector<ChunkEntry> KnownChunks() const;

    // Whether any video (type 0) chunk has been discovered yet. True after the
    // engine's construction-time bootstrap for any file that has video.
    [[nodiscard]] bool KnowsAnyVideo() const noexcept { return knowsVideo_; }

    // A full, eager linear scan from offset 0 (the pre-lazy behaviour), returned
    // partitioned by type. Provided ONLY as a reference/verification path (tests
    // cross-check against it, and it's an escape hatch); the playback path never
    // calls it. Throws CcmfError on a truncated/malformed chunk sequence.
    struct FullIndex {
        std::vector<ChunkEntry> chunks;  // every chunk in file order
        std::vector<ChunkEntry> video;   // type 0 only
        std::vector<ChunkEntry> audio;   // type 1 only
    };
    [[nodiscard]] FullIndex IndexAll();

    // Reads and returns the raw payload bytes for one chunk entry, inflating any
    // compression (spec 4.1.2) so callers get the decompressed payload.
    [[nodiscard]] std::vector<std::byte> ReadChunkPayload(const ChunkEntry& entry) const;

    // Reads `length` bytes of a chunk's *on-wire* (possibly compressed) payload
    // starting `payloadOffset` bytes in, without reading the rest -- for peeking
    // at a small header. Throws CcmfError if the range exceeds the chunk's
    // declared length. ReadChunkPayload(entry) is ReadChunkPayloadRange(entry, 0,
    // entry.length) followed by decompression.
    [[nodiscard]] std::vector<std::byte> ReadChunkPayloadRange(const ChunkEntry& entry,
                                                                std::uint64_t payloadOffset,
                                                                std::uint64_t length) const;

private:
    // Inserts `entry` into the sparse index (idempotent, keyed by offset) and
    // returns the stored copy. Every discovery path goes through here.
    const ChunkEntry& Record(const ChunkEntry& entry);

    // The header at `offset`, from the index if already known, else read (and
    // recorded) from disk. std::nullopt on a malformed/out-of-range header.
    [[nodiscard]] std::optional<ChunkEntry> ReadHeaderAt(std::uint64_t offset);

    // Records FirstChunk() and TailChunks() so the index brackets any pts in
    // [0, duration]; sets knowsVideo_ if a video chunk turns up. Idempotent.
    void EnsureBootstrap();

    // Offset of a chunk boundary whose pts is <= `target`, homed in as close to
    // `target` as the sparse index and resync allow. Bootstrap must have run.
    [[nodiscard]] std::uint64_t HomeToPtsAtMost(std::uint64_t target);

    std::filesystem::path path_;
    mutable std::ifstream file_;
    std::uint64_t fileSize_ = 0;

    std::map<std::uint64_t, ChunkEntry> known_;  // offset -> chunk; the sparse index
    std::optional<std::vector<ChunkEntry>> tailChunks_;
    bool bootstrapped_ = false;
    bool knowsVideo_ = false;
};

}  // namespace ccmfplayer

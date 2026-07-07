#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

#include "engine/chunk.hpp"

namespace ccmfplayer {

// One entry in a file's chunk index: everything needed to seek to and read a
// chunk's payload without re-scanning the file.
struct ChunkEntry {
    std::uint64_t offset = 0;    // byte offset of the chunk's marker byte
    std::uint64_t pts = 0;
    std::uint32_t length = 0;    // payload length in bytes (excludes the header)
    std::uint8_t type = 0;
    std::uint8_t compression = 0;
};

// Opens a .ccmf file and indexes its chunk sequence (spec 4.1): one
// header-only scan from offset 0 -- a stored file "is a bare sequence of
// container chunks" (spec 1) read by "chaining" from the start (spec
// 4.1.1) -- giving fast seeking later without ever decoding a payload up
// front. Chunks are additionally partitioned by type into VideoChunks()/
// AudioChunks() views, each required to be non-decreasing in PTS (a
// structural assumption the playback engine's seek binary search depends
// on); a file that violates it is rejected at load time rather than
// producing silently-wrong seeks later.
//
// Not copyable (owns a file handle); movable.
class CcmfFile {
public:
    // Opens `path` and builds the index. Throws CcmfError if the file can't
    // be opened, or its chunk sequence is malformed (spec 7).
    explicit CcmfFile(const std::filesystem::path& path);

    CcmfFile(const CcmfFile&) = delete;
    CcmfFile& operator=(const CcmfFile&) = delete;
    CcmfFile(CcmfFile&&) = default;
    CcmfFile& operator=(CcmfFile&&) = default;

    // Every chunk in file order, including types this player doesn't
    // otherwise interpret (e.g. a reserved/subtitle type) -- present so
    // skipped chunks are still visible to callers that want to know.
    [[nodiscard]] std::span<const ChunkEntry> Chunks() const noexcept { return chunks_; }
    // Video (type 0) chunks only, in ascending-PTS file order.
    [[nodiscard]] std::span<const ChunkEntry> VideoChunks() const noexcept { return videoChunks_; }
    // Audio (type 1) chunks only, in ascending-PTS file order.
    [[nodiscard]] std::span<const ChunkEntry> AudioChunks() const noexcept { return audioChunks_; }

    // Reads and returns the raw payload bytes for one chunk entry (seeks the
    // underlying file). `entry` must be one this CcmfFile produced (from
    // Chunks()/VideoChunks()/AudioChunks()); behavior is undefined otherwise.
    // Logically const (doesn't change the index), even though it physically
    // moves the file's read position -- hence the mutable stream member.
    [[nodiscard]] std::vector<std::byte> ReadChunkPayload(const ChunkEntry& entry) const;

    // Reads just `length` bytes of a chunk's payload starting `payloadOffset`
    // bytes into it, without reading the rest -- for peeking at a small
    // header (e.g. an audio chunk's a-hdr byte, spec 4.6) without paying to
    // read the whole payload. Throws CcmfError if `payloadOffset + length`
    // exceeds the chunk's declared length. ReadChunkPayload(entry) is
    // exactly ReadChunkPayloadRange(entry, 0, entry.length).
    [[nodiscard]] std::vector<std::byte> ReadChunkPayloadRange(const ChunkEntry& entry,
                                                                std::uint64_t payloadOffset,
                                                                std::uint64_t length) const;

private:
    void BuildIndex();

    std::filesystem::path path_;
    mutable std::ifstream file_;
    std::vector<ChunkEntry> chunks_;
    std::vector<ChunkEntry> videoChunks_;
    std::vector<ChunkEntry> audioChunks_;
};

}  // namespace ccmfplayer

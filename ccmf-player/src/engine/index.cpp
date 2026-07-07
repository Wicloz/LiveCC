#include "engine/index.hpp"

#include <array>

namespace ccmfplayer {

CcmfFile::CcmfFile(const std::filesystem::path& path)
    : path_(path), file_(path, std::ios::binary) {
    if (!file_) {
        throw CcmfError("failed to open file: " + path.string());
    }
    BuildIndex();
}

void CcmfFile::BuildIndex() {
    file_.seekg(0, std::ios::end);
    const std::streamoff fileSizeSigned = file_.tellg();
    if (fileSizeSigned < 0) {
        throw CcmfError("failed to determine file size: " + path_.string());
    }
    const auto fileSize = static_cast<std::uint64_t>(fileSizeSigned);

    std::array<std::byte, kChunkHeaderSize> headerBuf{};
    std::uint64_t offset = 0;
    while (offset < fileSize) {
        if (offset + kChunkHeaderSize > fileSize) {
            throw CcmfError("truncated chunk header at offset " + std::to_string(offset));
        }
        file_.seekg(static_cast<std::streamoff>(offset));
        file_.read(reinterpret_cast<char*>(headerBuf.data()),
                   static_cast<std::streamsize>(headerBuf.size()));
        if (!file_) {
            throw CcmfError("failed to read chunk header at offset " + std::to_string(offset));
        }

        const ChunkHeader header = ParseChunkHeader(headerBuf);
        if (offset + kChunkHeaderSize + header.length > fileSize) {
            throw CcmfError("truncated chunk payload at offset " + std::to_string(offset));
        }

        const ChunkEntry entry{offset, header.pts, header.length, header.type,
                                header.compression};
        chunks_.push_back(entry);

        if (header.type == kChunkTypeVideo) {
            if (!videoChunks_.empty() && entry.pts < videoChunks_.back().pts) {
                throw CcmfError("non-monotonic PTS in video chunk sequence at offset "
                                 + std::to_string(offset));
            }
            videoChunks_.push_back(entry);
        } else if (header.type == kChunkTypeAudio) {
            if (!audioChunks_.empty() && entry.pts < audioChunks_.back().pts) {
                throw CcmfError("non-monotonic PTS in audio chunk sequence at offset "
                                 + std::to_string(offset));
            }
            audioChunks_.push_back(entry);
        }
        // Any other type (reserved, or a future subtitle chunk) is recorded
        // in chunks_ above but otherwise skipped -- spec 4.1: "a decoder
        // encountering an unknown type MUST skip length bytes and continue."

        offset += kChunkHeaderSize + header.length;
    }
}

std::vector<std::byte> CcmfFile::ReadChunkPayloadRange(const ChunkEntry& entry,
                                                        std::uint64_t payloadOffset,
                                                        std::uint64_t length) const {
    if (payloadOffset + length > entry.length) {
        throw CcmfError("payload range out of bounds for chunk at offset "
                         + std::to_string(entry.offset));
    }
    std::vector<std::byte> data(length);
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(entry.offset + kChunkHeaderSize + payloadOffset));
    file_.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(length));
    if (!file_ || static_cast<std::uint64_t>(file_.gcount()) != length) {
        throw CcmfError("failed to read chunk payload range at offset "
                         + std::to_string(entry.offset)
                         + " (file changed since indexing?)");
    }
    return data;
}

std::vector<std::byte> CcmfFile::ReadChunkPayload(const ChunkEntry& entry) const {
    return ReadChunkPayloadRange(entry, 0, entry.length);
}

}  // namespace ccmfplayer

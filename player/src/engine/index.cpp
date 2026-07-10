#include "engine/index.hpp"

#include <algorithm>
#include <string>

#include "engine/resync.hpp"

namespace ccmfplayer {

namespace {

// How far (in 48 kHz samples, spec 4.2) before a target PTS a resolved region
// must begin, so a video GOP or audio chunk already in progress at the target
// is captured. 5 s comfortably exceeds any single chunk's span (audio chunks
// are ~2 s; video GOPs a few seconds). RegionAround() doubles it if an even
// longer GOP turns out to have started earlier still.
constexpr std::uint64_t kRegionGuardSamples = 5 * kSampleRate;

// Links a candidate must chain to be trusted during seek homing (spec 4.1.1).
constexpr std::size_t kHomingRequiredLinks = 2;

// Safety caps so a pathological file can't spin the homing / region loops.
constexpr int kMaxHomingIterations = 256;
constexpr int kMaxRegionGrows = 32;

}  // namespace

CcmfFile::CcmfFile(const std::filesystem::path& path)
    : path_(path), file_(path, std::ios::binary) {
    if (!file_) {
        throw CcmfError("failed to open file: " + path.string());
    }
    file_.seekg(0, std::ios::end);
    const std::streamoff sizeSigned = file_.tellg();
    if (sizeSigned < 0) {
        throw CcmfError("failed to determine file size: " + path.string());
    }
    fileSize_ = static_cast<std::uint64_t>(sizeSigned);
}

const ChunkEntry& CcmfFile::Record(const ChunkEntry& entry) {
    if (entry.type == kChunkTypeVideo) {
        knowsVideo_ = true;
    }
    return known_.insert_or_assign(entry.offset, entry).first->second;
}

std::optional<ChunkEntry> CcmfFile::ReadHeaderAt(std::uint64_t offset) {
    if (const auto it = known_.find(offset); it != known_.end()) {
        return it->second;
    }
    ChunkEntry entry;
    if (!TryReadHeaderAt(file_, offset, fileSize_, entry)) {
        return std::nullopt;
    }
    return Record(entry);
}

std::optional<ChunkEntry> CcmfFile::FirstChunk() {
    return ReadHeaderAt(0);
}

std::optional<ChunkEntry> CcmfFile::NextChunkAfter(const ChunkEntry& entry) {
    const std::uint64_t next = entry.End();
    if (next >= fileSize_) {
        return std::nullopt;
    }
    return ReadHeaderAt(next);
}

const std::vector<ChunkEntry>& CcmfFile::TailChunks() {
    if (!tailChunks_) {
        std::vector<ChunkEntry> tail;
        if (auto found = FindTailChunks(file_, fileSize_)) {
            tail = std::move(*found);
            for (const ChunkEntry& entry : tail) {
                Record(entry);
            }
        }
        tailChunks_ = std::move(tail);
    }
    return *tailChunks_;
}

void CcmfFile::EnsureBootstrap() {
    if (bootstrapped_) {
        return;
    }
    bootstrapped_ = true;
    (void)FirstChunk();   // records the offset-0 chunk
    (void)TailChunks();   // records the trailing chain (duration anchor)
}

std::uint64_t CcmfFile::HomeToPtsAtMost(std::uint64_t target) {
    EnsureBootstrap();

    // Bracket `target` by pts among the chunks known so far: `lo` is the closest
    // known chunk at/below it, `hi` the closest above.
    auto bracket = [&](std::optional<ChunkEntry>& lo, std::optional<ChunkEntry>& hi) {
        lo.reset();
        hi.reset();
        for (const auto& [offset, entry] : known_) {
            if (entry.pts <= target) {
                if (!lo || entry.pts > lo->pts || (entry.pts == lo->pts && entry.offset > lo->offset)) {
                    lo = entry;
                }
            } else {
                if (!hi || entry.pts < hi->pts || (entry.pts == hi->pts && entry.offset < hi->offset)) {
                    hi = entry;
                }
            }
        }
    };

    std::optional<ChunkEntry> lo, hi;
    bracket(lo, hi);
    if (!lo) {
        // `target` precedes even the first chunk: start at the very beginning.
        return 0;
    }

    for (int iter = 0; iter < kMaxHomingIterations; ++iter) {
        if (!hi || hi->offset <= lo->End()) {
            break;  // lo's next boundary is hi (or EOF): lo is pinned just below target
        }

        // Interpolate a byte offset for `target` between the bracket, then resync
        // forward to the next real boundary there.
        std::optional<ChunkEntry> mid;
        const double frac = static_cast<double>(target - lo->pts) /
                            static_cast<double>(hi->pts - lo->pts);
        auto guess = static_cast<std::uint64_t>(
            static_cast<double>(lo->offset) +
            frac * static_cast<double>(hi->offset - lo->offset));
        guess = std::clamp(guess, lo->End(), hi->offset - 1);

        std::vector<ChunkEntry> collected;
        if (auto m = Resync(file_, guess, hi->offset, fileSize_, kHomingRequiredLinks, collected)) {
            for (const ChunkEntry& e : collected) {
                Record(e);
            }
            if (*m > lo->offset && *m < hi->offset) {
                mid = known_.at(*m);
            }
        }
        if (!mid) {
            // Interpolation found nothing strictly inside the bracket; take the
            // guaranteed next boundary (lo's chain-adjacent successor) instead.
            // This always makes progress and terminates the loop.
            mid = ReadHeaderAt(lo->End());
            if (!mid || mid->offset <= lo->offset || mid->offset >= hi->offset) {
                break;
            }
        }

        if (mid->pts <= target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    return lo->offset;
}

std::vector<ChunkEntry> CcmfFile::RegionAround(std::uint64_t pts) {
    EnsureBootstrap();

    std::uint64_t guard = kRegionGuardSamples;
    for (int grow = 0;; ++grow) {
        const std::uint64_t target = pts > guard ? pts - guard : 0;
        // When the target clamps to 0 (seeking within the first `guard` window),
        // begin at the very first chunk: homing to "the last chunk with pts <= 0"
        // would skip earlier chunks that share pts 0 (e.g. a video GOP alongside
        // an audio chunk both at pts 0). For target > 0 the guard keeps the start
        // strictly before the active GOP, so homing there is safe and cheaper.
        const std::uint64_t startOffset = target == 0 ? 0 : HomeToPtsAtMost(target);

        std::vector<ChunkEntry> run;
        bool sawVideoAtOrBefore = false;
        std::optional<ChunkEntry> cur = ReadHeaderAt(startOffset);
        while (cur) {
            run.push_back(*cur);
            if (cur->type == kChunkTypeVideo && cur->pts <= pts) {
                sawVideoAtOrBefore = true;
            }
            if (cur->pts > pts) {
                break;  // bracketed the target
            }
            const std::uint64_t next = cur->End();
            if (next >= fileSize_) {
                break;  // end of file
            }
            cur = ReadHeaderAt(next);
        }

        // If the file has video but this run captured no video chunk active at
        // `pts`, the active GOP began before the region start (a long GOP) --
        // extend the guard and re-home earlier. `target == 0` means we've
        // already reached the head, so the run is as complete as it gets.
        if (!knowsVideo_ || sawVideoAtOrBefore || target == 0 || grow >= kMaxRegionGrows) {
            return run;
        }
        guard *= 2;
    }
}

std::vector<ChunkEntry> CcmfFile::KnownChunks() const {
    std::vector<ChunkEntry> out;
    out.reserve(known_.size());
    for (const auto& [offset, entry] : known_) {
        out.push_back(entry);
    }
    return out;  // std::map keeps it ascending by offset
}

CcmfFile::FullIndex CcmfFile::IndexAll() {
    FullIndex full;
    std::uint64_t offset = 0;
    while (offset < fileSize_) {
        ChunkEntry entry;
        if (!TryReadHeaderAt(file_, offset, fileSize_, entry)) {
            throw CcmfError("malformed or truncated chunk at offset " + std::to_string(offset));
        }
        Record(entry);
        full.chunks.push_back(entry);
        if (entry.type == kChunkTypeVideo) {
            full.video.push_back(entry);
        } else if (entry.type == kChunkTypeAudio) {
            full.audio.push_back(entry);
        }
        offset = entry.End();
    }
    return full;
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
    // ReadChunkPayloadRange returns on-wire (possibly compressed) bytes; the
    // type decoders expect the raw payload, so inflate here (spec 4.1.2).
    return DecompressPayload(ReadChunkPayloadRange(entry, 0, entry.length),
                             entry.compression);
}

}  // namespace ccmfplayer

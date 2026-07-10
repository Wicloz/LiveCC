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
        return it->second;  // surrogate hit: no I/O
    }
    ChunkEntry entry;
    ++diskHeaderReads_;
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

    // Safeguarded interpolation search over the monotone offset<->pts mapping.
    // Interpolation converges superlinearly (~log log n) when the mapping is
    // smooth -- the common case -- but degrades toward linear when chunk sizes
    // vary wildly (dense all-keyframe GOPs next to tiny repeat-only GOPs make
    // bytes a poor proxy for time). A Brent-style safeguard forces a bisection
    // step whenever the previous step failed to at least halve the bracket, so
    // the byte width shrinks geometrically no matter how misleading the
    // interpolation is -- bounding the worst case to O(log2(bytes)) probes while
    // leaving the smooth-file fast path untouched.
    std::uint64_t prevWidth = hi->offset - lo->offset;
    bool forceBisect = false;

    for (int iter = 0; iter < kMaxHomingIterations; ++iter) {
        if (!hi || hi->offset <= lo->End()) {
            break;  // lo's next boundary is hi (or EOF): lo is pinned just below target
        }
        const std::uint64_t gapLo = lo->End();
        const std::uint64_t gapHi = hi->offset;

        std::uint64_t guess;
        if (forceBisect) {
            guess = gapLo + (gapHi - gapLo) / 2;  // guaranteed geometric shrink
        } else {
            const double frac = static_cast<double>(target - lo->pts) /
                                static_cast<double>(hi->pts - lo->pts);
            guess = static_cast<std::uint64_t>(
                static_cast<double>(lo->offset) +
                frac * static_cast<double>(hi->offset - lo->offset));
            guess = std::clamp(guess, gapLo, gapHi - 1);
        }

        // Resync forward from the guess to the next real boundary in the gap.
        std::optional<ChunkEntry> mid;
        std::vector<ChunkEntry> collected;
        ++homingProbes_;
        if (auto m = Resync(file_, guess, gapHi, fileSize_, kHomingRequiredLinks, collected)) {
            for (const ChunkEntry& e : collected) {
                Record(e);
            }
            if (*m > lo->offset && *m < gapHi) {
                mid = known_.at(*m);
            }
        }
        if (!mid) {
            // The guess fell inside the last chunk before hi (no boundary at or
            // after it within the gap): step to lo's guaranteed chain-adjacent
            // successor. Poor progress, but the safeguard below will bisect next.
            mid = ReadHeaderAt(lo->End());
            if (!mid || mid->offset <= lo->offset || mid->offset >= gapHi) {
                break;
            }
        }

        if (mid->pts <= target) {
            lo = mid;
        } else {
            hi = mid;
        }

        const std::uint64_t width = hi->offset - lo->offset;
        forceBisect = width > prevWidth / 2;  // didn't halve -> bisect next step
        prevWidth = width;
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
        // The active GOP began before the region start (a long, repeat-heavy
        // GOP). Grow the look-back geometrically by x4 (not x2) so an extreme
        // GOP spanning many guard-windows costs a handful of re-homes, not a
        // long ladder; the already-walked suffix stays in known_, so each
        // re-home only probes the new backward extension.
        guard *= 4;
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

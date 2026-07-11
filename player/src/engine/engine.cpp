#include "engine/engine.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <map>
#include <utility>

#include "engine/audio.hpp"

namespace ccmfplayer {

namespace {

// Bound on how many chunks the constructor walks from the head while probing
// for the first video chunk (grid size) and for audio presence -- so an
// audio-only file (no video chunk to find) can't turn the probe into a full
// scan. Real video files carry their keyframe within the first chunk or two.
constexpr int kHeadProbeChunks = 64;

// An audio chunk older than this before a target PTS can't possibly still cover
// it (chunks are ~2 s, spec 4.6); used to skip decoding stale candidates when
// resolving the roles active at a PTS.
constexpr std::uint64_t kMaxAudioChunkSamples = 4 * kSampleRate;

}  // namespace

std::size_t FindChunkIndexForPts(std::span<const ChunkEntry> chunks, std::uint64_t pts) noexcept {
    if (chunks.empty()) {
        return 0;
    }
    // First chunk whose pts is strictly greater than `pts`; the answer is
    // one before that (clamped to the first chunk if `pts` precedes it).
    auto it = std::upper_bound(
        chunks.begin(), chunks.end(), pts,
        [](std::uint64_t value, const ChunkEntry& entry) { return value < entry.pts; });
    if (it == chunks.begin()) {
        return 0;
    }
    return static_cast<std::size_t>(std::distance(chunks.begin(), std::prev(it)));
}

std::size_t FindFrameIndexForPts(std::span<const Frame> frames, std::uint64_t ptsIntoGop) noexcept {
    std::uint64_t accumulated = 0;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        accumulated += frames[i].duration;
        if (ptsIntoGop < accumulated) {
            return i;
        }
    }
    return frames.empty() ? 0 : frames.size() - 1;
}

PlaybackEngine::PlaybackEngine(const std::filesystem::path& path, std::uint32_t outputChannels)
    : file_(path), outputChannels_(outputChannels == 0 ? 1 : outputChannels) {
    outputRoles_ = OutputRolesForChannelCount(outputChannels_);
    BootstrapVideoAndDuration();
    Seek(0);
}

void PlaybackEngine::BootstrapVideoAndDuration() {
    // Head probe (bounded): find the first video chunk for the grid size, and
    // note whether audio appears near the head.
    std::optional<ChunkEntry> firstVideo;
    {
        std::optional<ChunkEntry> cur = file_.FirstChunk();
        for (int i = 0; cur && i < kHeadProbeChunks; ++i) {
            if (cur->type == kChunkTypeVideo && !firstVideo) {
                firstVideo = cur;
            } else if (cur->type == kChunkTypeAudio) {
                hasAudio_ = true;
            }
            if (firstVideo && hasAudio_) {
                break;  // learned both facts already
            }
            cur = file_.NextChunkAfter(*cur);
        }
    }

    // The trailing chain (read backwards from EOF, spec 4.1.1) anchors both the
    // duration and a second audio-presence check.
    const std::vector<ChunkEntry>& tail = file_.TailChunks();
    for (const ChunkEntry& entry : tail) {
        if (entry.type == kChunkTypeAudio) {
            hasAudio_ = true;
        }
    }

    // Grid size from the first video chunk. A video file whose very first video
    // chunk can't be decoded isn't playable at all: let that throw from the
    // constructor (historical contract), rather than pretending it opened.
    if (firstVideo) {
        const DecodedGop gop = DecodeVideoPayload(file_.ReadChunkPayload(*firstVideo));
        width_ = gop.width;
        height_ = gop.height;
        hasVideo_ = true;
    }

    // Duration: the maximum end-time across the trailing chunks (spec 4.2). A
    // corrupt trailing chunk falls back to its own start pts (best-effort).
    std::uint64_t duration = 0;
    for (const ChunkEntry& entry : tail) {
        std::uint64_t end = entry.pts;
        try {
            if (entry.type == kChunkTypeVideo) {
                const DecodedGop gop = DecodeVideoPayload(file_.ReadChunkPayload(entry));
                std::uint64_t span = 0;
                for (const Frame& frame : gop.frames) {
                    span += frame.duration;
                }
                end = entry.pts + span;
            } else if (entry.type == kChunkTypeAudio) {
                const DecodedAudio audio = DecodeAudioPayload(file_.ReadChunkPayload(entry));
                end = entry.pts + audio.pcm.size();
            } else {
                continue;
            }
        } catch (const std::exception&) {
            end = entry.pts;
        }
        duration = std::max(duration, end);
    }
    duration_ = duration;
}

bool PlaybackEngine::DecodeGopAt(const ChunkEntry& entry) noexcept {
    if (decodedVideoOffset_ == entry.offset) {
        return true;  // already the decoded GOP
    }
    try {
        decodedGop_ = DecodeVideoPayload(file_.ReadChunkPayload(entry));
        decodedVideoOffset_ = entry.offset;
        lastError_.clear();
        return true;
    } catch (const std::exception& e) {
        lastError_ = e.what();
        // Leave decodedGop_/decodedVideoOffset_ as they were: keep showing the
        // last successfully-decoded frame instead of blanking.
        return false;
    }
}

void PlaybackEngine::ResolveVideoAt(std::uint64_t pts) noexcept {
    if (!hasVideo_) {
        return;
    }
    // Reuse the cached GOP while it's still the active one.
    if (decodedVideoOffset_.has_value() && pts >= activeVideoStart_ && pts < activeVideoEnd_) {
        currentFrameIndex_ = FindFrameIndexForPts(decodedGop_.frames, pts - activeVideoStart_);
        return;
    }

    std::vector<ChunkEntry> run;
    try {
        run = file_.RegionAround(pts);
    } catch (const std::exception& e) {
        lastError_ = e.what();
        return;
    }

    std::optional<ChunkEntry> active;
    for (const ChunkEntry& entry : run) {
        if (entry.type != kChunkTypeVideo || entry.pts > pts) {
            continue;
        }
        if (!active || entry.pts > active->pts) {
            active = entry;
        }
    }
    if (!active) {
        return;  // couldn't resolve a GOP here; leave the last frame shown
    }

    if (DecodeGopAt(*active)) {
        // Advance the active-range cache only on a successful decode, so it
        // always describes the GOP actually on screen. Its end is the GOP's own
        // span (sum of frame durations) -- a GOP holds until its frames run out,
        // and the next one starts there. Deriving the end locally like this is
        // robust to interleaving (the next *video* chunk needn't be in `run`)
        // and to long GOPs; using the next video chunk's pts instead would leave
        // the cache open to end-of-file whenever the run stopped on an audio
        // chunk, freezing playback on the first GOP.
        // A GOP is self-describing (spec 4.4): its own width/height may differ
        // from earlier GOPs (a mid-file resolution change). Track the active
        // GOP's grid so Width()/Height() describe what CurrentFrame() actually
        // is, letting the app resize its view when it changes.
        width_ = decodedGop_.width;
        height_ = decodedGop_.height;
        std::uint64_t span = 0;
        for (const Frame& frame : decodedGop_.frames) {
            span += frame.duration;
        }
        activeVideoStart_ = active->pts;
        activeVideoEnd_ = active->pts + std::max<std::uint64_t>(span, 1);
        currentFrameIndex_ =
            FindFrameIndexForPts(decodedGop_.frames, pts > active->pts ? pts - active->pts : 0);
    } else if (decodedVideoOffset_.has_value()) {
        // Decode failed: keep the last good GOP visible, clamped by how far
        // ahead `pts` is within *that* GOP (its start is still activeVideoStart_,
        // which we deliberately did not move) -- so a seek well past it shows its
        // final frame rather than an early one.
        const std::uint64_t into = pts > activeVideoStart_ ? pts - activeVideoStart_ : 0;
        currentFrameIndex_ = FindFrameIndexForPts(decodedGop_.frames, into);
    }
}

const DecodedAudio* PlaybackEngine::DecodeAudioCached(const ChunkEntry& entry) noexcept {
    for (auto it = audioCache_.begin(); it != audioCache_.end(); ++it) {
        if (it->first == entry.offset) {
            audioCache_.splice(audioCache_.begin(), audioCache_, it);  // promote to MRU
            return &audioCache_.front().second;
        }
    }
    try {
        DecodedAudio decoded = DecodeAudioPayload(file_.ReadChunkPayload(entry));
        audioCache_.emplace_front(entry.offset, std::move(decoded));
    } catch (const std::exception& e) {
        lastError_ = e.what();
        return nullptr;
    }
    while (audioCache_.size() > audioCacheCap_) {
        audioCache_.pop_back();
    }
    return &audioCache_.front().second;
}

void PlaybackEngine::RefreshActiveAudio(std::uint64_t pts) noexcept {
    activeAudio_.clear();
    if (!hasAudio_) {
        return;
    }

    std::vector<ChunkEntry> run;
    try {
        run = file_.RegionAround(pts);
    } catch (const std::exception& e) {
        lastError_ = e.what();
        return;
    }

    // Per source role, keep the latest audio chunk that actually covers `pts`.
    std::map<std::uint8_t, ActiveRole> byRole;
    for (const ChunkEntry& entry : run) {
        if (entry.type != kChunkTypeAudio || entry.pts > pts) {
            continue;
        }
        if (entry.pts + kMaxAudioChunkSamples <= pts) {
            continue;  // too old to still cover `pts`
        }
        const DecodedAudio* decoded = DecodeAudioCached(entry);
        if (!decoded) {
            continue;
        }
        if (pts >= entry.pts + decoded->pcm.size()) {
            continue;  // decoded span doesn't reach `pts`
        }
        auto it = byRole.find(decoded->channel);
        if (it == byRole.end() || entry.pts > it->second.chunk.pts) {
            // Copy the PCM now: subsequent DecodeAudioCached calls may evict it.
            byRole[decoded->channel] = ActiveRole{decoded->channel, entry, decoded->pcm};
        }
    }

    activeAudioValidFrom_ = pts;
    if (byRole.empty()) {
        activeAudioValidUntil_ = pts;  // nothing covers the cursor: PullAudio stops
        return;
    }
    std::uint64_t validUntil = std::numeric_limits<std::uint64_t>::max();
    for (auto& [role, active] : byRole) {
        validUntil = std::min(validUntil, active.chunk.pts + active.pcm.size());
        activeAudio_.push_back(std::move(active));
    }
    activeAudioValidUntil_ = validUntil;
}

void PlaybackEngine::Play() noexcept {
    if (duration_ == 0) {
        return;
    }
    if (currentPts_ >= duration_) {
        Seek(0);
    }
    playing_ = true;
}

void PlaybackEngine::Pause() noexcept {
    playing_ = false;
}

void PlaybackEngine::TogglePlayPause() noexcept {
    if (playing_) {
        Pause();
    } else {
        Play();
    }
}

void PlaybackEngine::Seek(std::uint64_t pts) noexcept {
    currentPts_ = std::min(pts, duration_);
    audioCursorPts_ = currentPts_;
    activeAudio_.clear();
    activeAudioValidFrom_ = 0;
    activeAudioValidUntil_ = 0;  // force a refresh on the next PullAudio
    ResolveVideoAt(currentPts_);
}

void PlaybackEngine::Advance(double dtSeconds) noexcept {
    if (!playing_ || dtSeconds <= 0.0) {
        return;
    }
    if (hasAudio_) {
        // Audio is the master clock while present (see the class doc's sync
        // model): follow wherever PullAudio has gotten to.
        currentPts_ = std::min(audioCursorPts_, duration_);
    } else {
        const auto deltaSamples = static_cast<std::uint64_t>(dtSeconds * kSampleRate);
        currentPts_ = std::min(currentPts_ + deltaSamples, duration_);
    }
    ResolveVideoAt(currentPts_);
    if (currentPts_ >= duration_) {
        if (looping_ && duration_ > 0) {
            Seek(0);  // stays playing_ == true; resets currentPts_/audioCursorPts_ to 0
        } else {
            playing_ = false;
        }
    }
}

const Frame* PlaybackEngine::CurrentFrame() const noexcept {
    if (!decodedVideoOffset_.has_value() || currentFrameIndex_ >= decodedGop_.frames.size()) {
        return nullptr;
    }
    return &decodedGop_.frames[currentFrameIndex_];
}

std::size_t PlaybackEngine::PullAudio(std::span<std::int16_t> out) noexcept {
    if (!hasAudio_ || outputChannels_ == 0) {
        return 0;
    }
    const std::size_t channels = outputChannels_;
    const std::size_t framesRequested = out.size() / channels;
    std::size_t framesWritten = 0;

    // Scratch buffers reused across every segment/channel below, so a warmed
    // PullAudio does no per-call heap allocation for these -- allocation in the
    // audio-feed path is exactly what real-time audio wants to avoid. (The
    // per-channel `sources` still comes from ResolveOutputSources by value; only
    // these two, which we build ourselves, are hoisted out of the loop.)
    std::vector<std::uint8_t> present;
    std::vector<const ActiveRole*> srcRoles;

    while (framesWritten < framesRequested) {
        if (activeAudio_.empty() || audioCursorPts_ >= activeAudioValidUntil_ ||
            audioCursorPts_ < activeAudioValidFrom_) {
            RefreshActiveAudio(audioCursorPts_);
        }
        if (activeAudio_.empty()) {
            break;  // no audio covers the cursor -> stop (silence past here)
        }

        // Contiguous samples available across every active source role, so a
        // malformed per-role length can't drive an out-of-bounds read.
        std::size_t available = std::numeric_limits<std::size_t>::max();
        bool ok = true;
        for (const ActiveRole& role : activeAudio_) {
            if (audioCursorPts_ < role.chunk.pts) {
                ok = false;
                break;
            }
            const auto offset = static_cast<std::size_t>(audioCursorPts_ - role.chunk.pts);
            if (offset >= role.pcm.size()) {
                ok = false;
                break;
            }
            available = std::min(available, role.pcm.size() - offset);
        }
        if (!ok || available == 0) {
            break;
        }

        const std::size_t toCopy = std::min(available, framesRequested - framesWritten);

        // Roles present in this segment, and (once per segment) the source
        // roles that feed each fixed output channel.
        present.clear();
        present.reserve(activeAudio_.size());
        for (const ActiveRole& role : activeAudio_) {
            present.push_back(role.role);
        }

        for (std::size_t ch = 0; ch < channels; ++ch) {
            const std::vector<std::uint8_t> sources =
                ResolveOutputSources(outputRoles_[ch], present);
            srcRoles.clear();
            srcRoles.reserve(sources.size());
            for (std::uint8_t wanted : sources) {
                for (const ActiveRole& role : activeAudio_) {
                    if (role.role == wanted) {
                        srcRoles.push_back(&role);
                        break;
                    }
                }
            }
            for (std::size_t f = 0; f < toCopy; ++f) {
                std::int32_t sum = 0;
                for (const ActiveRole* role : srcRoles) {
                    const auto idx =
                        static_cast<std::size_t>(audioCursorPts_ - role->chunk.pts) + f;
                    sum += role->pcm[idx];
                }
                const std::int16_t sample =
                    srcRoles.empty()
                        ? std::int16_t{0}
                        : static_cast<std::int16_t>(sum / static_cast<std::int32_t>(srcRoles.size()));
                out[(framesWritten + f) * channels + ch] = sample;
            }
        }

        framesWritten += toCopy;
        audioCursorPts_ += toCopy;
    }

    return framesWritten * channels;
}

}  // namespace ccmfplayer

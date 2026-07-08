#include "engine/engine.hpp"

#include <algorithm>
#include <exception>
#include <limits>

namespace ccmfplayer {

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

PlaybackEngine::PlaybackEngine(const std::filesystem::path& path) : file_(path) {
    SelectAudioChannel();
    ComputeDuration();
    Seek(0);
}

void PlaybackEngine::SelectAudioChannel() {
    std::vector<ChunkEntry> mono, left, right;
    for (const ChunkEntry& entry : file_.AudioChunks()) {
        try {
            // The a-hdr is the first payload byte; a compressed payload must be
            // inflated first (its leading bytes are the LZ4 wrapper, not a-hdr).
            const auto payloadHead = entry.compression == kCompressionNone
                ? file_.ReadChunkPayloadRange(entry, 0, 1)
                : file_.ReadChunkPayload(entry);
            const auto channel = static_cast<std::uint8_t>(
                std::to_integer<std::uint8_t>(payloadHead[0]) & 0x0F);
            if (channel == kChannelMono) {
                mono.push_back(entry);
            } else if (channel == kChannelFrontLeft) {
                left.push_back(entry);
            } else if (channel == kChannelFrontRight) {
                right.push_back(entry);
            }
            // Other roles (center/LFE/surround/rear) are present in the
            // file but out of scope for this player's output -- see the
            // class doc's audio scope note.
        } catch (const std::exception&) {
            continue;  // malformed chunk header peek: skip it, don't fail the whole file
        }
    }

    audioRoles_.clear();
    if (!left.empty() && !right.empty()) {
        audioRoles_.push_back(AudioRoleState{std::move(left), std::nullopt, {}});
        audioRoles_.push_back(AudioRoleState{std::move(right), std::nullopt, {}});
    } else if (!mono.empty()) {
        audioRoles_.push_back(AudioRoleState{std::move(mono), std::nullopt, {}});
    } else if (!left.empty()) {
        // front-left with no matching front-right: still play it.
        audioRoles_.push_back(AudioRoleState{std::move(left), std::nullopt, {}});
    }
}

void PlaybackEngine::ComputeDuration() {
    std::uint64_t videoEnd = 0;
    if (!file_.VideoChunks().empty()) {
        // The first chunk establishes the grid size (constant for the whole
        // file) and, if it can't even be decoded, the file isn't playable
        // as video at all -- let that exception propagate from the
        // constructor rather than pretending the file opened successfully.
        const DecodedGop firstGop =
            DecodeVideoPayload(file_.ReadChunkPayload(file_.VideoChunks().front()));
        width_ = firstGop.width;
        height_ = firstGop.height;

        const ChunkEntry& last = file_.VideoChunks().back();
        try {
            const DecodedGop lastGop = DecodeVideoPayload(file_.ReadChunkPayload(last));
            std::uint64_t span = 0;
            for (const Frame& frame : lastGop.frames) {
                span += frame.duration;
            }
            videoEnd = last.pts + span;
        } catch (const std::exception&) {
            videoEnd = last.pts;  // best-effort duration if the trailing chunk is bad
        }
    }

    std::uint64_t audioEnd = 0;
    if (!audioRoles_.empty() && !audioRoles_.front().chunks.empty()) {
        const ChunkEntry& last = audioRoles_.front().chunks.back();
        try {
            const DecodedAudio audio = DecodeAudioPayload(file_.ReadChunkPayload(last));
            audioEnd = last.pts + audio.pcm.size();
        } catch (const std::exception&) {
            audioEnd = last.pts;
        }
    }

    duration_ = std::max(videoEnd, audioEnd);
}

void PlaybackEngine::DecodeVideoChunkAt(std::size_t chunkIndex) noexcept {
    if (decodedVideoChunkIndex_ == chunkIndex) {
        return;
    }
    try {
        decodedGop_ = DecodeVideoPayload(file_.ReadChunkPayload(file_.VideoChunks()[chunkIndex]));
        decodedVideoChunkIndex_ = chunkIndex;
        lastError_.clear();
    } catch (const std::exception& e) {
        lastError_ = e.what();
        // Leave decodedGop_/decodedVideoChunkIndex_ as they were: keep
        // showing the last successfully-decoded frame instead of blanking.
    }
}

void PlaybackEngine::ResolveVideoFrame() noexcept {
    if (file_.VideoChunks().empty()) {
        return;
    }
    const std::size_t desiredChunkIdx = FindChunkIndexForPts(file_.VideoChunks(), currentPts_);
    DecodeVideoChunkAt(desiredChunkIdx);
    if (!decodedVideoChunkIndex_.has_value()) {
        return;  // nothing has ever decoded successfully
    }
    const ChunkEntry& activeChunk = file_.VideoChunks()[*decodedVideoChunkIndex_];
    const std::uint64_t ptsIntoGop =
        currentPts_ > activeChunk.pts ? currentPts_ - activeChunk.pts : 0;
    currentFrameIndex_ = FindFrameIndexForPts(decodedGop_.frames, ptsIntoGop);
}

bool PlaybackEngine::EnsureAudioRoleDecoded(AudioRoleState& role, std::uint64_t pts) noexcept {
    if (role.chunks.empty()) {
        return false;
    }
    const std::size_t chunkIdx = FindChunkIndexForPts(role.chunks, pts);
    if (role.decodedChunkIndex == chunkIdx) {
        return true;
    }
    try {
        role.decodedPcm = DecodeAudioPayload(file_.ReadChunkPayload(role.chunks[chunkIdx])).pcm;
        role.decodedChunkIndex = chunkIdx;
        lastError_.clear();
        return true;
    } catch (const std::exception& e) {
        lastError_ = e.what();
        return false;
    }
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
    ResolveVideoFrame();
}

void PlaybackEngine::Advance(double dtSeconds) noexcept {
    if (!playing_ || dtSeconds <= 0.0) {
        return;
    }
    if (!audioRoles_.empty()) {
        // Audio is the master clock while both tracks are present (see the
        // class doc's sync model): just follow wherever PullAudio has
        // gotten to.
        currentPts_ = std::min(audioCursorPts_, duration_);
    } else {
        const auto deltaSamples = static_cast<std::uint64_t>(dtSeconds * kSampleRate);
        currentPts_ = std::min(currentPts_ + deltaSamples, duration_);
    }
    ResolveVideoFrame();
    if (currentPts_ >= duration_) {
        if (looping_ && duration_ > 0) {
            Seek(0);  // stays playing_ == true; resets currentPts_/audioCursorPts_ to 0
        } else {
            playing_ = false;
        }
    }
}

const Frame* PlaybackEngine::CurrentFrame() const noexcept {
    if (!decodedVideoChunkIndex_.has_value() || currentFrameIndex_ >= decodedGop_.frames.size()) {
        return nullptr;
    }
    return &decodedGop_.frames[currentFrameIndex_];
}

std::size_t PlaybackEngine::PullAudio(std::span<std::int16_t> out) noexcept {
    if (audioRoles_.empty()) {
        return 0;
    }
    const std::size_t channels = audioRoles_.size();
    const std::size_t framesRequested = out.size() / channels;
    std::size_t framesWritten = 0;

    while (framesWritten < framesRequested) {
        bool ready = true;
        for (AudioRoleState& role : audioRoles_) {
            if (!EnsureAudioRoleDecoded(role, audioCursorPts_)) {
                ready = false;
            }
        }
        if (!ready) {
            break;
        }

        // Bound how many contiguous samples are available across EVERY
        // role independently (rather than trusting they all match), so a
        // malformed file with mismatched per-role chunk lengths still can't
        // read out of bounds.
        std::size_t available = std::numeric_limits<std::size_t>::max();
        bool exhausted = false;
        for (const AudioRoleState& role : audioRoles_) {
            const ChunkEntry& chunkEntry = role.chunks[*role.decodedChunkIndex];
            if (audioCursorPts_ < chunkEntry.pts) {
                exhausted = true;
                break;
            }
            const std::uint64_t offset = audioCursorPts_ - chunkEntry.pts;
            if (offset >= role.decodedPcm.size()) {
                exhausted = true;
                break;
            }
            available =
                std::min(available, static_cast<std::size_t>(role.decodedPcm.size() - offset));
        }
        if (exhausted) {
            break;
        }

        const std::size_t toCopy = std::min(available, framesRequested - framesWritten);
        for (std::size_t ch = 0; ch < channels; ++ch) {
            const AudioRoleState& role = audioRoles_[ch];
            const ChunkEntry& chunkEntry = role.chunks[*role.decodedChunkIndex];
            const auto offset = static_cast<std::size_t>(audioCursorPts_ - chunkEntry.pts);
            for (std::size_t f = 0; f < toCopy; ++f) {
                out[(framesWritten + f) * channels + ch] = role.decodedPcm[offset + f];
            }
        }
        framesWritten += toCopy;
        audioCursorPts_ += toCopy;
    }

    return framesWritten * channels;
}

}  // namespace ccmfplayer

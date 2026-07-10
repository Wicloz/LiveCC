#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <list>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "engine/audio.hpp"
#include "engine/index.hpp"
#include "engine/video.hpp"

namespace ccmfplayer {

// Resolves which chunk in an ascending-PTS chunk list is "active" at `pts`:
// the last chunk whose own pts is <= `pts` (every chunk holds until the next
// one starts). A `pts` before the first chunk clamps to index 0; `chunks`
// MUST be sorted ascending by pts. A generic binary-search helper over an
// in-memory chunk run (e.g. one RegionAround() result); returns 0 for an empty
// `chunks` (callers must not dereference without checking emptiness first).
[[nodiscard]] std::size_t FindChunkIndexForPts(std::span<const ChunkEntry> chunks,
                                                std::uint64_t pts) noexcept;

// Resolves which frame in a GOP's decoded frame list is active `ptsIntoGop`
// samples after the GOP's own start, by accumulating frame durations
// (spec 4.5's `duration`) until the running total exceeds `ptsIntoGop`.
// Clamps to the last frame if `ptsIntoGop` runs past the GOP's total
// duration. `frames` must not be empty.
[[nodiscard]] std::size_t FindFrameIndexForPts(std::span<const Frame> frames,
                                                std::uint64_t ptsIntoGop) noexcept;

// Plays back one .ccmf file: owns all decode state and exposes a pull-based
// interface (CurrentFrame()/PullAudio()) so an app layer (raylib, Stage 6)
// can ask "what's due right now" every render tick without knowing anything
// about chunks, GOPs, or codecs. Everything here is framework-independent
// and fully unit-testable without a window or audio device.
//
// Audio scope: the OUTPUT layout is fixed by the app/OS device (mono or
// stereo, `outputChannels` in the constructor), independent of what the file
// carries. The file's role set may change at any point (spec 5.4), so nothing
// is pre-scanned; instead, whatever roles are present near the current PTS are
// discovered opportunistically and up/down-mixed into the fixed output roles
// (ResolveOutputSources, audio.hpp): a mono source fills both stereo channels,
// a stereo source downmixes to a mono output, and so on. PullAudio() therefore
// always emits exactly ChannelCount() interleaved channels regardless of the
// source layout.
//
// Sync model: while playing, if the file has audio, audio is the master
// clock -- PullAudio() advances the audio cursor as the app layer's audio
// device consumes samples, and the video position (CurrentPts()) simply
// follows it every Advance() call, rather than running two independently
// drifting clocks. A video-only file falls back to a dt-driven clock.
//
// Resilience: a corrupt chunk anywhere past the first video keyframe does
// not crash playback -- decode failures are caught internally and surfaced
// via HasError()/LastError(), leaving the last successfully-decoded frame/
// audio in place rather than throwing out of Seek()/Advance()/PullAudio().
// The constructor is the one place that still throws (CcmfError): a file
// whose index can't be built, or whose very first video chunk can't be
// decoded, isn't playable at all.
class PlaybackEngine {
public:
    // `outputChannels` is the fixed device output layout: 1 (mono) or 2
    // (stereo, the default). The file's own channel layout is irrelevant to
    // this -- source roles are remixed into these output channels (see the
    // class doc's audio scope).
    explicit PlaybackEngine(const std::filesystem::path& path, std::uint32_t outputChannels = 2);

    PlaybackEngine(const PlaybackEngine&) = delete;
    PlaybackEngine& operator=(const PlaybackEngine&) = delete;
    PlaybackEngine(PlaybackEngine&&) = default;
    PlaybackEngine& operator=(PlaybackEngine&&) = default;

    [[nodiscard]] bool HasVideo() const noexcept { return hasVideo_; }
    // Best-effort: true if any audio chunk was found near the file's head or
    // tail. (A file whose audio appears only strictly mid-file, with none near
    // either end, may read as false -- producers keep a stable layout, so this
    // is a practical, not exact, signal; see the class doc.)
    [[nodiscard]] bool HasAudio() const noexcept { return hasAudio_; }
    // 0 if HasAudio() is false; otherwise the fixed output channel count (1 or
    // 2), NOT the file's source channel count. PullAudio's output is
    // interleaved frames of exactly this many samples each.
    [[nodiscard]] std::uint32_t ChannelCount() const noexcept {
        return hasAudio_ ? outputChannels_ : 0;
    }
    // Grid size of the GOP CurrentFrame() belongs to. Usually constant, but a
    // self-describing file MAY change resolution mid-stream (spec 4.4), so this
    // tracks the active GOP and can change after an Advance()/Seek() -- callers
    // rendering to a fixed-size surface should re-check it and resize on change.
    [[nodiscard]] std::uint16_t Width() const noexcept { return width_; }
    [[nodiscard]] std::uint16_t Height() const noexcept { return height_; }
    // The file's total span in samples (spec 4.2 units), computed once at
    // load from the last video/audio chunk actually decoded -- exact unless
    // that trailing chunk turned out to be corrupt, in which case it falls
    // back to that chunk's own start pts.
    [[nodiscard]] std::uint64_t Duration() const noexcept { return duration_; }
    [[nodiscard]] std::uint64_t CurrentPts() const noexcept { return currentPts_; }
    [[nodiscard]] bool IsPlaying() const noexcept { return playing_; }

    // When true, reaching the end of the file restarts from 0 and keeps
    // playing instead of auto-pausing (see Advance()). Off by default.
    [[nodiscard]] bool IsLooping() const noexcept { return looping_; }
    void SetLooping(bool looping) noexcept { looping_ = looping; }
    void ToggleLooping() noexcept { looping_ = !looping_; }

    [[nodiscard]] bool HasError() const noexcept { return !lastError_.empty(); }
    [[nodiscard]] const std::string& LastError() const noexcept { return lastError_; }

    // Starts (or resumes) playback. Replays from the start if called after
    // reaching the end. A no-op if the file has nothing to play at all.
    void Play() noexcept;
    void Pause() noexcept;
    void TogglePlayPause() noexcept;

    // Moves the playback position to `pts` (clamped to [0, Duration()]),
    // decoding whatever GOP/audio chunk that falls in. Does not change
    // IsPlaying(). Cheap: at most one GOP's worth of frames gets decoded.
    // A caller streaming audio through a separate device buffer (Stage 6's
    // AudioOutput) should flush anything already queued there after calling
    // this -- CurrentPts() dropping below its pre-call value (a wraparound)
    // is the signal that a loop restart happened inside Advance() and needs
    // the same treatment.
    void Seek(std::uint64_t pts) noexcept;

    // Advances the playback clock if playing (a no-op otherwise); see the
    // class doc's sync model for how `dtSeconds` is used (or ignored, for an
    // audio-driven file). On reaching the end, restarts from 0 if IsLooping()
    // is true, otherwise auto-pauses.
    void Advance(double dtSeconds) noexcept;

    // The frame that should be on screen right now, or nullptr if this file
    // has no video (or nothing has decoded successfully yet).
    [[nodiscard]] const Frame* CurrentFrame() const noexcept;

    // Fills `out` with up to out.size()/ChannelCount() interleaved audio
    // frames starting at the engine's current audio cursor, advancing the
    // cursor by however many samples were actually written; returns that
    // count (a multiple of ChannelCount()). `out.size()` must be a multiple
    // of ChannelCount(). Anything in `out` beyond what was written is left
    // untouched (callers wanting silence past end-of-track should zero `out`
    // first). A no-op returning 0 if HasAudio() is false.
    std::size_t PullAudio(std::span<std::int16_t> out) noexcept;

private:
    // One source audio role active at the cursor: its chunk plus decoded PCM.
    struct ActiveRole {
        std::uint8_t role = 0;
        ChunkEntry chunk;
        std::vector<std::int16_t> pcm;
    };

    void BootstrapVideoAndDuration();  // dims + duration from head probe + tail
    void ResolveVideoAt(std::uint64_t pts) noexcept;
    [[nodiscard]] bool DecodeGopAt(const ChunkEntry& entry) noexcept;  // false on decode failure
    void RefreshActiveAudio(std::uint64_t pts) noexcept;
    // Decodes an audio chunk with a small LRU cache (returns nullptr on decode
    // failure). Cache keyed by chunk offset.
    [[nodiscard]] const DecodedAudio* DecodeAudioCached(const ChunkEntry& entry) noexcept;

    CcmfFile file_;
    std::uint16_t width_ = 0;
    std::uint16_t height_ = 0;
    std::uint64_t duration_ = 0;
    bool hasVideo_ = false;
    bool hasAudio_ = false;
    std::uint32_t outputChannels_ = 2;
    std::vector<std::uint8_t> outputRoles_;  // fixed output layout (spec 4.6 roles)

    std::uint64_t currentPts_ = 0;
    bool playing_ = false;
    bool looping_ = false;
    std::string lastError_;

    // Video: the currently-decoded GOP and which chunk it came from, plus the
    // pts at which it stops being the active GOP (the next video chunk's pts,
    // or duration).
    std::optional<std::uint64_t> decodedVideoOffset_;
    std::uint64_t activeVideoStart_ = 0;
    std::uint64_t activeVideoEnd_ = 0;
    DecodedGop decodedGop_;
    std::size_t currentFrameIndex_ = 0;

    // Audio: the roles active at the cursor and the pts window over which that
    // set stays valid; refreshed lazily when the cursor leaves it.
    std::uint64_t audioCursorPts_ = 0;
    std::vector<ActiveRole> activeAudio_;
    std::uint64_t activeAudioValidFrom_ = 0;
    std::uint64_t activeAudioValidUntil_ = 0;

    // Small LRU of decoded audio chunks (offset -> PCM), so re-deriving the
    // active set within the same chunks doesn't re-decode.
    std::list<std::pair<std::uint64_t, DecodedAudio>> audioCache_;
    std::size_t audioCacheCap_ = 8;
};

}  // namespace ccmfplayer

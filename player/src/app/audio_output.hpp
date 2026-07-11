#pragma once

#include <cstdint>

#include <raylib.h>

#include "engine/engine.hpp"

namespace ccmfplayer {

// Owns a raylib AudioStream and keeps it fed from a PlaybackEngine's
// PullAudio(), once per render tick. Construct only when the engine's
// HasAudio() is true (a channel count of 0 makes no sense to stream).
class AudioOutput {
public:
    explicit AudioOutput(std::uint32_t channelCount);
    ~AudioOutput();

    AudioOutput(const AudioOutput&) = delete;
    AudioOutput& operator=(const AudioOutput&) = delete;

    // Tops up the stream's buffer from `engine` if raylib reports it's
    // ready for more, but only while `engine` is playing -- otherwise the
    // stream is paused so the audio (and PullAudio's cursor) doesn't keep
    // silently advancing while the UI shows "paused".
    void Refill(PlaybackEngine& engine);

    // Discards whatever is currently queued in the stream (stale
    // pre-seek audio raylib hasn't played yet) and restarts it. Call this
    // right after any PlaybackEngine::Seek() so playback doesn't briefly
    // replay a snippet from the old position.
    void Flush();

    // Pauses the stream and stops feeding it -- for an in-progress scrub drag,
    // where the position jumps every frame and playing the audio would be
    // garbage. Resume by calling Flush() (drops the stale buffer) then Refill().
    void Suspend();

private:
    std::uint32_t channelCount_;
    AudioStream stream_{};
};

}  // namespace ccmfplayer

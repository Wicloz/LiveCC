#include "app/audio_output.hpp"

#include <cstddef>
#include <vector>

namespace ccmfplayer {

namespace {
// One refill's worth of interleaved frames (~43ms at 48kHz). Small enough
// to keep seek-to-audible latency low; large enough that refilling once per
// render tick (~60 Hz) comfortably keeps the hardware buffer fed.
//
// MUST match what raylib actually allocates per internal sub-buffer, or
// UpdateAudioStream's frameCount argument can exceed it -- silently
// corrupting/dropping the write (raylib logs "STREAM: Attempting to write
// too many frames to buffer") while PlaybackEngine::PullAudio's cursor still
// advances as if the data had been delivered. That mismatch was the root
// cause of both "no audio" and "video plays too fast" (the video clock
// follows the audio cursor -- see engine.hpp's sync-model doc -- so a
// cursor racing ahead of what's actually reaching the speaker drags video
// with it). SetAudioStreamBufferSizeDefault() pins raylib's allocation to
// this exact value instead of relying on its undocumented default.
constexpr int kBufferFrames = 2048;
}  // namespace

AudioOutput::AudioOutput(std::uint32_t channelCount) : channelCount_(channelCount) {
    SetAudioStreamBufferSizeDefault(kBufferFrames);
    stream_ = LoadAudioStream(kSampleRate, 16, channelCount_);
    PlayAudioStream(stream_);
}

AudioOutput::~AudioOutput() {
    UnloadAudioStream(stream_);
}

void AudioOutput::Refill(PlaybackEngine& engine) {
    if (!engine.IsPlaying()) {
        if (IsAudioStreamPlaying(stream_)) {
            PauseAudioStream(stream_);
        }
        return;
    }
    if (!IsAudioStreamPlaying(stream_)) {
        ResumeAudioStream(stream_);
    }
    if (!IsAudioStreamProcessed(stream_)) {
        return;
    }

    std::vector<std::int16_t> buffer(static_cast<std::size_t>(kBufferFrames) * channelCount_, 0);
    engine.PullAudio(buffer);
    UpdateAudioStream(stream_, buffer.data(), kBufferFrames);
}

void AudioOutput::Flush() {
    StopAudioStream(stream_);
    PlayAudioStream(stream_);
}

}  // namespace ccmfplayer

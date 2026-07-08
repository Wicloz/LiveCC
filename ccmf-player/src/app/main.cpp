#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

#include <raylib.h>

#include "app/audio_output.hpp"
#include "app/ui.hpp"
#include "app/video_view.hpp"
#include "engine/controls_math.hpp"
#include "engine/engine.hpp"

namespace {

// Used before any video frame is available to pace against (no video track,
// or nothing decoded yet) -- see TargetFpsForFrameDuration.
constexpr int kFallbackFps = 60;

// --selftest draws exactly one frame and exits 0 immediately, so the app can
// be smoke-tested from a script without a human at the window (no real
// window/GPU/audio device is faked -- everything still initializes for
// real, this only skips waiting for WindowShouldClose()).
bool ParseSelfTestFlag(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) {
            return true;
        }
    }
    return false;
}

const char* FirstNonFlagArg(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--", 2) != 0) {
            return argv[i];
        }
    }
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    const bool selfTest = ParseSelfTestFlag(argc, argv);
    const char* path = FirstNonFlagArg(argc, argv);

    InitWindow(960, 540, "CCMF Player");
    InitAudioDevice();

    // The render loop's target rate tracks the file's OWN video cadence
    // (recomputed every tick from CurrentFrame()->duration, not cached per
    // chunk -- spec 4.5 makes duration a per-frame field, so different GOPs
    // in the same file aren't required to share one frame rate). Starts at
    // the fallback so the pre-load usage/error screen (no engine, so the
    // per-tick recompute below never runs) still paces sanely instead of
    // spinning uncapped.
    int currentTargetFps = kFallbackFps;
    if (!selfTest) {
        SetTargetFPS(currentTargetFps);
    }

    std::unique_ptr<ccmfplayer::PlaybackEngine> engine;
    std::unique_ptr<ccmfplayer::VideoView> videoView;
    std::unique_ptr<ccmfplayer::AudioOutput> audioOutput;
    ccmfplayer::PlayerControls controls;
    std::string statusText = "usage: ccmf_player <path-to-file.ccmf>";

    if (path != nullptr) {
        try {
            engine = std::make_unique<ccmfplayer::PlaybackEngine>(path);
            if (engine->HasVideo()) {
                videoView =
                    std::make_unique<ccmfplayer::VideoView>(engine->Width(), engine->Height());
                videoView->Update(engine->CurrentFrame());
            } else {
                statusText = "No video track";
            }
            if (engine->HasAudio()) {
                audioOutput = std::make_unique<ccmfplayer::AudioOutput>(engine->ChannelCount());
            }
        } catch (const std::exception& e) {
            statusText = e.what();
            engine.reset();
        }
    }

    bool shouldClose = false;
    while (!shouldClose) {
        const double dt = GetFrameTime();
        const int windowWidth = GetScreenWidth();
        const int windowHeight = GetScreenHeight();

        if (engine) {
            const bool seeked = controls.Update(*engine, windowWidth, windowHeight);
            if (seeked && audioOutput) {
                audioOutput->Flush();
            }

            const std::uint64_t ptsBeforeAdvance = engine->CurrentPts();
            engine->Advance(dt);
            // A loop restart (PlaybackEngine::Advance reaching the end with
            // IsLooping() set) is a seek too -- CurrentPts() dropping below
            // its pre-call value is how it shows up here, since Advance()
            // has no other way to signal it happened.
            if (audioOutput && engine->CurrentPts() < ptsBeforeAdvance) {
                audioOutput->Flush();
            }

            if (videoView) {
                videoView->Update(engine->CurrentFrame());
            }
            if (audioOutput) {
                audioOutput->Refill(*engine);
            }

            if (!selfTest) {
                int desiredFps = kFallbackFps;
                if (const ccmfplayer::Frame* frame = engine->CurrentFrame()) {
                    desiredFps = ccmfplayer::TargetFpsForFrameDuration(frame->duration,
                                                                       kFallbackFps);
                }
                if (desiredFps != currentTargetFps) {
                    SetTargetFPS(desiredFps);
                    currentTargetFps = desiredFps;
                }
            }
        }

        BeginDrawing();
        ClearBackground(BLACK);
        if (videoView) {
            videoView->Draw(windowWidth, windowHeight);
        } else {
            DrawText(statusText.c_str(), 32, 32, 20, RAYWHITE);
        }
        if (engine) {
            controls.Draw(*engine, windowWidth, windowHeight);
        }
        EndDrawing();

        shouldClose = selfTest || WindowShouldClose();
    }

    audioOutput.reset();
    videoView.reset();
    engine.reset();
    CloseAudioDevice();
    CloseWindow();
    return 0;
}

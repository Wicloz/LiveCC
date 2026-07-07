#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

#include <raylib.h>

#include "app/audio_output.hpp"
#include "app/ui.hpp"
#include "app/video_view.hpp"
#include "engine/engine.hpp"

namespace {

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
    if (!selfTest) {
        // Mouse/keyboard state is sampled once per loop iteration (raylib
        // polls it inside EndDrawing()); a full press+release cycle
        // delivered within one iteration's window is invisible to every
        // frame that runs afterward, no matter how the button code is
        // written. A too-low frame rate is what turns a fast click into
        // "nothing happened" -- some mouse utilities (reported: Kensington
        // Konnect) synthesize clicks fast enough to fall inside a 60fps
        // (~16.6ms) window. 240fps (~4.2ms) shrinks that window well below
        // any observed synthesized-click duration; the video's own frame
        // rate is unrelated (paced by PlaybackEngine's PTS clock, not by
        // how often this loop redraws).
        SetTargetFPS(240);
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

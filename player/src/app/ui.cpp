#include "app/ui.hpp"

#include <string>

#include <raylib.h>

#include "engine/controls_math.hpp"

namespace ccmfplayer {

namespace {

constexpr float kBarHeight = 56.0f;
constexpr float kMargin = 12.0f;
constexpr float kButtonSize = 40.0f;
constexpr float kButtonSpacing = 12.0f;
constexpr float kTimeTextWidth = 150.0f;  // reserved space for "H:MM:SS / H:MM:SS"
constexpr float kSeekBarThickness = 8.0f;
constexpr std::uint64_t kKeySeekStepSamples = 5ull * kSampleRate;

// Minimum spacing between the actual (expensive) seek+decode while dragging the
// scrub bar. The playhead follows the mouse every frame regardless; only the
// preview decode is rate-limited, so a heavy per-seek cost on a long file can't
// stall the loop and make the bar lag the cursor. ~16 preview updates/second.
constexpr double kScrubSeekInterval = 0.06;

// MPV-style auto-hide: fully visible for kControlsShowSeconds after the last
// input activity, then linearly fades to invisible over kControlsFadeSeconds.
constexpr double kControlsShowSeconds = 1.4;
constexpr double kControlsFadeSeconds = 0.6;

// Multiplies a color's existing alpha by `factor` (0..1), rather than
// replacing it like raylib's own Fade() does -- so the bar's own translucency
// (e.g. Fade(BLACK, 0.6f) for the backdrop) survives being additionally faded
// out by the auto-hide timer.
Color WithAlpha(Color c, float factor) {
    c.a = static_cast<unsigned char>(static_cast<float>(c.a) * factor);
    return c;
}

float ButtonRowY(int windowHeight, float height) {
    return static_cast<float>(windowHeight) - kBarHeight + (kBarHeight - height) * 0.5f;
}

Rectangle PlayButtonRect(int /*windowWidth*/, int windowHeight) {
    return Rectangle{kMargin, ButtonRowY(windowHeight, kButtonSize), kButtonSize, kButtonSize};
}

Rectangle LoopButtonRect(int /*windowWidth*/, int windowHeight) {
    const float x = kMargin + kButtonSize + kButtonSpacing;
    return Rectangle{x, ButtonRowY(windowHeight, kButtonSize), kButtonSize, kButtonSize};
}

Rectangle SeekBarRect(int windowWidth, int windowHeight) {
    const float left = kMargin + kButtonSize + kButtonSpacing + kButtonSize + kButtonSpacing;
    const float right = static_cast<float>(windowWidth) - kMargin - kTimeTextWidth;
    const float width = right > left ? right - left : 0.0f;
    const float y = static_cast<float>(windowHeight) - kBarHeight * 0.5f - kSeekBarThickness * 0.5f;
    return Rectangle{left, y, width, kSeekBarThickness};
}

// A circular-arrow "repeat" icon: a ~3/4 ring plus an arrowhead continuing
// the sweep, built entirely from cardinal (0/270 degree) points so it needs
// no trig beyond raylib's own DrawRing -- matches the play/pause icon's
// style of drawn primitives rather than text or an external asset.
void DrawLoopIcon(Rectangle button, Color color) {
    const Vector2 center{button.x + button.width * 0.5f, button.y + button.height * 0.5f};
    const float radius = button.height * 0.28f;
    DrawRing(center, radius - 3.0f, radius, 0.0f, 270.0f, 32, color);
    // Arrowhead at the ring's leading (270 degree, i.e. top) end, pointing
    // right to continue the clockwise sweep.
    const Vector2 tipBase{center.x, center.y - radius};
    const Vector2 p1{tipBase.x - 2.0f, tipBase.y - 5.0f};
    const Vector2 p2{tipBase.x - 2.0f, tipBase.y + 5.0f};
    const Vector2 p3{tipBase.x + 6.0f, tipBase.y};
    DrawTriangle(p1, p2, p3, color);
}

}  // namespace

bool PlayerControls::Update(PlaybackEngine& engine, int windowWidth, int windowHeight) {
    bool seeked = false;

    const bool spacePressed = IsKeyPressed(KEY_SPACE);
    const bool loopKeyPressed = IsKeyPressed(KEY_L);
    if (spacePressed) {
        engine.TogglePlayPause();
    }
    if (loopKeyPressed) {
        engine.ToggleLooping();
    }

    // Manual down/up edge tracking off the raw IsMouseButtonDown() state,
    // rather than raylib's own IsMouseButtonPressed()/Released() -- see
    // ui.hpp's class doc for why (some mouse utilities' input has been
    // reported to make GLFW's edge tracking miss clicks intermittently).
    const bool leftDown = IsMouseButtonDown(MOUSE_BUTTON_LEFT);
    const bool justPressed = leftDown && !leftButtonWasDown_;
    const bool justReleased = !leftDown && leftButtonWasDown_;
    leftButtonWasDown_ = leftDown;

    const Rectangle playButton = PlayButtonRect(windowWidth, windowHeight);
    const Rectangle loopButton = LoopButtonRect(windowWidth, windowHeight);
    const Rectangle seekBar = SeekBarRect(windowWidth, windowHeight);
    // A taller invisible hit area than the thin visual bar, so it's easy to
    // click without pixel-perfect aim.
    const Rectangle seekHitArea{seekBar.x, static_cast<float>(windowHeight) - kBarHeight,
                                seekBar.width, kBarHeight};
    const Vector2 mouse = GetMousePosition();

    // Any input activity resets the auto-hide timer (see ui.hpp's class doc):
    // mouse movement, a held/pressed button, or a keyboard shortcut. The
    // first call just establishes a baseline (and starts the bar visible on
    // load) rather than counting a "jump" from (0,0) as movement.
    const bool mouseMoved = mouseSeen_ && (mouse.x != lastMouseX_ || mouse.y != lastMouseY_);
    lastMouseX_ = mouse.x;
    lastMouseY_ = mouse.y;
    if (!mouseSeen_ || mouseMoved || leftDown || spacePressed || loopKeyPressed
        || IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_HOME)
        || IsKeyPressed(KEY_END)) {
        lastActivityTime_ = GetTime();
    }
    mouseSeen_ = true;

    // Press-arms / release-commits: the action fires on release only if the
    // cursor is still over the same button, and only if the press that
    // armed it started there too (so a drag from elsewhere onto a button,
    // released there, does nothing).
    if (justPressed) {
        playButtonArmed_ = CheckCollisionPointRec(mouse, playButton);
        loopButtonArmed_ = CheckCollisionPointRec(mouse, loopButton);
        if (CheckCollisionPointRec(mouse, seekHitArea)) {
            dragging_ = true;
        }
    }
    if (justReleased) {
        if (playButtonArmed_ && CheckCollisionPointRec(mouse, playButton)) {
            engine.TogglePlayPause();
        }
        if (loopButtonArmed_ && CheckCollisionPointRec(mouse, loopButton)) {
            engine.ToggleLooping();
        }
        if (dragging_) {
            // Commit the drop: land exactly on the release position (the
            // throttle may have skipped the last few mouse moves), and report a
            // seek so the caller flushes stale audio and resumes from here.
            const double fraction = SeekBarFractionForMouseX(mouse.x, seekBar.x, seekBar.width);
            scrubTargetPts_ = PtsForFraction(fraction, engine.Duration());
            engine.Seek(scrubTargetPts_);
            seeked = true;
        }
        playButtonArmed_ = false;
        loopButtonArmed_ = false;
        dragging_ = false;
    }

    if (dragging_ && leftDown) {
        const double fraction = SeekBarFractionForMouseX(mouse.x, seekBar.x, seekBar.width);
        scrubTargetPts_ = PtsForFraction(fraction, engine.Duration());
        // Throttle the actual seek/decode so it can't block the loop; the
        // playhead is drawn from scrubTargetPts_ every frame (see Draw), so the
        // bar stays glued to the cursor even as the preview catches up. Audio
        // is suspended by the caller for the whole drag (IsScrubbing()), not
        // flushed per seek -- so `seeked` is deliberately NOT set here.
        const double now = GetTime();
        if (now - lastScrubSeekTime_ >= kScrubSeekInterval) {
            engine.Seek(scrubTargetPts_);
            lastScrubSeekTime_ = now;
        }
    }

    std::uint64_t keyTarget = engine.CurrentPts();
    bool keySeek = false;
    if (IsKeyPressed(KEY_RIGHT)) {
        keyTarget += kKeySeekStepSamples;
        keySeek = true;
    }
    if (IsKeyPressed(KEY_LEFT)) {
        keyTarget = keyTarget > kKeySeekStepSamples ? keyTarget - kKeySeekStepSamples : 0;
        keySeek = true;
    }
    if (IsKeyPressed(KEY_HOME)) {
        keyTarget = 0;
        keySeek = true;
    }
    if (IsKeyPressed(KEY_END)) {
        keyTarget = engine.Duration();
        keySeek = true;
    }
    if (keySeek) {
        engine.Seek(keyTarget);
        seeked = true;
    }

    return seeked;
}

float PlayerControls::VisibilityAlpha() const {
    const double elapsed = GetTime() - lastActivityTime_;
    if (elapsed <= kControlsShowSeconds) {
        return 1.0f;
    }
    const double fadeElapsed = elapsed - kControlsShowSeconds;
    if (fadeElapsed >= kControlsFadeSeconds) {
        return 0.0f;
    }
    return 1.0f - static_cast<float>(fadeElapsed / kControlsFadeSeconds);
}

void PlayerControls::Draw(const PlaybackEngine& engine, int windowWidth, int windowHeight) const {
    const float alpha = VisibilityAlpha();
    if (alpha <= 0.0f) {
        return;
    }

    const float barTop = static_cast<float>(windowHeight) - kBarHeight;
    DrawRectangle(0, static_cast<int>(barTop), windowWidth, static_cast<int>(kBarHeight),
                  WithAlpha(Fade(BLACK, 0.6f), alpha));

    const Rectangle playButton = PlayButtonRect(windowWidth, windowHeight);
    DrawRectangleRec(playButton, WithAlpha(Fade(WHITE, 0.15f), alpha));
    if (engine.IsPlaying()) {
        DrawRectangle(static_cast<int>(playButton.x + 10), static_cast<int>(playButton.y + 8), 6,
                      24, WithAlpha(RAYWHITE, alpha));
        DrawRectangle(static_cast<int>(playButton.x + 24), static_cast<int>(playButton.y + 8), 6,
                      24, WithAlpha(RAYWHITE, alpha));
    } else {
        const Vector2 p1{playButton.x + 12, playButton.y + 8};
        const Vector2 p2{playButton.x + 12, playButton.y + 32};
        const Vector2 p3{playButton.x + 30, playButton.y + 20};
        DrawTriangle(p1, p2, p3, WithAlpha(RAYWHITE, alpha));
    }

    const Rectangle loopButton = LoopButtonRect(windowWidth, windowHeight);
    // Filled background when active, outline-only when not -- an
    // unambiguous on/off state independent of icon precision.
    DrawRectangleRec(loopButton, WithAlpha(
        engine.IsLooping() ? Fade(SKYBLUE, 0.45f) : Fade(WHITE, 0.15f), alpha));
    DrawRectangleLinesEx(loopButton, 1.0f,
                        WithAlpha(engine.IsLooping() ? SKYBLUE : Fade(WHITE, 0.4f), alpha));
    DrawLoopIcon(loopButton, WithAlpha(engine.IsLooping() ? WHITE : Fade(RAYWHITE, 0.7f), alpha));

    // Mid-drag, show the position the mouse is over (scrubTargetPts_), which
    // updates every frame, rather than CurrentPts() which only catches up at
    // the throttled preview rate -- so the playhead and time read-out track the
    // cursor with no lag.
    const std::uint64_t displayPts = dragging_ ? scrubTargetPts_ : engine.CurrentPts();

    const Rectangle seekBar = SeekBarRect(windowWidth, windowHeight);
    DrawRectangleRec(seekBar, WithAlpha(Fade(WHITE, 0.25f), alpha));
    const double fraction = FractionForPts(displayPts, engine.Duration());
    const auto filledWidth = static_cast<float>(fraction) * seekBar.width;
    if (filledWidth > 0.0f) {
        DrawRectangle(static_cast<int>(seekBar.x), static_cast<int>(seekBar.y),
                      static_cast<int>(filledWidth), static_cast<int>(seekBar.height),
                      WithAlpha(RAYWHITE, alpha));
    }
    DrawCircle(static_cast<int>(seekBar.x + filledWidth),
               static_cast<int>(seekBar.y + seekBar.height * 0.5f), 7.0f,
               WithAlpha(RAYWHITE, alpha));

    const std::string text =
        FormatTimecode(displayPts) + " / " + FormatTimecode(engine.Duration());
    const int textX = static_cast<int>(static_cast<float>(windowWidth) - kMargin - kTimeTextWidth
                                       + 8.0f);
    const int textY = static_cast<int>(barTop + (kBarHeight - 20.0f) * 0.5f);
    DrawText(text.c_str(), textX, textY, 18, WithAlpha(RAYWHITE, alpha));
}

}  // namespace ccmfplayer

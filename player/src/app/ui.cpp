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

    if (IsKeyPressed(KEY_SPACE)) {
        engine.TogglePlayPause();
    }
    if (IsKeyPressed(KEY_L)) {
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
        playButtonArmed_ = false;
        loopButtonArmed_ = false;
        dragging_ = false;
    }

    if (dragging_ && leftDown) {
        const double fraction = SeekBarFractionForMouseX(mouse.x, seekBar.x, seekBar.width);
        engine.Seek(PtsForFraction(fraction, engine.Duration()));
        seeked = true;
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

void PlayerControls::Draw(const PlaybackEngine& engine, int windowWidth, int windowHeight) const {
    const float barTop = static_cast<float>(windowHeight) - kBarHeight;
    DrawRectangle(0, static_cast<int>(barTop), windowWidth, static_cast<int>(kBarHeight),
                  Fade(BLACK, 0.6f));

    const Rectangle playButton = PlayButtonRect(windowWidth, windowHeight);
    DrawRectangleRec(playButton, Fade(WHITE, 0.15f));
    if (engine.IsPlaying()) {
        DrawRectangle(static_cast<int>(playButton.x + 10), static_cast<int>(playButton.y + 8), 6,
                      24, RAYWHITE);
        DrawRectangle(static_cast<int>(playButton.x + 24), static_cast<int>(playButton.y + 8), 6,
                      24, RAYWHITE);
    } else {
        const Vector2 p1{playButton.x + 12, playButton.y + 8};
        const Vector2 p2{playButton.x + 12, playButton.y + 32};
        const Vector2 p3{playButton.x + 30, playButton.y + 20};
        DrawTriangle(p1, p2, p3, RAYWHITE);
    }

    const Rectangle loopButton = LoopButtonRect(windowWidth, windowHeight);
    // Filled background when active, outline-only when not -- an
    // unambiguous on/off state independent of icon precision.
    DrawRectangleRec(loopButton, engine.IsLooping() ? Fade(SKYBLUE, 0.45f) : Fade(WHITE, 0.15f));
    DrawRectangleLinesEx(loopButton, 1.0f, engine.IsLooping() ? SKYBLUE : Fade(WHITE, 0.4f));
    DrawLoopIcon(loopButton, engine.IsLooping() ? WHITE : Fade(RAYWHITE, 0.7f));

    const Rectangle seekBar = SeekBarRect(windowWidth, windowHeight);
    DrawRectangleRec(seekBar, Fade(WHITE, 0.25f));
    const double fraction = FractionForPts(engine.CurrentPts(), engine.Duration());
    const auto filledWidth = static_cast<float>(fraction) * seekBar.width;
    if (filledWidth > 0.0f) {
        DrawRectangle(static_cast<int>(seekBar.x), static_cast<int>(seekBar.y),
                      static_cast<int>(filledWidth), static_cast<int>(seekBar.height), RAYWHITE);
    }
    DrawCircle(static_cast<int>(seekBar.x + filledWidth),
               static_cast<int>(seekBar.y + seekBar.height * 0.5f), 7.0f, RAYWHITE);

    const std::string text =
        FormatTimecode(engine.CurrentPts()) + " / " + FormatTimecode(engine.Duration());
    const int textX = static_cast<int>(static_cast<float>(windowWidth) - kMargin - kTimeTextWidth
                                       + 8.0f);
    const int textY = static_cast<int>(barTop + (kBarHeight - 20.0f) * 0.5f);
    DrawText(text.c_str(), textX, textY, 18, RAYWHITE);
}

}  // namespace ccmfplayer

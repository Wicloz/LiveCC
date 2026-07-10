#pragma once

#include "engine/engine.hpp"

namespace ccmfplayer {

// Draws and drives the on-screen transport controls: a play/pause button, a
// loop toggle button, a draggable scrub bar, and a current/total time
// readout, plus keyboard shortcuts (Space, L, Left/Right +-5s, Home/End).
// All the underlying hit-testing/fraction math lives in controls_math.hpp
// (pure, unit-tested); this class is just the raylib input/drawing glue
// around it, which is why it isn't itself unit-tested -- see the project
// plan's Stage 7 notes.
//
// Button clicks are detected with our own down/up edge tracking off
// IsMouseButtonDown() (press arms a button, release while still hovering it
// commits the action) rather than raylib's IsMouseButtonPressed/Released,
// which some mouse utilities' synthesized input has been reported to make
// GLFW miss intermittently.
//
// The bar auto-hides MPV-style: it sits over the bottom of the video rather
// than reserving its own space (so it doesn't shrink the video area), and
// would otherwise permanently cover part of the picture whenever the video's
// own aspect ratio leaves no letterbox bar at the bottom to hide in. Instead
// it's fully visible only for a bit after the last mouse move/click/key
// press, then fades out -- see kControlsShowSeconds/kControlsFadeSeconds in
// ui.cpp.
class PlayerControls {
public:
    // Reads mouse/keyboard input and acts on `engine` (Play/Pause/Seek/loop)
    // directly. Returns true if a seek (including a loop restart) happened
    // this call, so the caller can flush any audio output that might
    // otherwise replay a stale pre-seek snippet.
    bool Update(PlaybackEngine& engine, int windowWidth, int windowHeight);

    // Draws the controls; call after the video so they render on top of it.
    // A no-op once the auto-hide fade has reached zero.
    void Draw(const PlaybackEngine& engine, int windowWidth, int windowHeight) const;

private:
    float VisibilityAlpha() const;

    bool leftButtonWasDown_ = false;
    bool playButtonArmed_ = false;
    bool loopButtonArmed_ = false;
    bool dragging_ = false;

    double lastActivityTime_ = 0.0;
    float lastMouseX_ = 0.0f;
    float lastMouseY_ = 0.0f;
    bool mouseSeen_ = false;
};

}  // namespace ccmfplayer

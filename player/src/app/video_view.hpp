#pragma once

#include <cstdint>
#include <vector>

#include <raylib.h>

#include "engine/render.hpp"
#include "engine/video.hpp"

namespace ccmfplayer {

// Owns the GPU texture a decoded video frame is uploaded into, and draws it
// scaled/letterboxed to fit an arbitrary window size. One instance per
// playing file -- the texture is sized to that file's grid at construction
// and never resized.
class VideoView {
public:
    VideoView(std::uint16_t gridWidth, std::uint16_t gridHeight);
    ~VideoView();

    VideoView(const VideoView&) = delete;
    VideoView& operator=(const VideoView&) = delete;

    // Re-renders `frame` into the CPU pixel buffer and uploads it to the GPU
    // texture. A no-op if `frame` is nullptr, so the last successfully
    // uploaded frame just stays on screen (e.g. while paused, or while the
    // engine's HasError() means nothing new decoded this tick).
    void Update(const Frame* frame);

    // Draws the current texture centered in a `windowWidth` x `windowHeight`
    // viewport, scaled to fit while preserving aspect ratio (letterboxed),
    // with point/nearest filtering so the cell dither pattern stays crisp
    // rather than blurring under magnification.
    void Draw(int windowWidth, int windowHeight) const;

private:
    std::uint16_t gridWidth_;
    std::uint16_t gridHeight_;
    std::vector<std::uint8_t> pixels_;
    Texture2D texture_{};
};

}  // namespace ccmfplayer

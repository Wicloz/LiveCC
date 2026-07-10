#pragma once

#include <cstdint>
#include <vector>

#include <raylib.h>

#include "engine/render.hpp"
#include "engine/video.hpp"

namespace ccmfplayer {

// Owns the GPU texture a decoded video frame is uploaded into, and draws it
// scaled/letterboxed to fit an arbitrary window size. The texture is sized to a
// specific grid at construction; because a file MAY change resolution mid-stream
// (spec 4.4), the owner re-creates the VideoView when the engine's grid size
// changes (compare GridWidth()/GridHeight() to the engine's Width()/Height()).
class VideoView {
public:
    VideoView(std::uint16_t gridWidth, std::uint16_t gridHeight);
    ~VideoView();

    VideoView(const VideoView&) = delete;
    VideoView& operator=(const VideoView&) = delete;

    [[nodiscard]] std::uint16_t GridWidth() const noexcept { return gridWidth_; }
    [[nodiscard]] std::uint16_t GridHeight() const noexcept { return gridHeight_; }

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

#include "app/video_view.hpp"

#include <algorithm>

namespace ccmfplayer {

namespace {
int PixelWidth(std::uint16_t gridWidth) { return static_cast<int>(gridWidth) * 2; }
int PixelHeight(std::uint16_t gridHeight) { return static_cast<int>(gridHeight) * 3; }
}  // namespace

VideoView::VideoView(std::uint16_t gridWidth, std::uint16_t gridHeight)
    : gridWidth_(gridWidth),
      gridHeight_(gridHeight),
      pixels_(RenderedPixelBufferSize(gridWidth, gridHeight), 0) {
    Image image{};
    image.data = pixels_.data();
    image.width = PixelWidth(gridWidth_);
    image.height = PixelHeight(gridHeight_);
    image.mipmaps = 1;
    image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8;
    texture_ = LoadTextureFromImage(image);
    SetTextureFilter(texture_, TEXTURE_FILTER_POINT);
}

VideoView::~VideoView() {
    UnloadTexture(texture_);
}

void VideoView::Update(const Frame* frame) {
    if (frame == nullptr) {
        return;
    }
    RenderCellsToRgb(*frame, gridWidth_, gridHeight_, pixels_);
    UpdateTexture(texture_, pixels_.data());
}

void VideoView::Draw(int windowWidth, int windowHeight) const {
    const auto texW = static_cast<float>(texture_.width);
    const auto texH = static_cast<float>(texture_.height);
    const float scale = std::min(static_cast<float>(windowWidth) / texW,
                                 static_cast<float>(windowHeight) / texH);
    const float drawW = texW * scale;
    const float drawH = texH * scale;
    const float x = (static_cast<float>(windowWidth) - drawW) * 0.5f;
    const float y = (static_cast<float>(windowHeight) - drawH) * 0.5f;

    const Rectangle src{0.0f, 0.0f, texW, texH};
    const Rectangle dst{x, y, drawW, drawH};
    DrawTexturePro(texture_, src, dst, Vector2{0.0f, 0.0f}, 0.0f, WHITE);
}

}  // namespace ccmfplayer

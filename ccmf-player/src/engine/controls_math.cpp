#include "engine/controls_math.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "engine/chunk.hpp"

namespace ccmfplayer {

double SeekBarFractionForMouseX(double mouseX, double barLeft, double barWidth) noexcept {
    if (barWidth <= 0.0) {
        return 0.0;
    }
    return std::clamp((mouseX - barLeft) / barWidth, 0.0, 1.0);
}

std::uint64_t PtsForFraction(double fraction, std::uint64_t duration) noexcept {
    const double clamped = std::clamp(fraction, 0.0, 1.0);
    return static_cast<std::uint64_t>(clamped * static_cast<double>(duration));
}

double FractionForPts(std::uint64_t pts, std::uint64_t duration) noexcept {
    if (duration == 0) {
        return 0.0;
    }
    const double fraction = static_cast<double>(pts) / static_cast<double>(duration);
    return std::clamp(fraction, 0.0, 1.0);
}

std::string FormatTimecode(std::uint64_t pts) {
    const std::uint64_t totalSeconds = pts / kSampleRate;
    const std::uint64_t hours = totalSeconds / 3600;
    const std::uint64_t minutes = (totalSeconds % 3600) / 60;
    const std::uint64_t seconds = totalSeconds % 60;

    std::ostringstream out;
    if (hours > 0) {
        out << hours << ':' << std::setw(2) << std::setfill('0') << minutes << ':'
            << std::setw(2) << std::setfill('0') << seconds;
    } else {
        out << minutes << ':' << std::setw(2) << std::setfill('0') << seconds;
    }
    return out.str();
}

int TargetFpsForFrameDuration(std::uint16_t durationSamples, int fallbackFps) noexcept {
    if (durationSamples == 0) {
        return fallbackFps;
    }
    const double fps = static_cast<double>(kSampleRate) / static_cast<double>(durationSamples);
    const int rounded = static_cast<int>(fps + 0.5);
    return std::clamp(rounded, 1, 240);
}

}  // namespace ccmfplayer

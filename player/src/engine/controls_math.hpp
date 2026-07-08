#pragma once

#include <cstdint>
#include <string>

namespace ccmfplayer {

// Maps a mouse x-coordinate to a [0,1] fraction along a horizontal seek bar
// spanning [barLeft, barLeft+barWidth), clamped at both ends -- a click
// before the bar's start seeks to 0, a click past its end seeks to the end.
// Returns 0 for a degenerate (zero or negative width) bar rather than
// dividing by zero.
[[nodiscard]] double SeekBarFractionForMouseX(double mouseX, double barLeft,
                                              double barWidth) noexcept;

// Converts a [0,1] fraction (clamped) of a file's total `duration` (spec 4.2
// samples) to an absolute PTS.
[[nodiscard]] std::uint64_t PtsForFraction(double fraction, std::uint64_t duration) noexcept;

// The inverse of PtsForFraction: how far into `duration` a given `pts` is,
// as a [0,1] fraction (clamped; 0 for a zero-length file rather than
// dividing by zero).
[[nodiscard]] double FractionForPts(std::uint64_t pts, std::uint64_t duration) noexcept;

// Formats a PTS (spec 4.2 samples) as "m:ss", or "h:mm:ss" once past an
// hour (with both minutes and seconds zero-padded in that form, matching
// common media-player convention).
[[nodiscard]] std::string FormatTimecode(std::uint64_t pts);

// Converts a frame's hold duration (spec 4.5's `duration`, 48 kHz samples)
// to a render-loop target frame rate: the app paces its window (and, with
// it, how often raylib polls input) to match whatever the *currently
// playing* video content actually needs, rather than a fixed guess -- a
// file's chunks aren't required to share one frame rate (spec 4.5: duration
// is per-frame), so this is meant to be recomputed from CurrentFrame()
// every tick, not cached across a chunk boundary.
//
// Returns `fallbackFps` for a zero duration (shouldn't happen for a real
// frame, but a zero would otherwise divide by zero), and clamps the result
// to [1, 240] so a pathological near-zero or huge duration value can't ask
// the window for an unusable target.
[[nodiscard]] int TargetFpsForFrameDuration(std::uint16_t durationSamples,
                                            int fallbackFps) noexcept;

}  // namespace ccmfplayer

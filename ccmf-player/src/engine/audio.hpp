#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "engine/chunk.hpp"

namespace ccmfplayer {

// Audio codec values (a-hdr high nibble, spec 4.6).
inline constexpr std::uint8_t kAudioCodecPcm8 = 0;
inline constexpr std::uint8_t kAudioCodecDfpwm = 1;

// Channel role values (a-hdr low nibble; shared numbering with the CAPS
// `channels` bitmask, spec 4.6/5.4).
inline constexpr std::uint8_t kChannelMono = 0;
inline constexpr std::uint8_t kChannelFrontLeft = 1;
inline constexpr std::uint8_t kChannelFrontRight = 2;
inline constexpr std::uint8_t kChannelCenter = 3;
inline constexpr std::uint8_t kChannelLfe = 4;
inline constexpr std::uint8_t kChannelSurroundLeft = 5;
inline constexpr std::uint8_t kChannelSurroundRight = 6;
inline constexpr std::uint8_t kChannelRearLeft = 7;
inline constexpr std::uint8_t kChannelRearRight = 8;

// One decoded audio chunk payload: its channel role and codec (spec 4.6's
// a-hdr) plus the samples, always normalized to signed 16-bit PCM regardless
// of the wire codec (scaled from the underlying unsigned 8-bit amplitude via
// ScaleU8ToI16).
struct DecodedAudio {
    std::uint8_t codec = 0;
    std::uint8_t channel = 0;
    std::vector<std::int16_t> pcm;
};

// Converts unsigned 8-bit PCM (spec 4.6's pcm8: amplitude = byte - 128, the
// native ComputerCraft speaker format) to signed 16-bit PCM.
[[nodiscard]] std::vector<std::int16_t> DecodePcm8ToI16(std::span<const std::byte> data);

// Decodes DFPWM1a (spec 4.6) to unsigned 8-bit PCM, with FRESH decoder state
// every call -- the spec requires the decoder to reset at the start of every
// chunk (so each chunk is independently decodable), matching
// server/dfpwm.py's decode(), which this is a byte-exact port of.
[[nodiscard]] std::vector<std::uint8_t> DecodeDfpwmToU8(std::span<const std::byte> data);

// Scales unsigned 8-bit PCM (amplitude = byte - 128) to signed 16-bit PCM.
// Shared by both codec paths: PCM8 is this format already; DFPWM decodes to
// it first (DecodeDfpwmToU8) and is then scaled the same way.
[[nodiscard]] std::vector<std::int16_t> ScaleU8ToI16(std::span<const std::uint8_t> u8);

// Decodes one audio chunk payload (spec 4.6): [a-hdr u8][samples]. Throws
// CcmfError on an empty payload or an unsupported codec.
[[nodiscard]] DecodedAudio DecodeAudioPayload(std::span<const std::byte> payload);

}  // namespace ccmfplayer

#include "engine/audio.hpp"

#include <algorithm>
#include <string>

namespace ccmfplayer {

namespace {

// DFPWM1a predictor constants (cc.audio.dfpwm), spec 4.6 / [DFPWM].
constexpr std::int32_t kPrecPow = 1 << 10;       // PREC = 10
constexpr std::int32_t kPrecPowHalf = 1 << 9;
constexpr std::int32_t kStrengthMin = 1 << 3;    // 2^(PREC - 8 + 1)

}  // namespace

std::vector<std::int16_t> ScaleU8ToI16(std::span<const std::uint8_t> u8) {
    std::vector<std::int16_t> out(u8.size());
    for (std::size_t i = 0; i < u8.size(); ++i) {
        out[i] = static_cast<std::int16_t>((static_cast<int>(u8[i]) - 128) * 256);
    }
    return out;
}

std::vector<std::int16_t> DecodePcm8ToI16(std::span<const std::byte> data) {
    std::vector<std::uint8_t> u8(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        u8[i] = std::to_integer<std::uint8_t>(data[i]);
    }
    return ScaleU8ToI16(u8);
}

// Bit-exact port of CC:Tweaked's cc.audio.dfpwm decoder (cross-checked against
// server/dfpwm.py, itself validated against ffmpeg sample-for-sample). C++20
// guarantees signed right shift is arithmetic (floor toward -infinity,
// [expr.shift]), matching the reference `>>` on ints exactly, so the predictor
// arithmetic ports directly with no sign-handling differences; every
// intermediate stays well within int32_t range (charge/antijerk/low_pass hover
// near +-128, strength within [8, 1023]), so there's no overflow concern.
//
// The running low_pass accumulator IS the decoder state and is deliberately
// never clamped between steps -- only each emitted sample is clamped to
// [-128,127] before the +128 bias. That lets the clamp happen inline as we
// write straight into the uint8 output: one pass, one allocation, no int32
// scratch buffer (the old two-pass form materialized an 8x-size int32 array
// just to re-scan it here).
std::vector<std::uint8_t> DecodeDfpwmToU8(std::span<const std::byte> data) {
    std::vector<std::uint8_t> out(data.size() * 8);

    std::int32_t charge = 0;
    std::int32_t strength = 0;
    bool prevBit = false;
    std::int32_t prevCharge = 0;
    std::int32_t lowPass = 0;

    std::size_t o = 0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        std::uint8_t byteVal = std::to_integer<std::uint8_t>(data[i]);
        for (int j = 0; j < 8; ++j) {
            const bool bit = (byteVal & 1) != 0;
            byteVal = static_cast<std::uint8_t>(byteVal >> 1);

            const std::int32_t target = bit ? 127 : -128;
            std::int32_t nextCharge =
                charge + ((strength * (target - charge) + kPrecPowHalf) >> 10);
            if (nextCharge == charge && nextCharge != target) {
                nextCharge += bit ? 1 : -1;
            }
            const std::int32_t z = (bit == prevBit) ? (kPrecPow - 1) : 0;
            std::int32_t nextStrength = strength;
            if (nextStrength != z) {
                nextStrength += (bit == prevBit) ? 1 : -1;
            }
            if (nextStrength < kStrengthMin) {
                nextStrength = kStrengthMin;
            }

            std::int32_t antijerk = nextCharge;
            if (bit != prevBit) {
                antijerk = (nextCharge + prevCharge + 1) >> 1;
            }

            charge = nextCharge;
            strength = nextStrength;
            prevBit = bit;
            prevCharge = nextCharge;

            lowPass += ((antijerk - lowPass) * 140 + 128) >> 8;
            const std::int32_t clamped =
                std::clamp(lowPass, std::int32_t{-128}, std::int32_t{127});
            out[o++] = static_cast<std::uint8_t>(clamped + 128);
        }
    }
    return out;
}

std::vector<std::uint8_t> OutputRolesForChannelCount(std::uint32_t channelCount) {
    if (channelCount <= 1) {
        return {kChannelMono};
    }
    return {kChannelFrontLeft, kChannelFrontRight};
}

namespace {
bool Contains(const std::vector<std::uint8_t>& roles, std::uint8_t role) {
    return std::find(roles.begin(), roles.end(), role) != roles.end();
}
}  // namespace

std::vector<std::uint8_t> ResolveOutputSources(std::uint8_t outputRole,
                                               const std::vector<std::uint8_t>& present) {
    if (present.empty()) {
        return {};  // no audio here at all -> silence
    }
    if (Contains(present, outputRole)) {
        return {outputRole};  // exact match: pass through
    }

    switch (outputRole) {
        case kChannelMono:
            // Prefer a proper L+R downmix; otherwise the generic fallback below
            // averages whatever is present (ultimately the mono mix).
            if (Contains(present, kChannelFrontLeft) && Contains(present, kChannelFrontRight)) {
                return {kChannelFrontLeft, kChannelFrontRight};
            }
            break;
        case kChannelFrontLeft:
            if (Contains(present, kChannelMono)) return {kChannelMono};
            if (Contains(present, kChannelFrontRight)) return {kChannelFrontRight};
            break;
        case kChannelFrontRight:
            if (Contains(present, kChannelMono)) return {kChannelMono};
            if (Contains(present, kChannelFrontLeft)) return {kChannelFrontLeft};
            break;
        default:
            if (Contains(present, kChannelMono)) return {kChannelMono};
            break;
    }

    // Generic fallback: average every present role (a full downmix). Guarantees
    // a mapped output channel always plays rather than going silent.
    return present;
}

DecodedAudio DecodeAudioPayload(std::span<const std::byte> payload) {
    if (payload.empty()) {
        throw CcmfError("empty audio payload");
    }
    const auto header = std::to_integer<std::uint8_t>(payload[0]);

    DecodedAudio out;
    out.codec = static_cast<std::uint8_t>(header >> 4);
    out.channel = static_cast<std::uint8_t>(header & 0x0F);

    const auto samples = payload.subspan(1);
    if (out.codec == kAudioCodecPcm8) {
        out.pcm = DecodePcm8ToI16(samples);
    } else if (out.codec == kAudioCodecDfpwm) {
        out.pcm = ScaleU8ToI16(DecodeDfpwmToU8(samples));
    } else {
        throw CcmfError("unsupported audio codec " + std::to_string(out.codec));
    }
    return out;
}

}  // namespace ccmfplayer

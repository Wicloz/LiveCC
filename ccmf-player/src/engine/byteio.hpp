#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace ccmfplayer {

// Reads a little-endian unsigned integer of `NumBytes` bytes from the start
// of `buf` (spec 3: "all multi-byte integers are unsigned little-endian").
// `buf` must contain at least `NumBytes` bytes -- this is a low-level
// primitive used only after a caller has already bound-checked the buffer
// (spec 7 requires that check to happen before any field is trusted), so it
// intentionally has no failure mode of its own.
template <std::size_t NumBytes>
[[nodiscard]] constexpr std::uint64_t ReadLittleEndian(std::span<const std::byte> buf) noexcept {
    static_assert(NumBytes >= 1 && NumBytes <= 8, "NumBytes must fit in a uint64_t");
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < NumBytes; ++i) {
        value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(buf[i])) << (8 * i);
    }
    return value;
}

}  // namespace ccmfplayer

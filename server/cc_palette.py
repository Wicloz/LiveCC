"""
CC Tweaked palette + 2x3 sub-pixel frame renderer.

Each monitor character cell is drawn as a 2-wide x 3-tall grid of sub-pixels
using CC's "teletext" drawing characters (codes 0x80-0x9F).  A cell carries a
foreground and a background colour; the glyph selects which of the 6 sub-pixels
show fg vs bg.  So a W x H character monitor renders (W*2) x (H*3) pixels.

Sub-pixel bit layout within a cell (value in parentheses):
    top-left (1)     top-right (2)
    mid-left (4)     mid-right (8)
    bot-left (16)    bot-right (control)

Only 5 bits map to a character (0x80 + bits).  The bottom-right sub-pixel is
always the background colour; if it should be foreground we invert the other
five bits and swap fg/bg.

Blit colour order (index -> blit char '0'..'f') matches CC's default palette.
"""

from __future__ import annotations

import numpy as np

# RGB values matching CC Tweaked's default palette, indexed 0..15.
_CC_RGB = np.array([
    (240, 240, 240),  # 0  white
    (242, 178,  51),  # 1  orange
    (229, 127, 216),  # 2  magenta
    (153, 178, 242),  # 3  light_blue
    (222, 222, 108),  # 4  yellow
    (127, 204,  25),  # 5  lime
    (242, 178, 204),  # 6  pink
    ( 76,  76,  76),  # 7  gray
    (153, 153, 153),  # 8  light_gray
    ( 76, 153, 178),  # 9  cyan
    (178, 102, 229),  # a  purple
    ( 51, 102, 204),  # b  blue
    (127, 102,  76),  # c  brown
    ( 87, 166,  78),  # d  green
    (204,  76,  76),  # e  red
    ( 17,  17,  17),  # f  black
], dtype=np.float32)

# ASCII bytes for blit chars: b'0123456789abcdef'
_BLIT_LUT = np.frombuffer(b"0123456789abcdef", dtype=np.uint8)

# Bit weights for the first five sub-pixels (top-left..bottom-left).
_BITS = np.array([1, 2, 4, 8, 16], dtype=np.uint8)


# Channel precision of the precomputed nearest-colour LUT.  6 bits (input snapped
# to steps of 4) is plenty: CC's 16 palette colours sit far enough apart that a
# ±4 input change almost never flips which one is nearest, so 8 bits would 64x the
# table (16 MiB vs 256 KiB) and the build cost for no visible gain.  The builder
# is chunked, so 8 bits is *viable* (set _BITS_PER_CHANNEL = 8) — just not worth it.
_BITS_PER_CHANNEL = 6
_SHIFT = 8 - _BITS_PER_CHANNEL

# Chroma weight in the CIELAB nearest-colour search: distance = ΔL² + w·(Δa²+Δb²).
# Equal weight (w=1) is "correct" perceptual ΔE, but CC's palette has only one
# light colour (white) and a big lightness gap below it, so plain ΔE collapses
# every light, low-chroma pixel to white — light blues, pale tints all wash out.
# Up-weighting chroma pulls those back toward their hue.  It's safe for neutrals
# (black/gray/light_gray/white are all chroma-zero, so lightness still separates
# them — the grey ramp is unchanged) and preserves the palette round-trip.
# Higher = more colourful / less white (w≈6 ≈ the old RGB look); 1 = literal ΔE.
_CHROMA_WEIGHT = 4.0


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (last axis = R,G,B in 0..255) to CIELAB (D65 white point)."""
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([                                   # linear sRGB -> XYZ (D65)
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = (lin @ m.T) / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    return np.stack([
        116 * f[..., 1] - 16,            # L*
        500 * (f[..., 0] - f[..., 1]),   # a*
        200 * (f[..., 1] - f[..., 2]),   # b*
    ], axis=-1)


def _build_lut() -> np.ndarray:
    """Pre-compute a table mapping reduced-precision RGB -> nearest palette index.

    "Nearest" is CIELAB ΔE (perceptual), not RGB Euclidean distance.  In RGB a
    luminance gap counts the same as a chroma gap, so neutral greys map to
    chromatic entries — light greys (~190-210) land on pink, mid greys on brown —
    because CC's neutral colours are sparse along the brightness axis and pink /
    brown sit in the gaps.  Lab keeps a neutral grey's a*/b* near 0, so it snaps to
    black / grey / light-grey / white as expected.

    Built one R-slice at a time so peak memory stays small even at 8-bit.
    """
    n = 1 << _BITS_PER_CHANNEL
    step = 1 << _SHIFT
    # Reconstruct each bucket at its CENTRE, not its floor.  quantize() indexes
    # with a plain >>_SHIFT (floor), so bucket i covers inputs [i*step, i*step +
    # step-1]; building the grid at i*step biased every lookup ~step/2 low.  The
    # (step-1)/2 offset removes that bias at build time — no per-frame add/clamp,
    # unlike rounding before the shift.  For 8-bit step=1, offset=0 (already exact).
    levels = np.arange(n, dtype=np.float32) * step + (step - 1) / 2.0
    pal_lab = _srgb_to_lab(_CC_RGB)                           # (16, 3)
    gg, bb = np.meshgrid(levels, levels, indexing="ij")      # (n, n) each
    lut = np.empty((n, n, n), dtype=np.uint8)
    for ri in range(n):
        rr = np.full((n, n), levels[ri], dtype=np.float32)
        lab = _srgb_to_lab(np.stack([rr, gg, bb], axis=-1))  # (n, n, 3)
        diff = lab[:, :, None, :] - pal_lab                  # (n, n, 16, 3)
        d = diff[..., 0] ** 2 + _CHROMA_WEIGHT * (diff[..., 1:] ** 2).sum(-1)
        lut[ri] = np.argmin(d, axis=2).astype(np.uint8)
    return lut


_LUT: np.ndarray = _build_lut()


def quantize(frame_rgb: np.ndarray) -> np.ndarray:
    """Map an H x W x 3 uint8 RGB array to an H x W uint8 array of palette indices."""
    r = frame_rgb[:, :, 0] >> _SHIFT
    g = frame_rgb[:, :, 1] >> _SHIFT
    b = frame_rgb[:, :, 2] >> _SHIFT
    return _LUT[r, g, b]


def encode_frame(frame_rgb: np.ndarray) -> bytes:
    """
    Encode a (H*3) x (W*2) x 3 RGB frame into the binary wire format:

        bytes 0-1 : W (uint16 big-endian, character columns)
        bytes 2-3 : H (uint16 big-endian, character rows)
        then H rows, each: W text bytes (0x80-0x9F) + W fg + W bg (ASCII hex)

    A row's three strings are exactly what mon.blit(text, fg, bg) expects.
    """
    ph, pw, _ = frame_rgb.shape
    h_cells = ph // 3
    w_cells = pw // 2

    idx = quantize(frame_rgb)                                  # (ph, pw)
    # Group into cells -> (H, W, 6); sub-pixel order s = sub_row*2 + sub_col.
    cells = (idx[: h_cells * 3, : w_cells * 2]
             .reshape(h_cells, 3, w_cells, 2)
             .transpose(0, 2, 1, 3)
             .reshape(h_cells, w_cells, 6))

    # Pick the two dominant palette colours per cell (mode + runner-up).
    onehot = cells[..., None] == np.arange(16)
    counts = onehot.sum(axis=2).astype(np.int16)               # (H, W, 16)
    fg_idx = counts.argmax(axis=2)                             # (H, W)
    counts2 = counts.copy()
    np.put_along_axis(counts2, fg_idx[..., None], -1, axis=2)
    bg_idx = counts2.argmax(axis=2)                            # (H, W)

    # Assign each sub-pixel to whichever of the two colours is nearer in RGB.
    cell_rgb = _CC_RGB[cells]                                  # (H, W, 6, 3)
    d_fg = ((cell_rgb - _CC_RGB[fg_idx][:, :, None, :]) ** 2).sum(-1)
    d_bg = ((cell_rgb - _CC_RGB[bg_idx][:, :, None, :]) ** 2).sum(-1)
    is_fg = d_fg <= d_bg                                       # (H, W, 6)

    invert = is_fg[..., 5]                                     # bottom-right is fg
    bits = is_fg[..., :5].astype(np.uint8)                     # (H, W, 5)
    data = (bits * _BITS).sum(axis=2).astype(np.uint8)
    data_inv = ((1 - bits) * _BITS).sum(axis=2).astype(np.uint8)

    char = (0x80 + np.where(invert, data_inv, data)).astype(np.uint8)
    final_fg = np.where(invert, bg_idx, fg_idx)
    final_bg = np.where(invert, fg_idx, bg_idx)

    rows = np.empty((h_cells, 3, w_cells), dtype=np.uint8)
    rows[:, 0, :] = char
    rows[:, 1, :] = _BLIT_LUT[final_fg]
    rows[:, 2, :] = _BLIT_LUT[final_bg]

    header = bytes((
        (w_cells >> 8) & 0xFF, w_cells & 0xFF,
        (h_cells >> 8) & 0xFF, h_cells & 0xFF,
    ))
    return header + rows.tobytes()

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
    (76,  76,  76),  # 7  gray
    (153, 153, 153),  # 8  light_gray
    (76, 153, 178),  # 9  cyan
    (178, 102, 229),  # a  purple
    (51, 102, 204),  # b  blue
    (127, 102,  76),  # c  brown
    (87, 166,  78),  # d  green
    (204,  76,  76),  # e  red
    (17,  17,  17),  # f  black
], dtype=np.float32)

# ASCII bytes for blit chars: b'0123456789abcdef'
_BLIT_LUT = np.frombuffer(b"0123456789abcdef", dtype=np.uint8)

# Bit weights for the first five sub-pixels (top-left..bottom-left).
_BITS = np.array([1, 2, 4, 8, 16], dtype=np.uint8)


# Channel precision of the precomputed nearest-colour LUT.  6 bits/channel keeps
# the table compact enough for reasonable cache behavior while preserving the
# colour discrimination needed by the blit search.
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
_CHROMA_WEIGHT = 6


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


def _build_dist_lut() -> np.ndarray:
    """Pre-compute perceptual distance to every palette entry for each RGB bucket."""
    n = 1 << _BITS_PER_CHANNEL
    step = 1 << _SHIFT
    levels = np.arange(n, dtype=np.float32) * step + (step - 1) / 2.0
    pal_lab = _srgb_to_lab(_CC_RGB)                           # (16, 3)
    gg, bb = np.meshgrid(levels, levels, indexing="ij")
    lut = np.empty((n, n, n, 16), dtype=np.float32)
    for ri in range(n):
        rr = np.full((n, n), levels[ri], dtype=np.float32)
        lab = _srgb_to_lab(np.stack([rr, gg, bb], axis=-1))  # (n, n, 3)
        diff = lab[:, :, None, :] - pal_lab                  # (n, n, 16, 3)
        lut[ri] = diff[..., 0] ** 2 + _CHROMA_WEIGHT * (diff[..., 1:] ** 2).sum(-1)
    return lut


_DIST_LUT: np.ndarray = _build_dist_lut()

# Ordered foreground/background pair tables.  The blit representation is
# directional because the bottom-right sub-pixel must be background in the
# emitted wire format, even though swapping FG/BG and inverting the mask is
# mathematically equivalent.
_PAIR_FG = np.repeat(np.arange(16, dtype=np.uint8), 16)
_PAIR_BG = np.tile(np.arange(16, dtype=np.uint8), 16)
_MASK_INVERT = np.uint8(31) - np.arange(32, dtype=np.uint8)

# Only compare blit pairs built from the best few palette candidates per cell.
# This keeps the output format unchanged while cutting the search cost sharply.
_PAIR_CANDIDATES = 2

# Precompute the full canonical 8192-state blit codebook: every foreground /
# background pair and every legal 5-bit mask.  The output state space is tiny,
# so we can store the canonical glyph byte and the final fg/bg bytes directly.
_STATE_COUNT = 16 * 16 * 32
_STATE_INDEX = np.empty((16, 16, 32), dtype=np.uint16)
_STATE_GLYPH = np.empty(_STATE_COUNT, dtype=np.uint8)
_STATE_FG = np.empty(_STATE_COUNT, dtype=np.uint8)
_STATE_BG = np.empty(_STATE_COUNT, dtype=np.uint8)
_STATE_MASK = np.empty(_STATE_COUNT, dtype=np.uint8)
_STATE_RENDERED = np.empty((_STATE_COUNT, 6), dtype=np.uint8)

_state = 0
for fg in range(16):
    for bg in range(16):
        for mask in range(32):
            _STATE_INDEX[fg, bg, mask] = np.uint16(_state)
            _STATE_GLYPH[_state] = np.uint8(0x80 + mask)
            _STATE_FG[_state] = np.uint8(fg)
            _STATE_BG[_state] = np.uint8(bg)
            _STATE_MASK[_state] = np.uint8(mask)
            rendered = np.full(6, bg, dtype=np.uint8)
            for bit, pixel in enumerate((0, 1, 2, 3, 4)):
                if mask & (1 << bit):
                    rendered[pixel] = np.uint8(fg)
            _STATE_RENDERED[_state] = rendered
            _state += 1

_PAIR_CHUNK = 64


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

    frame_rgb = np.asarray(frame_rgb, dtype=np.uint8)
    r = frame_rgb[: h_cells * 3, : w_cells * 2, 0] >> _SHIFT
    g = frame_rgb[: h_cells * 3, : w_cells * 2, 1] >> _SHIFT
    b = frame_rgb[: h_cells * 3, : w_cells * 2, 2] >> _SHIFT

    # Group into cells -> (H, W, 6, 16); sub-pixel order s = sub_row*2 + sub_col.
    cells_dist = (_DIST_LUT[r, g, b]
                  .reshape(h_cells, 3, w_cells, 2, 16)
                  .transpose(0, 2, 1, 3, 4)
                  .reshape(h_cells, w_cells, 6, 16))

    # Evaluate every ordered FG/BG pair built from the top-k palette candidates.
    cell_cost = cells_dist.sum(axis=2)                         # (H, W, 16)
    cand = np.argpartition(cell_cost, _PAIR_CANDIDATES, axis=-1)[..., :_PAIR_CANDIDATES]
    cand_dist = np.take_along_axis(cells_dist, cand[:, :, None, :], axis=-1)

    pair_scores = np.minimum(cand_dist[..., :, None], cand_dist[..., None, :]).sum(axis=2)
    best_local = pair_scores.reshape(h_cells, w_cells, -1).argmin(axis=-1)
    fg_local = best_local // _PAIR_CANDIDATES
    bg_local = best_local % _PAIR_CANDIDATES
    fg_idx = np.take_along_axis(cand, fg_local[..., None], axis=-1)[..., 0]
    bg_idx = np.take_along_axis(cand, bg_local[..., None], axis=-1)[..., 0]

    # Re-evaluate the chosen pair so we can emit the canonical mask/glyph form.
    d_fg = np.take_along_axis(cells_dist, fg_idx[:, :, None, None], axis=-1)[..., 0]
    d_bg = np.take_along_axis(cells_dist, bg_idx[:, :, None, None], axis=-1)[..., 0]
    is_fg = d_fg <= d_bg                                       # (H, W, 6)

    invert = is_fg[..., 5]                                     # bottom-right is fg
    mask = np.packbits(is_fg[..., :5], axis=-1, bitorder="little")[..., 0]
    final_fg = np.where(invert, bg_idx, fg_idx)
    final_bg = np.where(invert, fg_idx, bg_idx)
    state_mask = np.where(invert, _MASK_INVERT[mask], mask)
    state_index = _STATE_INDEX[final_fg, final_bg, state_mask]
    char = _STATE_GLYPH[state_index]
    final_fg = _STATE_FG[state_index]
    final_bg = _STATE_BG[state_index]

    rows = np.empty((h_cells, 3, w_cells), dtype=np.uint8)
    rows[:, 0, :] = char
    rows[:, 1, :] = _BLIT_LUT[final_fg]
    rows[:, 2, :] = _BLIT_LUT[final_bg]

    header = bytes((
        (w_cells >> 8) & 0xFF, w_cells & 0xFF,
        (h_cells >> 8) & 0xFF, h_cells & 0xFF,
    ))
    return header + rows.tobytes()

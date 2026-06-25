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


def _build_lut() -> np.ndarray:
    """Pre-compute a 64x64x64 table mapping 6-bit RGB -> nearest palette index."""
    idx = np.arange(64, dtype=np.float32) * 4.0
    r, g, b = np.meshgrid(idx, idx, idx, indexing="ij")
    grid = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    dists = np.sum((grid[:, None, :] - _CC_RGB[None, :, :]) ** 2, axis=2)
    return np.argmin(dists, axis=1).reshape(64, 64, 64).astype(np.uint8)


_LUT: np.ndarray = _build_lut()


def quantize(frame_rgb: np.ndarray) -> np.ndarray:
    """Map an H x W x 3 uint8 RGB array to an H x W uint8 array of palette indices."""
    r6 = frame_rgb[:, :, 0] >> 2
    g6 = frame_rgb[:, :, 1] >> 2
    b6 = frame_rgb[:, :, 2] >> 2
    return _LUT[r6, g6, b6]


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

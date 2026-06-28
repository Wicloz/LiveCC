"""
Transcoder fidelity benchmark.

Encodes a frame, decodes the blit wire format back to pixels, and compares to the
source.  Two PSNRs are reported:

  * PSNR        — raw per-pixel error.  Dithering deliberately injects
                  high-frequency noise, which *lowers* this number even though it
                  removes banding, so read it as a floor, not the whole story.
  * PSNR 2x2    — after a 2x2 box blur of both images, approximating the eye
                  integrating neighbouring sub-pixels at viewing distance.  This
                  is where dithering pays off, so it should beat raw PSNR.

Also reports the fraction of cells that actually dither (a non-solid glyph).

Run:  python benchmarks/bench_quality.py
"""

from __future__ import annotations

import numpy as np

import _bench
from _bench import CONTENT, GRIDS, fmt, section, table

from cc_encoder import _CC_RGB, encode_frame

# ascii hex byte -> palette index 0..15
_HEX = np.full(256, 0, np.intp)
for _i, _c in enumerate(b"0123456789abcdef"):
    _HEX[_c] = _i


def decode(buf: bytes) -> np.ndarray:
    """Reverse encode_frame: blit wire bytes -> (H*3, W*2, 3) uint8 image."""
    w = buf[0] << 8 | buf[1]
    h = buf[2] << 8 | buf[3]
    body = np.frombuffer(buf, np.uint8, offset=4).reshape(h, 3, w)
    glyph, fg, bg = body[:, 0, :], body[:, 1, :], body[:, 2, :]
    mask = glyph.astype(np.intp) - 0x80                 # (H,W) 5 active bits
    fg_idx = _HEX[fg]
    bg_idx = _HEX[bg]

    idx = np.empty((h, w, 6), np.intp)
    for s in range(5):                                  # s0..s4 from the mask
        idx[..., s] = np.where((mask >> s) & 1, fg_idx, bg_idx)
    idx[..., 5] = bg_idx                                # bottom-right is always bg
    rgb = _CC_RGB[idx].astype(np.float32)               # (H,W,6,3)
    return (rgb.reshape(h, w, 3, 2, 3)
            .transpose(0, 2, 1, 3, 4)
            .reshape(h * 3, w * 2, 3))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float("inf") if mse <= 1e-9 else 10.0 * np.log10(255.0 ** 2 / mse)


def _box2(img: np.ndarray) -> np.ndarray:
    """2x2 box average (crop to even dims) — a cheap stand-in for the eye blending
    adjacent sub-pixels."""
    ph, pw, _ = img.shape
    ph -= ph % 2
    pw -= pw % 2
    a = img[:ph, :pw].astype(np.float32).reshape(ph // 2, 2, pw // 2, 2, 3)
    return a.mean((1, 3))


def _dither_fraction(buf: bytes) -> float:
    w = buf[0] << 8 | buf[1]
    h = buf[2] << 8 | buf[3]
    glyph = np.frombuffer(buf, np.uint8, offset=4).reshape(h, 3, w)[:, 0, :]
    mask = glyph.astype(np.intp) - 0x80
    # solid cell == empty glyph (mask 0).  Anything else mixes the two colours.
    return float(np.mean(mask != 0))


def main(w: int = 82, h: int = 41) -> None:
    print(f"Fidelity  (grid {w}x{h}; higher PSNR = closer to source)")
    section("Per-content reconstruction error")
    rows = []
    for name, gen in CONTENT.items():
        src = gen(w, h)
        rec = decode(encode_frame(src))
        rows.append([
            name,
            fmt(_psnr(src, rec), 1),
            fmt(_psnr(_box2(src), _box2(rec)), 1),
            fmt(_dither_fraction(encode_frame(src)) * 100, 0) + "%",
        ])
    print(table(["content", "PSNR dB", "PSNR 2x2 dB", "cells dithered"], rows))


if __name__ == "__main__":
    main()

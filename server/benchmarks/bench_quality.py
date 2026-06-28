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

Also reports the fraction of cells that actually dither (a non-solid glyph).  If
the media/ folder has samples, a second table runs the same metrics on real
decoded frames.

Run:  python benchmarks/bench_quality.py
"""

from __future__ import annotations

import numpy as np

import harness
from harness import (CONTENT, decode_frame, find_media, fmt, have_ffmpeg,
                     sample_frames, section, table)

from cc_encoder import encode_frame


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


def _row(name: str, frame: np.ndarray) -> list:
    buf = encode_frame(frame)
    rec = decode_frame(buf)
    return [
        name,
        fmt(_psnr(frame, rec), 1),
        fmt(_psnr(_box2(frame), _box2(rec)), 1),
        fmt(_dither_fraction(buf) * 100, 0) + "%",
    ]


_HEADERS = ["content", "PSNR dB", "PSNR 2x2 dB", "cells dithered"]


def synthetic(w: int = 82, h: int = 41) -> None:
    section(f"Synthetic content  (grid {w}x{h}; higher PSNR = closer to source)")
    print(table(_HEADERS, [_row(name, gen(w, h)) for name, gen in CONTENT.items()]))


def real_samples(w: int = 82, h: int = 41, frames_per: int = 4) -> None:
    samples = find_media("video")
    if not (samples and have_ffmpeg()):
        return
    section(f"Real media samples  (grid {w}x{h}, mean of up to {frames_per} frames)")
    rows = []
    for path in samples:
        frames = sample_frames(path, w, h, limit=frames_per)
        if not frames:
            continue
        vals = np.array([[_psnr(f, decode_frame(encode_frame(f))),
                          _psnr(_box2(f), _box2(decode_frame(encode_frame(f)))),
                          _dither_fraction(encode_frame(f)) * 100] for f in frames])
        m = vals.mean(0)
        rows.append([path.name[:28], fmt(m[0], 1), fmt(m[1], 1), fmt(m[2], 0) + "%"])
    if rows:
        print(table(_HEADERS, rows))


def main() -> None:
    synthetic()
    real_samples()


if __name__ == "__main__":
    main()

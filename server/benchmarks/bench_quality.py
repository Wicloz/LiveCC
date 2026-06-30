"""
Transcoder fidelity benchmark.

Encodes a frame, decodes the blit wire format back to pixels, and compares to the
source.  The headline metric is perceptual:

  * ΔE (S-CIELAB) — mean S-CIELAB colour difference (cc_metrics): CIELAB ΔE after a
                    human contrast-sensitivity spatial filter, so it scores the
                    *dithered* output the way the eye integrates neighbouring
                    sub-pixels.  LOWER is better.  Reported for the adaptive palette
                    (production default) and the fixed CC palette, to show the win.
  * PSNR          — raw per-pixel error.  Dithering injects high-frequency noise that
                    *lowers* this even though it removes banding, so it's a floor, not
                    the whole story — ΔE is the number to trust.

Also reports the fraction of cells that actually dither (a non-solid glyph).  If the
media/ folder has samples, a second table runs the same metrics on real frames.

Run:  python benchmarks/bench_quality.py
"""

from __future__ import annotations

import numpy as np

import harness
from harness import (CONTENT, decode_frame, find_media, fmt, have_ffmpeg,
                     sample_frames, section, table)

from cc_encoder import encode_frame
from cc_metrics import mean_scielab


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float("inf") if mse <= 1e-9 else 10.0 * np.log10(255.0 ** 2 / mse)


def _dither_fraction(buf: bytes, w: int, h: int) -> float:
    from cc_encoder import decode_32vid
    glyph, _fg, _bg, _pal = decode_32vid(buf, w, h)
    # solid cell == empty glyph (mask 0).  Anything else mixes the two colours.
    return float(np.mean((glyph.astype(np.intp) - 0x80) != 0))


def _row(name: str, frame: np.ndarray) -> list:
    w, h = frame.shape[1] // 2, frame.shape[0] // 3
    buf = encode_frame(frame)                                  # adaptive (default)
    rec = decode_frame(buf, w, h)
    fixed = decode_frame(encode_frame(frame, adaptive=False), w, h)   # fixed palette
    return [
        name,
        fmt(mean_scielab(frame, rec), 2),                      # ΔE adaptive (lower=better)
        fmt(mean_scielab(frame, fixed), 2),                    # ΔE fixed palette
        fmt(_psnr(frame, rec), 1),
        fmt(_dither_fraction(buf, w, h) * 100, 0) + "%",
    ]


_HEADERS = ["content", "dE adapt", "dE fixed", "PSNR dB", "cells dithered"]


def synthetic(w: int = 82, h: int = 41) -> None:
    section(f"Synthetic content  (grid {w}x{h}; lower dE = closer to source)")
    print(table(_HEADERS, [_row(name, gen(w, h)) for name, gen in CONTENT.items()]))


def real_samples(w: int = 82, h: int = 41, frames_per: int = 4) -> None:
    samples = find_media("video")
    if not (samples and have_ffmpeg()):
        return
    section(f"Real media samples  (grid {w}x{h}, mean of up to {frames_per} frames)")
    rows = []
    for path in samples:
        frames = sample_frames(path, w, h, limit=frames_per)
        if not frames:                       # every media file is expected to decode
            rows.append([path.name[:28], "!", "DECODE", "FAILED", ""])
            continue
        vals = np.array([[mean_scielab(f, decode_frame(encode_frame(f), w, h)),
                          mean_scielab(f, decode_frame(encode_frame(f, adaptive=False), w, h)),
                          _psnr(f, decode_frame(encode_frame(f), w, h)),
                          _dither_fraction(encode_frame(f), w, h) * 100] for f in frames])
        m = vals.mean(0)
        rows.append([path.name[:28], fmt(m[0], 2), fmt(m[1], 2), fmt(m[2], 1),
                     fmt(m[3], 0) + "%"])
    if rows:
        print(table(_HEADERS, rows))


def main() -> None:
    synthetic()
    real_samples()


if __name__ == "__main__":
    main()

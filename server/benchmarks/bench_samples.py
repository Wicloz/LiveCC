"""
Sample-driven encoder benchmark.

Times encode_frame on *real* frames decoded from whatever clips are in media/
(through the same ffmpeg scale + splitter the server uses), so the numbers reflect
actual video rather than synthetic patterns.  Skips cleanly if media/ is empty or
ffmpeg isn't installed.

  1. By grid   — real-frame encode cost vs character-grid size (compare to the
                 synthetic size sweep in bench_encoder).
  2. By sample — per-clip cost at one grid, showing content-driven variation.

Run:  python benchmarks/bench_samples.py
"""

from __future__ import annotations

import harness
from harness import (GRIDS, find_media, fmt, have_ffmpeg, measure,
                     sample_frames, section, table)

from cc_encoder import encode_frame
from transcoder import _encode_stride

_TARGET_FPS = 24


def _measure_over(frames) -> dict:
    """measure() rotating through a pool of frames, so timing is content-averaged."""
    state = {"k": 0}

    def fn():
        f = frames[state["k"] % len(frames)]
        state["k"] += 1
        return encode_frame(f)

    return measure(fn)


def by_grid(sample, frames_per: int = 4) -> None:
    section(f"1. Real-frame encode by grid  (sample: {sample.name}, target "
            f"{_TARGET_FPS} fps)")
    rows = []
    for label, w, h in GRIDS:
        frames = sample_frames(sample, w, h, limit=frames_per)
        if not frames:
            continue
        r = _measure_over(frames)
        eff = _TARGET_FPS / _encode_stride(r["mean_ms"] / 1000.0, _TARGET_FPS)
        rows.append([label, fmt(r["mean_ms"]), fmt(r["min_ms"]),
                     fmt(1000.0 / r["mean_ms"], 0), fmt(eff, 0)])
    print(table(["grid", "mean ms", "min ms", "fps max", "eff fps"], rows))


def by_sample(w: int = 82, h: int = 41, frames_per: int = 4) -> None:
    section(f"2. Real-frame encode by sample  (grid {w}x{h})")
    rows = []
    for path in find_media("video"):
        frames = sample_frames(path, w, h, limit=frames_per)
        if not frames:
            continue
        r = _measure_over(frames)
        rows.append([path.name[:30], fmt(r["mean_ms"]), fmt(r["min_ms"]),
                     fmt(1000.0 / r["mean_ms"], 0)])
    print(table(["sample", "mean ms", "min ms", "fps max"], rows))


def main() -> None:
    samples = find_media("video")
    if not have_ffmpeg():
        print("bench_samples: ffmpeg not found — skipping.")
        return
    if not samples:
        print(f"bench_samples: no media samples in {harness.MEDIA_DIR} — "
              "drop some clips there to benchmark on real video.")
        return
    by_grid(samples[0])
    by_sample()


if __name__ == "__main__":
    main()

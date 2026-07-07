"""
GopEncoder / CCMF container benchmark — the layer the file-spec refactor added on
top of encode_frame (05fdf41).  bench_encoder times the per-cell quantiser kernel;
this times the per-frame work GopEncoder.add() wraps around it in production
(transcoder.iter_video calls GopEncoder.add() per frame, not encode_frame
directly): scene-cut detection (subsampled RGB diff), delta-span diffing, and the
bit-packing (chars/nibbles) that turns grids into wire bytes.

Three views:
  1. Streaming sequences — GopEncoder.add() amortised over realistic frame
     sequences (static/motion/scene-cut mixes), by grid size.
  2. CCMF primitives      — pack/unpack chars & nibbles, delta_spans — the
     building blocks add()/flush() call every frame.
  3. Profile              — where a streaming GOP spends its time.

Run:  python benchmarks/bench_gop.py [--profile]
"""

from __future__ import annotations

import sys

import numpy as np

import harness
from harness import CONTENT, GRIDS, fmt, measure, section, table

import ccmf
from cc_encoder import GopEncoder

_TARGET_FPS = 24
_GOP_SAMPLES = int(ccmf.SAMPLE_RATE * 2.0)   # matches transcoder.GOP_SECONDS
_NOMINAL_DUR = round(ccmf.SAMPLE_RATE / _TARGET_FPS)


# --------------------------------------------------------------------------- #
# Frame sequences — same (H*3, W*2, 3) uint8 shape encode_frame takes, but as a
# realistic *sequence* so add() sees the delta/repeat/scene-cut paths it was
# built for, not just isolated frames.
# --------------------------------------------------------------------------- #

def _static_seq(w: int, h: int, n: int) -> list[np.ndarray]:
    """Identical frame every time -> every frame after the keyframe is `repeat`."""
    frame = CONTENT["photo"](w, h)
    return [frame] * n


def _motion_seq(w: int, h: int, n: int) -> list[np.ndarray]:
    """A small bright block drifts over a static photo background -> most cells
    unchanged, a few move each frame -> the `delta` path, small spans."""
    base = CONTENT["photo"](w, h)
    ph, pw = base.shape[:2]
    bs = max(3, min(ph, pw) // 8)
    frames = []
    for i in range(n):
        f = base.copy()
        x = (i * 3) % max(1, pw - bs)
        y = (i * 2) % max(1, ph - bs)
        f[y:y + bs, x:x + bs] = (255, 200, 50)
        frames.append(f)
    return frames


def _cuts_seq(w: int, h: int, n: int, period: int = 6) -> list[np.ndarray]:
    """Hard-switches between unrelated content every `period` frames, with the
    same small drifting block as _motion_seq *within* each period -> forces
    GopEncoder's RGB scene-cut check and mid-GOP re-keying on the switch, while
    the frames between switches stay genuinely (slightly) different from each
    other rather than pixel-identical repeats."""
    pool = [CONTENT["photo"](w, h), CONTENT["random"](w, h), CONTENT["solid"](w, h)]
    ph, pw = pool[0].shape[:2]
    bs = max(3, min(ph, pw) // 8)
    frames = []
    for i in range(n):
        f = pool[(i // period) % len(pool)].copy()
        x = (i * 3) % max(1, pw - bs)
        y = (i * 2) % max(1, ph - bs)
        f[y:y + bs, x:x + bs] = (255, 200, 50)
        frames.append(f)
    return frames


SEQUENCES = {
    "static": _static_seq,
    "motion": _motion_seq,
    "scene-cuts": _cuts_seq,
}


def _feed(seq) -> float:
    """Run a full sequence through a fresh GopEncoder, flushing at the end so no
    work is left uncounted; returns total wall time in seconds."""
    import time
    gop = GopEncoder(gop_samples=_GOP_SAMPLES, nominal_duration=_NOMINAL_DUR)
    pts = 0
    t0 = time.perf_counter()
    for frame in seq:
        gop.add(pts, frame)
        pts += _NOMINAL_DUR
    gop.flush(pts)
    return time.perf_counter() - t0


def streaming_sequences(n: int = 12, target_s: float = 0.4) -> None:
    """n frames/run (>= 2 scene-cut periods); repeats auto-calibrate per (grid,
    sequence) off the first run so a 400 ms terminal sweep and a multi-second
    max-monitor sweep both stay within `target_s`-ish total, same spirit as
    harness.measure() but state-carrying (a fresh GopEncoder per run) so it can't
    reuse measure()'s call-fn-in-a-loop shape."""
    section(f"1. Streaming sequences  (GopEncoder.add, {n} frames/run, "
            f"target {_TARGET_FPS} fps)")
    rows = []
    for label, w, h in GRIDS:
        for kind, gen in SEQUENCES.items():
            seq = gen(w, h, n)
            first = _feed(seq) / n
            repeats = max(1, min(5, round(target_s / max(first * n, 1e-6))))
            per_frame = [first] + [_feed(seq) / n for _ in range(repeats - 1)]
            mean_ms = 1000.0 * (sum(per_frame) / len(per_frame))
            min_ms = 1000.0 * min(per_frame)
            rows.append([label, kind, fmt(mean_ms), fmt(min_ms),
                        fmt(1000.0 / mean_ms, 0)])
        print(f"  ...{label} done", file=sys.stderr)
    print(table(["grid", "sequence", "mean ms", "min ms", "fps max"], rows))
    print("\nmean/min ms = GopEncoder.add() cost per frame, amortised over the "
          "sequence (includes the encode_frame kernel it wraps). 'static' only "
          "pays for the opening keyframe + cheap repeat units; 'motion' pays "
          "encode + a small delta_spans diff; 'scene-cuts' pays encode + a fresh "
          "generate_palette + re-key on every cut.")


# --------------------------------------------------------------------------- #
# CCMF primitives — the packing GopEncoder.add()/flush() call every frame.
# --------------------------------------------------------------------------- #

def primitives() -> None:
    section("2. CCMF primitives  (pack/unpack + delta_spans)")
    rows = []
    for label, w, h in GRIDS:
        n = w * h
        rng = np.random.default_rng(0)
        glyph = (0x80 + rng.integers(0, 32, (h, w), np.uint8)).astype(np.uint8)
        fg = rng.integers(0, 16, (h, w), np.uint8)
        bg = rng.integers(0, 16, (h, w), np.uint8)
        chars_packed = ccmf.pack_chars(glyph)
        fg_packed = ccmf.pack_nibbles(fg)

        # A second grid ~2% changed, for delta_spans on realistic sparse motion.
        glyph2, fg2, bg2 = glyph.copy(), fg.copy(), bg.copy()
        nflip = max(1, n // 50)
        flat_idx = rng.choice(n, nflip, replace=False)
        gf2 = glyph2.ravel()
        gf2[flat_idx] = 0x80 + rng.integers(0, 32, nflip, np.uint8)

        r_pack_c = measure(lambda: ccmf.pack_chars(glyph))
        r_unpack_c = measure(lambda: ccmf.unpack_chars(chars_packed, n))
        r_pack_n = measure(lambda: ccmf.pack_nibbles(fg))
        r_unpack_n = measure(lambda: ccmf.unpack_nibbles(fg_packed, n))
        r_delta = measure(lambda: ccmf.delta_spans((glyph, fg, bg), (glyph2, fg2, bg2)))
        rows.append([
            label, f"{n}",
            fmt(r_pack_c["mean_ms"], 3), fmt(r_unpack_c["mean_ms"], 3),
            fmt(r_pack_n["mean_ms"], 3), fmt(r_unpack_n["mean_ms"], 3),
            fmt(r_delta["mean_ms"], 3),
        ])
    print(table(
        ["grid", "cells", "pack chr", "unpack chr", "pack nib", "unpack nib",
         "delta(2%)"], rows))
    print("\nAll ms. delta(2%) diffs two grids with ~2% of cells changed — the "
          "sparse-motion case delta_spans is built for.")


def profile(w: int = 82, h: int = 41, n: int = 48) -> None:
    import cProfile
    import pstats

    seq = _motion_seq(w, h, n) + _cuts_seq(w, h, n // 4)
    for _ in range(2):
        _feed(seq)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        _feed(seq)
    pr.disable()
    section(f"cProfile  (5x {len(seq)}-frame mixed sequence, grid {w}x{h})")
    pstats.Stats(pr).sort_stats("cumulative").print_stats(15)


def main() -> None:
    streaming_sequences()
    primitives()
    if "--profile" in sys.argv:
        profile()


if __name__ == "__main__":
    main()

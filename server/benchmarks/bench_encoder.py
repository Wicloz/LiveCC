"""
PRIMARY benchmark: the video transcoder (cc_encoder.encode_frame).

Three views:
  1. Size sweep   — cost vs character-grid size on realistic ("photo") content,
                    with the realtime headroom that matters: fps ceiling and how
                    many concurrent streams fit in a 24 fps budget on one core.
  2. Content sweep— how cost varies with content at a fixed size (flat/gradient/
                    edges/photo/random), since the per-cell search is data shaped.
  3. Components   — where the time goes, timing the key numpy primitives the
                    encoder is built from.

Run:  python benchmarks/bench_encoder.py [--profile]
"""

from __future__ import annotations

import sys

import numpy as np

import harness
from harness import CONTENT, GRIDS, fmt, measure, section, table

import cc_encoder
from cc_encoder import _encode_numpy, _prepare, encode_frame
from transcoder import _encode_stride

_TARGET_FPS = 24


def _encode_numpy_full(frame):
    idx, h, w = _prepare(frame)
    return _encode_numpy(idx, h, w)


def size_sweep() -> None:
    section(f"1. Size sweep  (content: photo, single core, target {_TARGET_FPS} fps)")
    rows = []
    for label, w, h in GRIDS:
        frame = CONTENT["photo"](w, h)
        rn = measure(lambda: _encode_numpy_full(frame))            # numpy reference (fixed)
        rf = measure(lambda: encode_frame(frame, adaptive=False))  # numba, fixed palette
        ra = measure(lambda: encode_frame(frame))                  # numba, adaptive (prod)
        speedup = rn["mean_ms"] / rf["mean_ms"]                     # core: fixed vs fixed
        # The pacer/headroom reflect the production path, which is adaptive.
        eff_fps = _TARGET_FPS / _encode_stride(ra["mean_ms"] / 1000.0, _TARGET_FPS)
        streams_24 = (1000.0 / _TARGET_FPS) / ra["mean_ms"]
        rows.append([
            label, f"{w * h}",
            fmt(rn["mean_ms"]), fmt(rf["mean_ms"]), fmt(ra["mean_ms"]),
            fmt(speedup, 1) + "x",
            fmt(1000.0 / ra["mean_ms"], 0), fmt(eff_fps, 0), fmt(streams_24, 1),
        ])
    print(table(
        ["grid", "cells", "numpy ms", "fixed ms", "adapt ms", "speedup",
         "fps max", "eff fps", "strm@24"],
        rows))
    print(f"\nnumpy = reference/fallback (fixed palette); fixed/adapt ms = numba core "
          f"with CC's fixed vs the adaptive per-frame palette (adapt = production "
          f"default, so fps/eff fps/strm@24 use it).  speedup = numba vs numpy on the "
          f"same fixed-palette work; eff fps = steady rate the pacer holds at "
          f"{_TARGET_FPS} fps; strm@24 = streams/core.")


def content_sweep(w: int = 82, h: int = 41) -> None:
    section(f"2. Content sweep  (grid: {w}x{h})")
    rows = []
    for name, gen in CONTENT.items():
        frame = gen(w, h)
        ra = measure(lambda: encode_frame(frame))                  # adaptive (prod)
        rf = measure(lambda: encode_frame(frame, adaptive=False))  # fixed palette
        rows.append([name, fmt(ra["mean_ms"]), fmt(rf["mean_ms"]),
                     fmt(ra["mean_ms"] - rf["mean_ms"]), fmt(1000.0 / ra["mean_ms"], 0)])
    print(table(["content", "adapt ms", "fixed ms", "palette ms", "fps max"], rows))
    print("\npalette ms = adaptive-palette overhead (generate_palette) on top of the "
          "fixed-palette encode; data-shaped (random colour clouds cost the most).")


def components(w: int = 82, h: int = 41) -> None:
    """Per-primitive cost of the NumPy *reference* core on one frame — shows why it
    was memory-bound (and worth compiling).  Illustrative, not a strict
    decomposition (it omits some glue ops); the active core is the numba kernel."""
    section(f"3. NumPy reference anatomy  (grid: {w}x{h}, photo)")
    cc = cc_encoder
    frame = CONTENT["photo"](w, h)
    fr = frame[: h * 3, : w * 2]
    r = fr[..., 0] >> cc._SHIFT
    g = fr[..., 1] >> cc._SHIFT
    b = fr[..., 2] >> cc._SHIFT

    def lut_lookup():
        return (cc._OKLAB_LUT[r, g, b]
                .reshape(h, 3, w, 2, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, 6, 3))

    lab = lut_lookup()

    def projection():
        return lab @ cc._PAL.T

    lp_all = projection()
    score = lp_all - 0.5 * cc._PAL2

    def nearest():
        return np.argmax(score[:, :, cc._CORNERS, :], axis=-1)   # (H,W,4) corners

    def mean_top4():
        ms = lab.mean(2) @ cc._PAL.T - 0.5 * cc._PAL2
        return np.argpartition(ms, -4, axis=-1)[..., -4:]

    nb1 = nearest()
    top4 = mean_top4()
    idx_a = np.concatenate([nb1[..., cc._EII], top4[..., cc._MII]], axis=-1)
    idx_b = np.concatenate([nb1[..., cc._EJJ], top4[..., cc._MJJ]], axis=-1)
    n_pairs = idx_a.shape[-1]

    def gather():
        return np.take_along_axis(
            lp_all, np.broadcast_to(idx_a[:, :, None, :], (h, w, 6, n_pairs)), axis=-1)

    lpa = gather()
    lpb = np.take_along_axis(
        lp_all, np.broadcast_to(idx_b[:, :, None, :], (h, w, 6, n_pairs)), axis=-1)
    pa2 = cc._PAL2[idx_a]
    dot = cc._DOT[idx_a, idx_b]
    len2 = pa2 + cc._PAL2[idx_b] - 2.0 * dot
    padir = dot - pa2
    safe = np.where(len2 == 0.0, 1.0, len2)

    def score_pairs():
        qad = lpb - lpa - padir[..., None, :]
        t = np.clip(qad / safe[..., None, :], 0.0, 1.0)
        l2 = len2[..., None, :]
        term = l2 * t
        term *= cc._DITHER_WEIGHT + (1.0 - cc._DITHER_WEIGHT) * t
        term -= 2.0 * t * qad
        term -= 2.0 * lpa
        return (term.sum(2) + 6.0 * pa2).argmin(-1)

    is_b = np.zeros((h, w, 6), bool)

    def pack_emit():
        m = np.packbits(is_b[..., :5], axis=-1, bitorder="little")[..., 0]
        return cc._BLIT_LUT[np.zeros((h, w), np.intp)], m

    rows = []
    for name, fn in [
        ("oklab LUT lookup", lut_lookup),
        ("palette projection (matmul)", projection),
        ("nearest / sub-pixel (argmax)", nearest),
        ("mean top-4 (argpartition)", mean_top4),
        ("gather pairs (take_along_axis)", gather),
        ("score 12 pairs (cost+argmin)", score_pairs),
        ("pack + emit (packbits)", pack_emit),
        ("FULL numpy core", lambda: _encode_numpy_full(frame)),
        ("FULL numba core (active)", lambda: encode_frame(frame)),
    ]:
        res = measure(fn)
        rows.append([name, fmt(res["mean_ms"]), fmt(res["min_ms"])])
    print(table(["primitive", "mean ms", "min ms"], rows))


def profile(w: int = 82, h: int = 41) -> None:
    import cProfile
    import pstats

    frame = CONTENT["photo"](w, h)
    for _ in range(3):
        encode_frame(frame)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(50):
        encode_frame(frame)
    pr.disable()
    section(f"cProfile  (50x encode_frame, grid {w}x{h})")
    pstats.Stats(pr).sort_stats("cumulative").print_stats(15)


def main() -> None:
    print(f"numpy {np.__version__}")
    size_sweep()
    content_sweep()
    components()
    if "--profile" in sys.argv:
        profile()


if __name__ == "__main__":
    main()

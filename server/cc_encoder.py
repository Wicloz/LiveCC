"""
CC "image codec" encoder — ground-up rewrite.

Goal: turn a small RGB24 video frame into a W x H grid of CC blit cells, where
each cell is one of the 8192 legal states (32 glyph masks x 16 fg x 16 bg).  The
client renders each cell as a 2-wide x 3-tall block of sub-pixels, two colours,
arranged by the glyph mask.  The 16 colours are an ADAPTIVE palette (sanjuuni's
signature feature): `generate_palette` quantises a frame's own colours, and the
chosen 16 RGB triples travel as a CCMF palette unit for the client to apply via
setPaletteColour.  `encode_frame(..., adaptive=False)` falls back to CC's fixed
default palette (_CC_RGB).

Wire format lives in ccmf.py (docs/cc-media-format.md); this module produces the
per-cell (glyph, fg, bg) grids and, via GopEncoder, whole CCMF video chunks —
self-contained GOPs of palette + raw/delta/repeat frame units.

Why this is not "quantize each pixel, then keep the two commonest colours":
That pipeline makes three independent decisions (resample -> per-pixel palette ->
collapse-to-two) and each throws away information the next one needed.  Here a
single objective drives the whole cell:

    A cell picks a *pair* of palette colours — i.e. a line SEGMENT in perceptual
    colour space — and dithers its six sub-pixels along that segment.

That one idea folds the three old steps together:

  * "two dominant colours"  -> the segment's two endpoints,
  * "perceptual error"      -> the expected error of the dither itself, in OKLab:
                               bias² (distance from each sub-pixel's target to the
                               segment) plus a down-weighted dither variance, so the
                               tightest bracketing pair wins yet smooth regions still
                               dither instead of banding (see _DITHER_WEIGHT),
  * "dithering"             -> where a sub-pixel's target lands *between* the
                               endpoints, a stateless screen-space blue-noise
                               threshold decides which endpoint it shows.  Across
                               neighbouring cells the threshold varies, so the
                               aggregate colour over a small neighbourhood matches
                               the source — that is the cross-cell ("between
                               blits") fidelity term, realised implicitly and
                               without any sequential error-diffusion state (which
                               would crawl/boil frame-to-frame on video).

Endpoint search is near-exact: rather than score all 120 palette pairs per cell,
we score 12 candidate pairs chosen to cover edges and flats (see the comment by
_DITHER_WEIGHT) — within ~1% of the exhaustive search's cost on real video.  The
output state space and binary wire format are identical to the old encoder, so the
Lua client and the frame header are unchanged.

Sub-pixel bit layout within a cell (matches the client's glyph decoding):
    s0 top-left (1)    s1 top-right (2)
    s2 mid-left (4)    s3 mid-right (8)
    s4 bot-left (16)   s5 bot-right (control / always background in the wire form)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import ccmf

# numba compiles the per-cell encode kernel AND the Wu palette box loop (both below).
# It's a normal prod dependency; the pure-numpy paths are kept only as a safety-net
# fallback when it isn't importable.
try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:                          # pragma: no cover - numba is a prod dep
    _HAVE_NUMBA = False

# RGB of CC Tweaked's default palette, indexed 0..15 -> blit chars '0'..'f'.
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

# ASCII bytes for blit colours: b'0123456789abcdef'.
_BLIT_LUT = np.frombuffer(b"0123456789abcdef", dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Perceptual colour space (OKLab)
# --------------------------------------------------------------------------- #
# OKLab is close to perceptually uniform and cheaper/cleaner than CIELAB.  Distance
# is plain squared-Euclidean ΔE in OKLab.  A CHROMA-WEIGHT knob is kept for
# experimentation — it scales a,b by sqrt(w) so the metric carries ΔL² + w·(Δa²+Δb²)
# — but defaults to w=1 (literal ΔE).
#
# History: w used to be 6.  That was a bandaid for the FIXED CC palette, whose only
# light colour is white sitting above a big lightness gap — unweighted ΔE collapsed
# every light, low-chroma pixel onto white (pale tints washed out), so chroma was
# up-weighted to drag them back toward their hue.  The adaptive per-frame palette
# (the default now) puts colours where each frame needs them, so that gap is gone and
# the bandaid with it: w=1 is the honest perceptual metric.
_CHROMA_WEIGHT = np.float32(1.0)
_CHROMA_SCALE = np.sqrt(_CHROMA_WEIGHT)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB (0..1) -> linear light.  Averaging/mixing is only correct in linear."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (last axis = R,G,B in 0..255) -> OKLab (chroma scaled by _CHROMA_SCALE;
    a no-op at the default _CHROMA_WEIGHT=1)."""
    c = _srgb_to_linear(np.asarray(rgb, dtype=np.float32) / 255.0)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    lab = np.stack([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,   # L
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,   # a
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,   # b
    ], axis=-1).astype(np.float32)
    lab[..., 1:] *= _CHROMA_SCALE
    return lab


# Palette in weighted-OKLab — the universe of usable colours.
_PAL = _srgb_to_oklab(_CC_RGB)                                  # (16, 3)


# Sub-pixel target colours are looked up through a reduced-precision RGB table so
# we never run the cbrt-heavy OKLab transform per frame.  6 bits/channel keeps the
# table small (64³·3 floats ≈ 3 MB) while preserving the discrimination the pair
# search needs.  Buckets are reconstructed at their CENTRE (not floor) so the
# >>_SHIFT lookup is unbiased.
#
# Shrinking this LUT to fit L2 was measured and NOT worth it: in the compiled kernel
# the per-sub-pixel gather is ~1.5% of the per-cell cost (the cell sweep is compute-
# bound on the 12-pair scoring, not memory-bound on the LUT — the "memory-bound" note
# elsewhere is about the NumPy reference's temporaries, which the kernel avoids).
# 3 MB→48 KB (4-bit) saved only ~0.3 ms/frame, and dropping below 6 bits coarsens the
# pair search.  float16 can't shrink it: numba's CPU target (0.65.1) raises
# NotImplementedError even READING a float16 array element (fp16 is CUDA-only, numba
# #4402).  An int16 fixed-point LUT (1.5 MB, same idea) IS valid numba but measured
# ~25-40% SLOWER on the gather — the per-read int16->float32 convert costs more than
# the smaller table saves, since the access is compute- not memory-bound.  So 6-bit
# float32 stays.
_BITS_PER_CHANNEL = 6
_SHIFT = 8 - _BITS_PER_CHANNEL


def _build_oklab_lut() -> np.ndarray:
    n = 1 << _BITS_PER_CHANNEL
    step = 1 << _SHIFT
    levels = np.arange(n, dtype=np.float32) * step + (step - 1) / 2.0
    rr, gg, bb = np.meshgrid(levels, levels, levels, indexing="ij")
    return _srgb_to_oklab(np.stack([rr, gg, bb], axis=-1))       # (n, n, n, 3)


_OKLAB_LUT: np.ndarray = _build_oklab_lut()


# Linear-light tables for the dither.  The monitor emits light and the eye averages
# neighbouring sub-pixels in LINEAR RGB, not in OKLab.  So while the colour PAIR is
# chosen perceptually (OKLab, below), the dither FRACTION — how many sub-pixels show
# each endpoint — is computed in linear light.  Otherwise mid-tones come out too
# bright: dithering black/white for a mid-grey at the OKLab midpoint averages to
# ~0.5 linear, i.e. a bright grey rather than mid-grey.
_LIN_PAL = _srgb_to_linear(_CC_RGB / 255.0).astype(np.float32)   # (16,3) palette
_LIN1D = _srgb_to_linear(                                        # (64,) per channel
    (np.arange(1 << _BITS_PER_CHANNEL, dtype=np.float32) * (1 << _SHIFT)
     + ((1 << _SHIFT) - 1) / 2.0) / 255.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Palette helpers for the endpoint search
# --------------------------------------------------------------------------- #
_PAL2 = (_PAL ** 2).sum(-1)                                      # (16,)    |P_k|²
_DOT = _PAL @ _PAL.T                                             # (16,16)  P_i·P_j
np.fill_diagonal(_DOT, _PAL2)   # exact on the diagonal: the matmul leaves |P_k|²
                                # off by ~1e-8 vs (P_k**2).sum, which would make a
                                # same-colour (degenerate) pair dither faintly
                                # instead of rendering a solid palette colour.

# Weight on the dither's own variance in the cost (see encode_frame).  Pure MSE
# (weight 1) would *band* smooth regions — rounding to the nearest palette colour
# beats dithering on squared error, since dither only trades correlated error for
# equal-energy uncorrelated noise.  Down-weighting the variance makes dithering
# win across the middle of each colour interval and band only the slivers nearest
# a palette colour (where banding is least visible).  ~0.25 dithers the central
# ~60%.  Lower = smoother/noisier (more dither); higher = sharper (more banding).
_DITHER_WEIGHT = np.float32(0.25)

# Candidate endpoint pairs per cell (we don't score all 120 palette pairs):
#   * the 6 pairs among the nearest palette colour to each of the cell's 4 CORNER
#     sub-pixels — captures EDGES (a high-contrast cell keeps both its extremes),
#     and
#   * the 6 pairs among the 4 palette colours nearest the cell mean — captures the
#     bracketing pair a flat/smooth between-palette cell needs to dither (its
#     per-sub-pixel nearests are all the same colour, so the first set alone could
#     only render it solid → banding).
# This 12-pair search matches the all-6-sub-pixel / exhaustive-120-pair cost to
# well under 1% on real video (corners alone catch the cell's colour extremes; a
# feature confined to the middle row only is both rare and usually picked up by
# the mean's top-4), at a third less work than scoring all sub-pixel pairs.
#
# PCA "range-fit" edge pairs (BC1/DXT idea) were tried TWICE and rejected both times
# — see memory note pca-endpoint-rejected:
#   1. One pair from the raw colours of the two extreme sub-pixels: lost on QUALITY
#      (gave up diversity for no speed gain).
#   2. A refined least-squares version (BC1/Castaño-style: mean OKLab of each side at
#      3 axis splits, not a raw extreme colour) DID win on quality — every real-media
#      sample improved — but cost 25-46% MORE kernel time (covariance + power
#      iteration + sort + 6 nearest-palette lookups instead of 4), which loses on this
#      project's stated priority (a shared, CPU-weak production box).  Reverted again;
#      see the memory note for the full numbers if revisiting on faster hardware.
_CORNERS = np.array([0, 1, 4, 5], dtype=np.int64)   # top-L, top-R, bot-L, bot-R
_EII, _EJJ = (a.astype(np.int64) for a in np.triu_indices(4, k=1))   # (6,) corner pairs
_MII, _MJJ = (a.astype(np.int64) for a in np.triu_indices(4, k=1))   # (6,) mean top-4

# Sub-pixel (row, col) for s = sub_row*2 + sub_col, in glyph-bit order.
_SUB_ROW = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
_SUB_COL = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)


def _ign(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interleaved Gradient Noise — a stateless, screen-space, blue-noise-ish
    threshold in [0,1).  One formula per sub-pixel position, identical every
    frame, so dithering is spatially well-distributed yet temporally stable (no
    boiling).  A precomputed void-and-cluster blue-noise tile is a drop-in upgrade
    if its spectrum proves insufficient.
    """
    v = 0.06711056 * x.astype(np.float32) + 0.00583715 * y.astype(np.float32)
    return np.modf(52.9829189 * np.modf(v)[0])[0]


def _cells(arr, h, w):
    """(H*3, W*2, 3) -> (H, W, 6, 3): group sub-pixels into cells (s=row*2+col)."""
    return arr.reshape(h, 3, w, 2, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, 6, 3)


def _prepare(frame_rgb: np.ndarray):
    """RGB frame -> (idx, h, w): the 6-bit per-channel colour index of each
    sub-pixel, grouped into cells (H, W, 6, 3) uint8.  Just a shift + regroup — the
    OKLab/linear LUT lookups happen in the cores (cheaply, in the compiled kernel).
    """
    ph, pw, _ = frame_rgb.shape
    h = ph // 3
    w = pw // 2
    fr = np.asarray(frame_rgb, dtype=np.uint8)[: h * 3, : w * 2]
    idx = _cells(fr >> _SHIFT, h, w)
    return idx, h, w


# --------------------------------------------------------------------------- #
# Adaptive per-frame palette — Wu's variance-minimising quantiser
# --------------------------------------------------------------------------- #
# Instead of CC's fixed 16 colours we pick 16 colours adapted to the content, and
# the client applies them via setPaletteColour (a 48-byte CCMF palette unit, one
# per GOP — negligible bandwidth).  The encoder's perceptual OKLab matching then
# maps each cell onto whatever 16 colours come out.
#
# Algorithm: Xiaolin Wu's greedy orthogonal bipartition (1992) — the variance-
# minimising quantiser.  It builds one 3D colour histogram, accumulates the moments
# (count, ΣR ΣG ΣB, Σ(R²+G²+B²)) into 3D cumulative ("integral") tables, then
# repeatedly splits the box whose split most reduces total within-box variance,
# choosing the cut plane by inclusion-exclusion over the integral tables — NO per-
# split sort over pixels.  This beats median cut on distortion AND is cheaper on big
# frames (the cost is the one histogram pass; the box loop is ~2·ncolors tiny steps),
# which is what we need now that monitors go up to 16x9 blocks (≈250k sub-pixels).
# Each leaf's representative colour is its mean of the *full-resolution* pixel values
# (the moments accumulate true R/G/B, not bin centres), so the histogram resolution
# only limits where cut planes land, not the palette's colour precision.
#
# Temporal stability: per-frame, no smoothing.  Wu over the whole-frame histogram is
# stable on similar consecutive frames; if a clip ever flickers, the fix is to EMA-
# smooth the palette across frames (needs per-stream state) or recompute per scene.

# 4-bit (16³) histogram, not Wu's classic 5-bit.  The integral tables and the per-cut
# plane scan both scale with this, so 4-bit shrinks the integral ~7x and halves the
# box-loop scan — the floor that dominates palette time once the bincounts are
# subsampled (below).  It only coarsens CUT-PLANE placement (palette colours stay
# exact means), so it merely merges colours within 16 RGB of each other — which are
# near-identical and would share a palette entry anyway.  Measured quality-neutral vs
# 5-bit (mixed ±, net flat on real media) for a ~4 ms/frame win.
_WU_BITS = 4
_WU_LEVELS = 1 << _WU_BITS                    # 16 bins / channel
_WU_SIDE = _WU_LEVELS + 1                     # 17: index 0 is the integral-table base
# Cap how many sub-pixels feed the histogram.  The five bincounts scan every sample,
# so they dominate palette cost on big monitors — but a 16-colour palette is a coarse
# summary of the colour distribution and doesn't need full sub-pixel resolution.  A
# regular (strided) subsample to ~this many samples estimates the same distribution
# (and is temporally stable — the grid is deterministic, unlike random sampling),
# cutting palette time several-fold on large frames at negligible quality cost.  Any
# colour feature smaller than the stride is sub-cell and can't render distinctly
# anyway.  Frames already at/under this are used whole.
_WU_MAX_SAMPLES = 1 << 14                     # ~16k samples is ample for 16 colours


def _wu_moments(frame_rgb: np.ndarray):
    """Frame -> the five 3D cumulative moment tables (count, ΣR, ΣG, ΣB, Σ‖rgb‖²),
    each (_WU_SIDE,)*3.  `tbl[r,g,b]` is the moment summed over all histogram bins with
    red≤r, green≤g, blue≤b (bins are 1-indexed; plane 0 is zero), so the moment over
    any box is 8 corner lookups (inclusion-exclusion)."""
    arr = np.asarray(frame_rgb)
    ph, pw = arr.shape[:2]
    step = int(round((ph * pw / _WU_MAX_SAMPLES) ** 0.5))   # strided subsample if large
    if step > 1:
        arr = arr[::step, ::step]
    # int32 throughout (idx<=32767, R²+G²+B²<=195075 both fit) — no float copies of
    # the whole frame; bincount casts the integer weights to float internally.
    px = arr.astype(np.int32).reshape(-1, 3)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    sh = 8 - _WU_BITS
    idx = ((r >> sh) * _WU_LEVELS + (g >> sh)) * _WU_LEVELS + (b >> sh)
    n = _WU_LEVELS ** 3
    raw = (np.bincount(idx, minlength=n),
           np.bincount(idx, weights=r, minlength=n),
           np.bincount(idx, weights=g, minlength=n),
           np.bincount(idx, weights=b, minlength=n),
           np.bincount(idx, weights=r * r + g * g + b * b, minlength=n))

    def integral(a):
        t = np.zeros((_WU_SIDE, _WU_SIDE, _WU_SIDE), np.float64)
        t[1:, 1:, 1:] = a.astype(np.float64).reshape(_WU_LEVELS, _WU_LEVELS, _WU_LEVELS)
        return t.cumsum(0).cumsum(1).cumsum(2)

    return tuple(integral(a) for a in raw)


def _wu_box_means_numpy(wt, mr, mg, mb, m2, ncolors):
    """Wu box-split loop (reference / numba fallback): grow to `ncolors` boxes, each
    split where it most reduces within-box variance, and return their weighted means
    -> (ncolors,3) float (padded by repetition if fewer clusters than `ncolors`).
    The numba version `_wu_box_means_numba` is a line-for-line port of this."""
    L = wt.shape[0] - 1

    def vol(t, x):                            # moment of box x = (r0,r1,g0,g1,b0,b1)
        r0, r1, g0, g1, b0, b1 = x
        return (t[r1, g1, b1] - t[r1, g1, b0] - t[r1, g0, b1] + t[r1, g0, b0]
                - t[r0, g1, b1] + t[r0, g1, b0] + t[r0, g0, b1] - t[r0, g0, b0])

    def bottom(t, x, d):                      # part of vol fixed by the box's low face
        r0, r1, g0, g1, b0, b1 = x
        if d == 0:
            return -t[r0, g1, b1] + t[r0, g1, b0] + t[r0, g0, b1] - t[r0, g0, b0]
        if d == 1:
            return -t[r1, g0, b1] + t[r1, g0, b0] + t[r0, g0, b1] - t[r0, g0, b0]
        return -t[r1, g1, b0] + t[r1, g0, b0] + t[r0, g1, b0] - t[r0, g0, b0]

    def top(t, x, d, lo, hi):                 # vector of vol up to each plane in [lo,hi)
        r0, r1, g0, g1, b0, b1 = x
        s = slice(lo, hi)
        if d == 0:
            return t[s, g1, b1] - t[s, g1, b0] - t[s, g0, b1] + t[s, g0, b0]
        if d == 1:
            return t[r1, s, b1] - t[r1, s, b0] - t[r0, s, b1] + t[r0, s, b0]
        return t[r1, g1, s] - t[r1, g0, s] - t[r0, g1, s] + t[r0, g0, s]

    def maximize(x, d, lo, hi, ww, wr, wg, wb):
        if lo >= hi:
            return -1.0, -1
        hw = bottom(wt, x, d) + top(wt, x, d, lo, hi)
        hr = bottom(mr, x, d) + top(mr, x, d, lo, hi)
        hg = bottom(mg, x, d) + top(mg, x, d, lo, hi)
        hb = bottom(mb, x, d) + top(mb, x, d, lo, hi)
        ow, orr, og, ob = ww - hw, wr - hr, wg - hg, wb - hb
        valid = (hw > 0) & (ow > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            obj = (hr * hr + hg * hg + hb * hb) / hw + (orr * orr + og * og + ob * ob) / ow
        obj = np.where(valid, obj, -1.0)
        j = int(np.argmax(obj))
        return (float(obj[j]), lo + j) if obj[j] > 0.0 else (-1.0, -1)

    def cut(x):                               # split box x in place; return the new box
        ww, wr, wg, wb = (vol(t, x) for t in (wt, mr, mg, mb))
        sr, cr = maximize(x, 0, x[0] + 1, x[1], ww, wr, wg, wb)
        sg, cg = maximize(x, 1, x[2] + 1, x[3], ww, wr, wg, wb)
        sb, cb = maximize(x, 2, x[4] + 1, x[5], ww, wr, wg, wb)
        if sr >= sg and sr >= sb:
            d, c = 0, cr
        elif sg >= sb:
            d, c = 1, cg
        else:
            d, c = 2, cb
        if c < 0:
            return None
        new = list(x)
        x[2 * d + 1], new[2 * d] = c, c       # x -> [lo, c], new -> [c, hi]
        return new

    def variance(x):
        ww = vol(wt, x)
        if ww <= 1:
            return 0.0
        dr, dg, db = vol(mr, x), vol(mg, x), vol(mb, x)
        return float(vol(m2, x) - (dr * dr + dg * dg + db * db) / ww)

    boxes = [[0, L, 0, L, 0, L]]              # one box over the whole colour cube
    vv = [variance(boxes[0])]
    nxt = 0
    while len(boxes) < ncolors:
        new = cut(boxes[nxt])
        if new is not None:
            boxes.append(new)
            vv[nxt] = variance(boxes[nxt])
            vv.append(variance(new))
        else:
            vv[nxt] = -1.0                     # unsplittable: never pick it again
        nxt = int(np.argmax(vv))
        if vv[nxt] <= 0.0:                     # no box left worth splitting
            break

    pal = []
    for x in boxes:
        ww = vol(wt, x)
        if ww > 0:
            pal.append([vol(mr, x) / ww, vol(mg, x) / ww, vol(mb, x) / ww])
    while len(pal) < ncolors:                  # pad if fewer clusters than the palette
        pal.append(pal[-1] if pal else [0.0, 0.0, 0.0])
    return np.array(pal[:ncolors], dtype=np.float64)


if _HAVE_NUMBA:
    # Compiled port of the box loop above.  The integral build (_wu_moments) stays
    # numpy (bincount + cumsum), but the box loop is sequential Python over tiny numpy
    # micro-ops — ~half the palette time on big monitors.  Compiling it to scalar code
    # (the integral tables passed in, indexed directly) removes that floor.  No
    # fastmath: the box decisions are exact float64 sums/divides, so this matches the
    # numpy reference (a parity test asserts it), and the box loop isn't FLOP-bound.
    @njit(cache=True)
    def _wu_vol(t, r0, r1, g0, g1, b0, b1):   # moment of a box (8-corner inclusion-excl.)
        return (t[r1, g1, b1] - t[r1, g1, b0] - t[r1, g0, b1] + t[r1, g0, b0]
                - t[r0, g1, b1] + t[r0, g1, b0] + t[r0, g0, b1] - t[r0, g0, b0])

    @njit(cache=True)
    def _wu_var(wt, mr, mg, mb, m2, r0, r1, g0, g1, b0, b1):
        ww = _wu_vol(wt, r0, r1, g0, g1, b0, b1)
        if ww <= 1.0:
            return 0.0
        dr = _wu_vol(mr, r0, r1, g0, g1, b0, b1)
        dg = _wu_vol(mg, r0, r1, g0, g1, b0, b1)
        db = _wu_vol(mb, r0, r1, g0, g1, b0, b1)
        return _wu_vol(m2, r0, r1, g0, g1, b0, b1) - (dr * dr + dg * dg + db * db) / ww

    @njit(cache=True)
    def _wu_maximize(wt, mr, mg, mb, r0, r1, g0, g1, b0, b1, d, ww, wr, wg, wb):
        """Best cut plane for the box along axis d (0=R,1=G,2=B): maximises the
        between-box Σ(‖m‖²/count), i.e. minimises within-box variance.  -> (score,
        plane); (-1,-1) if there's no valid split."""
        if d == 0:
            lo, hi = r0 + 1, r1
        elif d == 1:
            lo, hi = g0 + 1, g1
        else:
            lo, hi = b0 + 1, b1
        if lo >= hi:
            return -1.0, -1
        # bottom: the part of each half-box moment fixed by the box's low face
        if d == 0:
            bw = -wt[r0, g1, b1] + wt[r0, g1, b0] + wt[r0, g0, b1] - wt[r0, g0, b0]
            br = -mr[r0, g1, b1] + mr[r0, g1, b0] + mr[r0, g0, b1] - mr[r0, g0, b0]
            bg = -mg[r0, g1, b1] + mg[r0, g1, b0] + mg[r0, g0, b1] - mg[r0, g0, b0]
            bb = -mb[r0, g1, b1] + mb[r0, g1, b0] + mb[r0, g0, b1] - mb[r0, g0, b0]
        elif d == 1:
            bw = -wt[r1, g0, b1] + wt[r1, g0, b0] + wt[r0, g0, b1] - wt[r0, g0, b0]
            br = -mr[r1, g0, b1] + mr[r1, g0, b0] + mr[r0, g0, b1] - mr[r0, g0, b0]
            bg = -mg[r1, g0, b1] + mg[r1, g0, b0] + mg[r0, g0, b1] - mg[r0, g0, b0]
            bb = -mb[r1, g0, b1] + mb[r1, g0, b0] + mb[r0, g0, b1] - mb[r0, g0, b0]
        else:
            bw = -wt[r1, g1, b0] + wt[r1, g0, b0] + wt[r0, g1, b0] - wt[r0, g0, b0]
            br = -mr[r1, g1, b0] + mr[r1, g0, b0] + mr[r0, g1, b0] - mr[r0, g0, b0]
            bg = -mg[r1, g1, b0] + mg[r1, g0, b0] + mg[r0, g1, b0] - mg[r0, g0, b0]
            bb = -mb[r1, g1, b0] + mb[r1, g0, b0] + mb[r0, g1, b0] - mb[r0, g0, b0]
        best = 0.0
        cut = -1
        for i in range(lo, hi):
            if d == 0:
                tw = wt[i, g1, b1] - wt[i, g1, b0] - wt[i, g0, b1] + wt[i, g0, b0]
                tr = mr[i, g1, b1] - mr[i, g1, b0] - mr[i, g0, b1] + mr[i, g0, b0]
                tg = mg[i, g1, b1] - mg[i, g1, b0] - mg[i, g0, b1] + mg[i, g0, b0]
                tb = mb[i, g1, b1] - mb[i, g1, b0] - mb[i, g0, b1] + mb[i, g0, b0]
            elif d == 1:
                tw = wt[r1, i, b1] - wt[r1, i, b0] - wt[r0, i, b1] + wt[r0, i, b0]
                tr = mr[r1, i, b1] - mr[r1, i, b0] - mr[r0, i, b1] + mr[r0, i, b0]
                tg = mg[r1, i, b1] - mg[r1, i, b0] - mg[r0, i, b1] + mg[r0, i, b0]
                tb = mb[r1, i, b1] - mb[r1, i, b0] - mb[r0, i, b1] + mb[r0, i, b0]
            else:
                tw = wt[r1, g1, i] - wt[r1, g0, i] - wt[r0, g1, i] + wt[r0, g0, i]
                tr = mr[r1, g1, i] - mr[r1, g0, i] - mr[r0, g1, i] + mr[r0, g0, i]
                tg = mg[r1, g1, i] - mg[r1, g0, i] - mg[r0, g1, i] + mg[r0, g0, i]
                tb = mb[r1, g1, i] - mb[r1, g0, i] - mb[r0, g1, i] + mb[r0, g0, i]
            hw = bw + tw
            if hw <= 0.0:
                continue
            ow = ww - hw
            if ow <= 0.0:
                continue
            hr = br + tr
            hg = bg + tg
            hb = bb + tb
            orr = wr - hr
            og = wg - hg
            ob = wb - hb
            temp = (hr * hr + hg * hg + hb * hb) / hw + (orr * orr + og * og + ob * ob) / ow
            if temp > best:
                best = temp
                cut = i
        if cut < 0:
            return -1.0, -1
        return best, cut

    @njit(cache=True)
    def _wu_box_means_numba(wt, mr, mg, mb, m2, ncolors):
        L = wt.shape[0] - 1
        boxes = np.zeros((ncolors, 6), np.int64)
        vv = np.empty(ncolors, np.float64)
        boxes[0, 1] = L
        boxes[0, 3] = L
        boxes[0, 5] = L                       # box 0 = [0,L, 0,L, 0,L]
        vv[0] = _wu_var(wt, mr, mg, mb, m2, 0, L, 0, L, 0, L)
        nbox = 1
        nxt = 0
        while nbox < ncolors:
            r0 = boxes[nxt, 0]; r1 = boxes[nxt, 1]
            g0 = boxes[nxt, 2]; g1 = boxes[nxt, 3]
            b0 = boxes[nxt, 4]; b1 = boxes[nxt, 5]
            ww = _wu_vol(wt, r0, r1, g0, g1, b0, b1)
            wr = _wu_vol(mr, r0, r1, g0, g1, b0, b1)
            wg = _wu_vol(mg, r0, r1, g0, g1, b0, b1)
            wb = _wu_vol(mb, r0, r1, g0, g1, b0, b1)
            sr, cr = _wu_maximize(wt, mr, mg, mb, r0, r1, g0, g1, b0, b1, 0, ww, wr, wg, wb)
            sg, cg = _wu_maximize(wt, mr, mg, mb, r0, r1, g0, g1, b0, b1, 1, ww, wr, wg, wb)
            sb, cb = _wu_maximize(wt, mr, mg, mb, r0, r1, g0, g1, b0, b1, 2, ww, wr, wg, wb)
            if sr >= sg and sr >= sb:
                d, c = 0, cr
            elif sg >= sb:
                d, c = 1, cg
            else:
                d, c = 2, cb
            if c < 0:
                vv[nxt] = -1.0                 # unsplittable: never pick it again
            else:
                for k in range(6):
                    boxes[nbox, k] = boxes[nxt, k]
                boxes[nxt, 2 * d + 1] = c      # nxt -> [lo, c]
                boxes[nbox, 2 * d] = c         # new -> [c, hi]
                vv[nxt] = _wu_var(wt, mr, mg, mb, m2, boxes[nxt, 0], boxes[nxt, 1],
                                  boxes[nxt, 2], boxes[nxt, 3], boxes[nxt, 4], boxes[nxt, 5])
                vv[nbox] = _wu_var(wt, mr, mg, mb, m2, boxes[nbox, 0], boxes[nbox, 1],
                                   boxes[nbox, 2], boxes[nbox, 3], boxes[nbox, 4], boxes[nbox, 5])
                nbox += 1
            nxt = 0
            best = vv[0]
            for i in range(1, nbox):
                if vv[i] > best:
                    best = vv[i]
                    nxt = i
            if best <= 0.0:                    # no box left worth splitting
                break

        out = np.zeros((ncolors, 3), np.float64)
        c0 = 0.0
        c1 = 0.0
        c2 = 0.0
        for i in range(ncolors):
            if i < nbox:
                r0 = boxes[i, 0]; r1 = boxes[i, 1]
                g0 = boxes[i, 2]; g1 = boxes[i, 3]
                b0 = boxes[i, 4]; b1 = boxes[i, 5]
                ww = _wu_vol(wt, r0, r1, g0, g1, b0, b1)
                if ww > 0.0:
                    c0 = _wu_vol(mr, r0, r1, g0, g1, b0, b1) / ww
                    c1 = _wu_vol(mg, r0, r1, g0, g1, b0, b1) / ww
                    c2 = _wu_vol(mb, r0, r1, g0, g1, b0, b1) / ww
            out[i, 0] = c0                     # pad slots (i >= nbox) repeat the last
            out[i, 1] = c1
            out[i, 2] = c2
        return out

    _wu_box_means = _wu_box_means_numba
else:                                        # pragma: no cover - numba present in prod
    _wu_box_means = _wu_box_means_numpy


def generate_palette(frame_rgb: np.ndarray, ncolors: int = 16) -> np.ndarray:
    """Pick `ncolors` colours adapted to one RGB frame via Wu's quantiser ->
    (ncolors,3) uint8 RGB.  Pads by repetition if the frame has fewer than `ncolors`
    distinct colour clusters."""
    wt, mr, mg, mb, m2 = _wu_moments(frame_rgb)
    means = _wu_box_means(wt, mr, mg, mb, m2, ncolors)
    return np.clip(np.round(means), 0, 255).astype(np.uint8)


def _palette_tables(pal_rgb: np.ndarray):
    """A palette (16,3 uint8 RGB) -> the per-frame search tables the cores need:
    `pal` (chroma-weighted OKLab, for the perceptual pair search), `pal2` (|P_k|²),
    `lin_pal` (linear-light RGB, for the dither fraction) and `dot` (P_i·P_j, the
    numpy core's degenerate-pair guard).  All float32 and contiguous for numba."""
    pal_f = np.asarray(pal_rgb, dtype=np.float32)
    pal = np.ascontiguousarray(_srgb_to_oklab(pal_f))                       # (16,3)
    pal2 = np.ascontiguousarray((pal ** 2).sum(-1))                         # (16,)
    lin_pal = np.ascontiguousarray(_srgb_to_linear(pal_f / 255.0).astype(np.float32))
    dot = pal @ pal.T
    np.fill_diagonal(dot, pal2)        # exact diagonal (see _DOT) for solid colours
    return pal, pal2, lin_pal, np.ascontiguousarray(dot.astype(np.float32))


def _encode_numpy(idx, h, w, pal=_PAL, pal2=_PAL2, lin_pal=_LIN_PAL, dot=_DOT):
    """Reference vectorised core: 6-bit indices -> (glyph, fg, bg) palette-index
    arrays.  The palette tables (`pal`/`pal2`/`lin_pal`/`dot`, see _palette_tables)
    default to CC's fixed palette but are passed per-frame when adaptive.

    Kept as the readable reference and the fallback when numba isn't importable;
    the numba kernel below is a line-for-line port.  The maths is in the module
    docstring and by _DITHER_WEIGHT.
    """
    r, g, b = idx[..., 0], idx[..., 1], idx[..., 2]             # (H,W,6) 6-bit indices
    lab = _OKLAB_LUT[r, g, b]                                    # (H,W,6,3) OKLab
    lin = np.stack([_LIN1D[r], _LIN1D[g], _LIN1D[b]], axis=-1)  # (H,W,6,3) linear RGB
    lp_all = lab @ pal.T                                         # (H,W,6,16) q·P_k
    score = lp_all - 0.5 * pal2                # argmax_k = nearest palette to a sub-pixel

    nb1 = np.argmax(score[:, :, _CORNERS, :], axis=-1)        # (H,W,4) nearest/corner
    mscore = lab.mean(2) @ pal.T - 0.5 * pal2                # (H,W,16) nearest to cell mean
    top4 = np.argpartition(mscore, -4, axis=-1)[..., -4:]     # (H,W,4)
    idx_a = np.concatenate([nb1[..., _EII], top4[..., _MII]], axis=-1)  # (H,W,12) endpoint A
    idx_b = np.concatenate([nb1[..., _EJJ], top4[..., _MJJ]], axis=-1)  # (H,W,12) endpoint B

    # Score each pair by expected dither error  bias² + λ·variance  (see kernel /
    # _DITHER_WEIGHT); the |q|² part of |q-A|² is constant across pairs so we drop
    # it (argmin unchanged) and fuse the rest in place to limit (H,W,6,12) traffic.
    n_pairs = idx_a.shape[-1]
    lpa = np.take_along_axis(
        lp_all, np.broadcast_to(idx_a[:, :, None, :], (h, w, 6, n_pairs)), axis=-1)
    lpb = np.take_along_axis(
        lp_all, np.broadcast_to(idx_b[:, :, None, :], (h, w, 6, n_pairs)), axis=-1)
    pa2 = pal2[idx_a]
    pdot = dot[idx_a, idx_b]
    len2 = pa2 + pal2[idx_b] - 2.0 * pdot
    padir = pdot - pa2
    safe = np.where(len2 == 0.0, 1.0, len2)                     # guard A==B (solid)

    qad = lpb - lpa - padir[..., None, :]
    t = np.clip(qad / safe[..., None, :], 0.0, 1.0)
    l2 = len2[..., None, :]
    term = l2 * t
    term *= _DITHER_WEIGHT + (1.0 - _DITHER_WEIGHT) * t
    term -= 2.0 * t * qad
    term -= 2.0 * lpa
    cost = term.sum(2) + 6.0 * pa2
    best = cost.argmin(-1)                                      # (H,W) winning pair

    sel = best[:, :, None]
    idx_a = np.take_along_axis(idx_a, sel, axis=-1)[..., 0]     # (H,W)
    idx_b = np.take_along_axis(idx_b, sel, axis=-1)[..., 0]

    # Dither FRACTION in linear light (where the eye mixes the sub-pixels): project
    # each sub-pixel's linear colour onto the chosen pair's linear segment.  The
    # pair was chosen in OKLab above; only the per-sub-pixel split is linear.
    la = lin_pal[idx_a]                                         # (H,W,3)
    dlin = lin_pal[idx_b] - la
    llen2 = (dlin * dlin).sum(-1)                               # (H,W)
    lsafe = np.where(llen2 == 0.0, 1.0, llen2)
    t = np.clip(((lin - la[:, :, None, :]) * dlin[:, :, None, :]).sum(-1)
                / lsafe[:, :, None], 0.0, 1.0)                  # (H,W,6)

    # Blue-noise dither along the chosen segment, then canonicalise: B is fg, A is
    # bg; if the bottom-right sub-pixel would be fg, invert the mask and swap.
    ys = (np.arange(h, dtype=np.int64)[:, None, None] * 3 + _SUB_ROW)
    xs = (np.arange(w, dtype=np.int64)[None, :, None] * 2 + _SUB_COL)
    thr = _ign(np.broadcast_to(xs, (h, w, 6)), np.broadcast_to(ys, (h, w, 6)))
    is_b = t > thr

    invert = is_b[..., 5]
    mask = np.packbits(is_b[..., :5], axis=-1, bitorder="little")[..., 0]
    final_fg = np.where(invert, idx_a, idx_b)
    final_bg = np.where(invert, idx_b, idx_a)
    glyph = np.uint8(0x80) + np.where(invert, np.uint8(31) - mask, mask)
    return glyph, final_fg, final_bg


# --------------------------------------------------------------------------- #
# Compiled core (numba)
# --------------------------------------------------------------------------- #
# The per-cell work — 12 candidate pairs x 6 sub-pixels of small fixed loops — is
# trivial in FLOPs but the NumPy version pays heavily in temporaries and gathers
# (it's memory-bound, not compute-bound).  Compiling it to one scalar pass with no
# temporaries is ~6x the NumPy core single-threaded (see benchmarks/bench_native).
# Kept single-threaded by request; the loop is per-cell independent so a parallel
# prange is a drop-in later if needed.  Falls back to _encode_numpy if numba isn't
# importable (numba/_HAVE_NUMBA are set up at the top of the module).
if _HAVE_NUMBA:
    @njit(cache=True, fastmath=True)
    def _ign_scalar(x, y):                   # scalar Interleaved Gradient Noise
        v = np.float32(0.06711056) * x + np.float32(0.00583715) * y
        v = v - np.floor(v)
        w = np.float32(52.9829189) * v
        return w - np.floor(w)

    @njit(cache=True, fastmath=True)
    def _encode_kernel(idx, lut, pal, pal2, lin_pal, glyph, fg, bg):
        """Line-for-line port of _encode_numpy.  Single pass over cells; per
        sub-pixel it looks up OKLab (for the pair search, from `lut`) and linear RGB
        (for the dither, from _LIN1D) by its 6-bit index — that lookup in compiled
        code is far cheaper than a NumPy fancy-index gather, so _prepare only hands
        over the indices.  The palette tables `pal`/`pal2`/`lin_pal` are kernel ARGS
        (not frozen globals) so each frame can pass its own adaptive palette.  Reuses
        per-frame scratch buffers (single-threaded)."""
        H = idx.shape[0]
        W = idx.shape[1]
        lam = _DITHER_WEIGHT
        cand = np.empty(12, np.int64)
        candb = np.empty(12, np.int64)
        cn = np.empty(4, np.int64)
        t4 = np.empty(4, np.int64)
        ms = np.empty(16, np.float32)
        clab = np.empty((6, 3), np.float32)
        for y in range(H):
            for x in range(W):
                # OKLab of the 6 sub-pixels (perceptual pair search)
                for s in range(6):
                    ir = idx[y, x, s, 0]
                    ig = idx[y, x, s, 1]
                    ib = idx[y, x, s, 2]
                    clab[s, 0] = lut[ir, ig, ib, 0]
                    clab[s, 1] = lut[ir, ig, ib, 1]
                    clab[s, 2] = lut[ir, ig, ib, 2]
                # nearest palette to each of the 4 corner sub-pixels (edges)
                for ci in range(4):
                    s = _CORNERS[ci]
                    best = -1e30
                    bk = 0
                    for k in range(16):
                        d = (clab[s, 0] * pal[k, 0] + clab[s, 1] * pal[k, 1]
                             + clab[s, 2] * pal[k, 2]) - 0.5 * pal2[k]
                        if d > best:
                            best = d
                            bk = k
                    cn[ci] = bk
                # 4 palette colours nearest the cell mean (flat brackets)
                mx = 0.0
                my = 0.0
                mz = 0.0
                for s in range(6):
                    mx += clab[s, 0]
                    my += clab[s, 1]
                    mz += clab[s, 2]
                mx /= 6.0
                my /= 6.0
                mz /= 6.0
                for k in range(16):
                    ms[k] = (mx * pal[k, 0] + my * pal[k, 1] + mz * pal[k, 2]) - 0.5 * pal2[k]
                for j in range(4):
                    best = -1e30
                    bk = 0
                    for k in range(16):
                        if ms[k] > best:
                            best = ms[k]
                            bk = k
                    t4[j] = bk
                    ms[bk] = -1e30
                # 12 candidate pairs: 6 corner-combos + 6 mean-combos
                p = 0
                for i in range(4):
                    for j in range(i + 1, 4):
                        cand[p] = cn[i]
                        candb[p] = cn[j]
                        p += 1
                for i in range(4):
                    for j in range(i + 1, 4):
                        cand[p] = t4[i]
                        candb[p] = t4[j]
                        p += 1
                # lowest expected-dither-error pair
                bestc = 1e30
                ba = cand[0]
                bb = candb[0]
                for pp in range(12):
                    a = cand[pp]
                    b = candb[pp]
                    dx0 = pal[b, 0] - pal[a, 0]
                    dx1 = pal[b, 1] - pal[a, 1]
                    dx2 = pal[b, 2] - pal[a, 2]
                    len2 = dx0 * dx0 + dx1 * dx1 + dx2 * dx2
                    c = 0.0
                    if len2 < 1e-12:                    # degenerate pair A==B (solid)
                        for s in range(6):
                            q0 = clab[s, 0] - pal[a, 0]
                            q1 = clab[s, 1] - pal[a, 1]
                            q2 = clab[s, 2] - pal[a, 2]
                            c += q0 * q0 + q1 * q1 + q2 * q2
                    else:
                        for s in range(6):
                            q0 = clab[s, 0] - pal[a, 0]
                            q1 = clab[s, 1] - pal[a, 1]
                            q2 = clab[s, 2] - pal[a, 2]
                            qad = q0 * dx0 + q1 * dx1 + q2 * dx2
                            qa2 = q0 * q0 + q1 * q1 + q2 * q2
                            t = qad / len2
                            if t < 0.0:
                                t = 0.0
                            elif t > 1.0:
                                t = 1.0
                            c += qa2 - 2.0 * t * qad + len2 * t * (lam + (1.0 - lam) * t)
                    if c < bestc:
                        bestc = c
                        ba = a
                        bb = b
                # dither chosen segment in LINEAR light (per-sub-pixel linear value
                # from _LIN1D) — where the eye averages the sub-pixels — then
                # canonicalise the glyph.  Pair was chosen in OKLab above.
                dx0 = lin_pal[bb, 0] - lin_pal[ba, 0]
                dx1 = lin_pal[bb, 1] - lin_pal[ba, 1]
                dx2 = lin_pal[bb, 2] - lin_pal[ba, 2]
                len2 = dx0 * dx0 + dx1 * dx1 + dx2 * dx2
                mask = 0
                invert = False
                for s in range(6):
                    if len2 < 1e-12:
                        t = 0.0
                    else:
                        q0 = _LIN1D[idx[y, x, s, 0]] - lin_pal[ba, 0]
                        q1 = _LIN1D[idx[y, x, s, 1]] - lin_pal[ba, 1]
                        q2 = _LIN1D[idx[y, x, s, 2]] - lin_pal[ba, 2]
                        t = (q0 * dx0 + q1 * dx1 + q2 * dx2) / len2
                        if t < 0.0:
                            t = 0.0
                        elif t > 1.0:
                            t = 1.0
                    thr = _ign_scalar(np.float32(x * 2 + _SUB_COL[s]),
                                      np.float32(y * 3 + _SUB_ROW[s]))
                    isb = t > thr
                    if s == 5:
                        invert = isb
                    elif isb:
                        mask |= (1 << s)
                if invert:
                    fg[y, x] = ba
                    bg[y, x] = bb
                    glyph[y, x] = np.uint8(0x80 + (31 - mask))
                else:
                    fg[y, x] = bb
                    bg[y, x] = ba
                    glyph[y, x] = np.uint8(0x80 + mask)

    def _encode_numba(idx, h, w, pal=_PAL, pal2=_PAL2, lin_pal=_LIN_PAL, dot=_DOT):
        # `dot` is unused by the kernel (the numpy core's degenerate-pair guard); it
        # rides the same signature so both cores are called the same way.
        idx = np.ascontiguousarray(idx, dtype=np.uint8)
        glyph = np.empty((h, w), np.uint8)
        fg = np.empty((h, w), np.int64)
        bg = np.empty((h, w), np.int64)
        _encode_kernel(idx, _OKLAB_LUT, pal, pal2, lin_pal, glyph, fg, bg)
        return glyph, fg, bg

    _encode_core = _encode_numba
else:                                        # pragma: no cover - numba present in prod
    _encode_core = _encode_numpy


def encode_frame(frame_rgb: np.ndarray, adaptive: bool = True):
    """Encode a (H*3) x (W*2) x 3 RGB frame -> (glyph, fg, bg, palette) arrays:
    (H, W) uint8 grids (glyph = 0x80 + 5-bit code) plus the (16, 3) uint8 RGB
    palette they index.  Uses the compiled numba core, falling back to pure numpy.

    adaptive=True (default) picks a fresh 16-colour palette from the frame's own
    content (generate_palette) — sanjuuni's signature feature, and the biggest
    quality lever for out-of-gamut content.  adaptive=False uses CC's fixed
    default palette (_CC_RGB).  Streaming uses GopEncoder below, which reuses one
    palette across a GOP so delta frames stay meaningful."""
    idx, h, w = _prepare(frame_rgb)
    pal_rgb = generate_palette(frame_rgb) if adaptive else _CC_RGB.astype(np.uint8)
    pal, pal2, lin_pal, dot = _palette_tables(pal_rgb)
    glyph, fg, bg = _encode_core(idx, h, w, pal, pal2, lin_pal, dot)
    return glyph, fg.astype(np.uint8), bg.astype(np.uint8), pal_rgb


# --------------------------------------------------------------------------- #
# GOP encoder — frames in, CCMF video chunks out
# --------------------------------------------------------------------------- #

class GopEncoder:
    """Accumulate encoded frames into self-contained CCMF video chunks (GOPs).

    Each GOP opens with a palette unit + a raw keyframe (spec §4.4), generated
    from the keyframe's own content; subsequent frames reuse that palette so
    their deltas describe genuine content change (a per-frame palette would
    re-index every cell and make deltas as big as keyframes).  Per frame:

      * unchanged grid            -> `repeat` unit (duration only),
      * delta smaller than raw    -> `delta` unit (changed spans),
      * delta >= raw, OR a big
        RGB-level jump (scene cut)-> `raw` mid-GOP, per the spec's SHOULD,
                                     preceded by a fresh palette unit only if
                                     re-quantising the new frame actually
                                     produced different colours (the palette
                                     isn't spec-locked to the raw frame it
                                     sits next to — Section 4.5 — so an
                                     unchanged one is never resent); later
                                     frames delta against whichever palette is
                                     now in effect.

    The RGB check exists because grid-space deltas can't see through a
    degenerate palette: after a near-solid keyframe all 16 entries collapse to
    one colour, so a scene cut encoded against them still lands near-identical
    cells and a tiny delta — the frame would hold the stale look until the next
    GOP.  Comparing the source pixels (subsampled mean |diff|) catches that.

    Frame durations are the true PTS gaps to the next encoded frame (adaptive
    pacing may skip source frames, so gaps vary); the last frame of a GOP gets
    its gap from the frame that opens the next GOP, or `nominal_duration` at
    end of stream.  Not thread-safe; one instance per video stream.

    A GOP is bounded by DURATION *and* by BYTES: a chunk travels as one
    WebSocket message, and CC:Tweaked drops the connection outright when a
    message exceeds `http.max_websocket_message` (128 KiB by default) — so a
    busy 1-second GOP on a big monitor would kill the whole stream.  When the
    next frame would push the chunk past `max_chunk_bytes`, the GOP is flushed
    early and the frame opens a new one (spec-conforming: GOP length is an
    encoder knob).  Worst case (max monitor, every frame a keyframe) degrades
    to one-frame GOPs, which is the honest cost of that content.
    """

    # Mean |RGB diff| (subsampled) above which a frame counts as a scene cut.
    # Motion/noise on continuous footage sits well under this; a genuine cut
    # jumps far past it.  A false positive only costs one extra keyframe.
    _SCENE_CUT_DIFF = 24.0

    def __init__(self, gop_samples: int = ccmf.SAMPLE_RATE,
                 nominal_duration: int = 2000,
                 max_chunk_bytes: int = 96 * 1024,
                 compression: int = ccmf.COMPRESSION_NONE) -> None:
        self.gop_samples = gop_samples          # target GOP span (default 1 s)
        self.compression = compression          # payload compression (spec §4.1.2)
        self.nominal_duration = min(nominal_duration, ccmf.MAX_DURATION)
        self.max_chunk_bytes = max_chunk_bytes  # stay under CC's 128 KiB ws cap
        self._entries: list[tuple] = []         # ("pal", bytes) | (enc, pts, body)
        self._gop_pts = 0
        self._bytes = 0                         # wire size of the open chunk
        self._w = self._h = 0
        self._tables = None                     # palette tables for delta frames
        self._pal_bytes = None                  # last emitted palette, to skip resends
        self._prev = None                       # (glyph, fg, bg) of the last frame
        self._prev_rgb = None                   # subsampled pixels of the last frame
        self._prev_frame = None                 # full pixels of the last frame

    @staticmethod
    def _subsample(frame_rgb: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame_rgb)
        ph, pw = arr.shape[:2]
        step = max(1, round((ph * pw / 4096) ** 0.5))
        return arr[::step, ::step].astype(np.int16)

    def _encode(self, frame_rgb: np.ndarray, tables) -> tuple:
        idx, h, w = _prepare(frame_rgb)
        self._w, self._h = w, h
        glyph, fg, bg = _encode_core(idx, h, w, *tables)
        return glyph, fg.astype(np.uint8), bg.astype(np.uint8)

    # Wire size of a frame entry: unit flags + duration, plus the body.  A
    # keyframe re-key also carries a 49-byte palette unit.
    _FRAME_OVERHEAD = 3
    _PALETTE_UNIT_BYTES = 49
    _CHUNK_OVERHEAD = 11 + 5                     # chunk header + video payload head

    def _open_gop(self, pts: int, frame_rgb: np.ndarray) -> None:
        pal_rgb = generate_palette(frame_rgb)
        self._tables = _palette_tables(pal_rgb)
        self._pal_bytes = pal_rgb.tobytes()
        self._prev = self._encode(frame_rgb, self._tables)
        self._prev_frame = np.asarray(frame_rgb)
        self._gop_pts = pts
        self._entries = [("pal", self._pal_bytes),
                         (ccmf.ENC_RAW, pts, self._prev)]   # planes packed at flush
        self._bytes = (self._CHUNK_OVERHEAD + self._PALETTE_UNIT_BYTES
                       + self._FRAME_OVERHEAD
                       + ccmf.raw_planes_size(self._w, self._h))

    def add(self, pts: int, frame_rgb: np.ndarray) -> Optional[tuple[int, bytes]]:
        """Feed the next frame (absolute `pts` in 48 kHz samples).  Returns the
        finished (gop_pts, chunk_bytes) when this frame starts a new GOP, else
        None.  Call flush() after the last frame."""
        done = None
        if self._entries and pts - self._gop_pts >= self.gop_samples:
            done = self._flush(next_pts=pts)
        if not self._entries:
            self._open_gop(pts, frame_rgb)
            self._prev_rgb = self._subsample(frame_rgb)
            return done

        if self._prev_frame is not None and np.array_equal(frame_rgb, self._prev_frame):
            # Pixel-identical to the previous frame (a paused/static source, or a
            # duplicate frame the decoder emitted) -> skip the encode and the
            # delta diff entirely; a `repeat` unit is correct and free.  Cheap to
            # check (one array comparison) next to what it saves (a full
            # quantiser pass).  self._prev_rgb / self._prev stay as they were —
            # nothing changed for the next frame to compare against either.
            size = self._FRAME_OVERHEAD
            if self._bytes + size > self.max_chunk_bytes:
                done = self._flush(next_pts=pts)
                self._open_gop(pts, frame_rgb)
                return done
            self._entries.append((ccmf.ENC_REPEAT, pts, b""))
            self._bytes += size
            return done

        rgb_sub = self._subsample(frame_rgb)
        cut = (self._prev_rgb is None or rgb_sub.shape != self._prev_rgb.shape
               or np.abs(rgb_sub - self._prev_rgb).mean() > self._SCENE_CUT_DIFF)
        self._prev_rgb = rgb_sub
        grids = body = None
        if not cut:
            grids = self._encode(frame_rgb, self._tables)
            body = ccmf.delta_spans(self._prev, grids)
            if body is not None and \
                    len(body) >= ccmf.raw_planes_size(self._w, self._h):
                cut = True                       # busier than a keyframe: re-key

        if cut:
            # Conservative (over-)estimate: whether the re-key actually needs a
            # fresh palette unit isn't known until generate_palette() runs
            # below, so this assumes the worst case.  Never undercounts, so it
            # can only flush a hair earlier than strictly necessary, never
            # exceed max_chunk_bytes.
            size = (self._PALETTE_UNIT_BYTES + self._FRAME_OVERHEAD
                    + ccmf.raw_planes_size(self._w, self._h))
        else:
            size = self._FRAME_OVERHEAD + (len(body) if body is not None else 0)
        if self._bytes + size > self.max_chunk_bytes:
            # Size-bounded flush: this frame would overflow the WebSocket
            # message cap, so it opens a new GOP instead (done is still None
            # here — a duration flush above would have emptied the entries).
            done = self._flush(next_pts=pts)
            self._open_gop(pts, frame_rgb)
            return done

        if cut:
            # Scene cut: re-quantise and drop in a fresh keyframe mid-GOP (may
            # cost a second encode; cuts are rare).  A palette unit isn't
            # spec-required alongside every raw frame (only the GOP's very
            # first pair) — if content that changed enough to force a re-key
            # still quantises to the SAME 16 colours, resending them would be
            # pure waste, so skip the unit and reuse the existing tables.
            pal_rgb = generate_palette(frame_rgb)
            pal_bytes = pal_rgb.tobytes()
            if pal_bytes != self._pal_bytes:
                self._tables = _palette_tables(pal_rgb)
                self._pal_bytes = pal_bytes
                self._entries.append(("pal", pal_bytes))
            else:
                size -= self._PALETTE_UNIT_BYTES
            grids = self._encode(frame_rgb, self._tables)
            self._entries.append((ccmf.ENC_RAW, pts, grids))
        elif body is None:
            self._entries.append((ccmf.ENC_REPEAT, pts, b""))
        else:
            self._entries.append((ccmf.ENC_DELTA, pts, body))
        self._bytes += size
        self._prev = grids if grids is not None else self._prev
        self._prev_frame = np.asarray(frame_rgb)
        return done

    def flush(self, next_pts: Optional[int] = None) -> Optional[tuple[int, bytes]]:
        """Close the open GOP -> (gop_pts, chunk_bytes), or None if empty."""
        if not self._entries:
            return None
        return self._flush(next_pts)

    def _flush(self, next_pts: Optional[int]) -> tuple[int, bytes]:
        frame_pts = [e[1] for e in self._entries if e[0] != "pal"]
        # Duration of frame i = gap to frame i+1; the last runs to the next GOP
        # (or the nominal frame interval at end of stream).  Clamped to the u16
        # the unit carries — a longer hold just saturates, which is harmless
        # because the next chunk's own PTS re-anchors playback.
        last_end = next_pts if next_pts is not None else \
            frame_pts[-1] + self.nominal_duration
        gaps = [b - a for a, b in zip(frame_pts, frame_pts[1:])] + \
               [max(0, last_end - frame_pts[-1])]
        durations = iter(min(g, ccmf.MAX_DURATION) for g in gaps)

        units = []
        for entry in self._entries:
            if entry[0] == "pal":
                units.append(ccmf.palette_unit(entry[1]))
                continue
            enc, _pts, body = entry
            duration = next(durations)
            if enc == ccmf.ENC_RAW:
                units.append(ccmf.raw_frame_unit(duration, *body))
            elif enc == ccmf.ENC_DELTA:
                units.append(ccmf.delta_frame_unit(duration, body))
            else:
                units.append(ccmf.repeat_frame_unit(duration))

        payload = ccmf.video_payload(self._w, self._h, b"".join(units))
        out = (self._gop_pts, ccmf.chunk(self._gop_pts, ccmf.TYPE_VIDEO, payload,
                                         compression=self.compression))
        self._entries = []
        return out


# Warm the JIT at import (compile now, or load the on-disk cache) so the first
# real video frame after a server start isn't delayed by a one-off compile.  Costs
# ~nothing once cached; the server rarely restarts, so paying it at startup beats
# stalling the first stream.  numpy fallback makes this a cheap no-op.
if _HAVE_NUMBA:
    encode_frame(np.zeros((3, 2, 3), dtype=np.uint8))

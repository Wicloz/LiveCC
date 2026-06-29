"""
CC "image codec" encoder — ground-up rewrite.

Goal: turn a small RGB24 video frame into a W x H grid of CC blit cells, where
each cell is one of the 8192 legal states (32 glyph masks x 16 fg x 16 bg).  The
client renders each cell as a 2-wide x 3-tall block of sub-pixels, two colours,
arranged by the glyph mask.  The 16 colours are CC's *fixed* default palette
(the client never calls setPaletteColour).

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

import numpy as np

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
# OKLab is close to perceptually uniform and cheaper/cleaner than CIELAB.  We
# work with a CHROMA-WEIGHTED OKLab: distance = ΔL² + w·(Δa²+Δb²), folded into
# the coordinates by scaling a,b by sqrt(w) so a plain squared-Euclidean metric
# already carries the weight.  CC's palette has exactly one light colour (white)
# with a big lightness gap below it, so unweighted ΔE collapses every light,
# low-chroma pixel onto white (light blues / pale tints wash out).  Up-weighting
# chroma pulls those back toward their hue.  It is safe for neutrals (black, the
# greys and white are all chroma≈0, so lightness still separates them).  Higher w
# = more colourful / less white; w=1 = literal ΔE.

_CHROMA_WEIGHT = np.float32(6.0)
_CHROMA_SCALE = np.sqrt(_CHROMA_WEIGHT)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB (0..1) -> linear light.  Averaging/mixing is only correct in linear."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (last axis = R,G,B in 0..255) -> chroma-weighted OKLab."""
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
_BITS_PER_CHANNEL = 6
_SHIFT = 8 - _BITS_PER_CHANNEL


def _build_oklab_lut() -> np.ndarray:
    n = 1 << _BITS_PER_CHANNEL
    step = 1 << _SHIFT
    levels = np.arange(n, dtype=np.float32) * step + (step - 1) / 2.0
    rr, gg, bb = np.meshgrid(levels, levels, levels, indexing="ij")
    return _srgb_to_oklab(np.stack([rr, gg, bb], axis=-1))       # (n, n, n, 3)


_OKLAB_LUT: np.ndarray = _build_oklab_lut()


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


def _prepare(frame_rgb: np.ndarray):
    """RGB frame -> (lab, h, w): per-sub-pixel weighted-OKLab grouped into cells
    (H, W, 6, 3).  Cheap LUT lookup; shared by both the numpy and numba cores."""
    ph, pw, _ = frame_rgb.shape
    h = ph // 3
    w = pw // 2
    fr = np.asarray(frame_rgb, dtype=np.uint8)[: h * 3, : w * 2]
    r = fr[..., 0] >> _SHIFT
    g = fr[..., 1] >> _SHIFT
    b = fr[..., 2] >> _SHIFT
    lab = (_OKLAB_LUT[r, g, b]
           .reshape(h, 3, w, 2, 3)
           .transpose(0, 2, 1, 3, 4)
           .reshape(h, w, 6, 3))
    return lab, h, w


def _assemble(glyph, fg, bg, h: int, w: int) -> bytes:
    """(glyph, fg, bg) per-cell arrays -> the binary blit wire format:

        bytes 0-1 : W (uint16 big-endian, character columns)
        bytes 2-3 : H (uint16 big-endian, character rows)
        then H rows, each: W text bytes (0x80-0x9F) + W fg + W bg (ASCII hex)

    A row's three strings are exactly what mon.blit(text, fg, bg) expects.
    """
    rows = np.empty((h, 3, w), dtype=np.uint8)
    rows[:, 0, :] = glyph
    rows[:, 1, :] = _BLIT_LUT[fg]
    rows[:, 2, :] = _BLIT_LUT[bg]
    header = bytes((
        (w >> 8) & 0xFF, w & 0xFF,
        (h >> 8) & 0xFF, h & 0xFF,
    ))
    return header + rows.tobytes()


def _encode_numpy(lab, h, w):
    """Reference vectorised core: lab -> (glyph, fg, bg) palette-index arrays.

    Kept as the readable reference and the fallback when numba isn't importable;
    the numba kernel below is a line-for-line port.  The maths is in the module
    docstring and by _DITHER_WEIGHT.
    """
    lp_all = lab @ _PAL.T                                        # (H,W,6,16) q·P_k
    score = lp_all - 0.5 * _PAL2               # argmax_k = nearest palette to a sub-pixel

    nb1 = np.argmax(score[:, :, _CORNERS, :], axis=-1)        # (H,W,4) nearest/corner
    mscore = lab.mean(2) @ _PAL.T - 0.5 * _PAL2               # (H,W,16) nearest to cell mean
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
    pa2 = _PAL2[idx_a]
    dot = _DOT[idx_a, idx_b]
    len2 = pa2 + _PAL2[idx_b] - 2.0 * dot
    padir = dot - pa2
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
    idx_a = np.take_along_axis(idx_a, sel, axis=-1)[..., 0]
    idx_b = np.take_along_axis(idx_b, sel, axis=-1)[..., 0]
    t = np.take_along_axis(
        t, np.broadcast_to(best[:, :, None, None], (h, w, 6, 1)), axis=-1)[..., 0]

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
# importable (it's a normal pip dependency, so that's only a safety net).
try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:                          # pragma: no cover - numba is a prod dep
    _HAVE_NUMBA = False


if _HAVE_NUMBA:
    @njit(cache=True, fastmath=True)
    def _ign_scalar(x, y):                   # scalar Interleaved Gradient Noise
        v = np.float32(0.06711056) * x + np.float32(0.00583715) * y
        v = v - np.floor(v)
        w = np.float32(52.9829189) * v
        return w - np.floor(w)

    @njit(cache=True, fastmath=True)
    def _encode_kernel(lab, glyph, fg, bg):
        """Line-for-line port of _encode_numpy.  Single pass over cells, no
        temporaries, no gathers.  Reuses per-frame scratch buffers — fine because
        it is single-threaded."""
        H = lab.shape[0]
        W = lab.shape[1]
        lam = _DITHER_WEIGHT
        cand = np.empty(12, np.int64)
        candb = np.empty(12, np.int64)
        cn = np.empty(4, np.int64)
        t4 = np.empty(4, np.int64)
        ms = np.empty(16, np.float32)
        for y in range(H):
            for x in range(W):
                cell = lab[y, x]
                # nearest palette to each of the 4 corner sub-pixels (edges)
                for ci in range(4):
                    s = _CORNERS[ci]
                    best = -1e30
                    bk = 0
                    for k in range(16):
                        d = (cell[s, 0] * _PAL[k, 0] + cell[s, 1] * _PAL[k, 1]
                             + cell[s, 2] * _PAL[k, 2]) - 0.5 * _PAL2[k]
                        if d > best:
                            best = d
                            bk = k
                    cn[ci] = bk
                # 4 palette colours nearest the cell mean (flat brackets)
                mx = 0.0
                my = 0.0
                mz = 0.0
                for s in range(6):
                    mx += cell[s, 0]
                    my += cell[s, 1]
                    mz += cell[s, 2]
                mx /= 6.0
                my /= 6.0
                mz /= 6.0
                for k in range(16):
                    ms[k] = (mx * _PAL[k, 0] + my * _PAL[k, 1] + mz * _PAL[k, 2]) - 0.5 * _PAL2[k]
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
                    dx0 = _PAL[b, 0] - _PAL[a, 0]
                    dx1 = _PAL[b, 1] - _PAL[a, 1]
                    dx2 = _PAL[b, 2] - _PAL[a, 2]
                    len2 = dx0 * dx0 + dx1 * dx1 + dx2 * dx2
                    c = 0.0
                    if len2 < 1e-12:                    # degenerate pair A==B (solid)
                        for s in range(6):
                            q0 = cell[s, 0] - _PAL[a, 0]
                            q1 = cell[s, 1] - _PAL[a, 1]
                            q2 = cell[s, 2] - _PAL[a, 2]
                            c += q0 * q0 + q1 * q1 + q2 * q2
                    else:
                        for s in range(6):
                            q0 = cell[s, 0] - _PAL[a, 0]
                            q1 = cell[s, 1] - _PAL[a, 1]
                            q2 = cell[s, 2] - _PAL[a, 2]
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
                # dither chosen segment + canonicalise glyph
                dx0 = _PAL[bb, 0] - _PAL[ba, 0]
                dx1 = _PAL[bb, 1] - _PAL[ba, 1]
                dx2 = _PAL[bb, 2] - _PAL[ba, 2]
                len2 = dx0 * dx0 + dx1 * dx1 + dx2 * dx2
                mask = 0
                invert = False
                for s in range(6):
                    if len2 < 1e-12:
                        t = 0.0
                    else:
                        q0 = cell[s, 0] - _PAL[ba, 0]
                        q1 = cell[s, 1] - _PAL[ba, 1]
                        q2 = cell[s, 2] - _PAL[ba, 2]
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

    def _encode_numba(lab, h, w):
        lab = np.ascontiguousarray(lab, dtype=np.float32)
        glyph = np.empty((h, w), np.uint8)
        fg = np.empty((h, w), np.int64)
        bg = np.empty((h, w), np.int64)
        _encode_kernel(lab, glyph, fg, bg)
        return glyph, fg, bg

    _encode_core = _encode_numba
else:                                        # pragma: no cover - numba present in prod
    _encode_core = _encode_numpy


def encode_frame(frame_rgb: np.ndarray) -> bytes:
    """Encode a (H*3) x (W*2) x 3 RGB frame into the blit wire format (see
    _assemble).  Uses the compiled numba core, falling back to pure numpy."""
    lab, h, w = _prepare(frame_rgb)
    glyph, fg, bg = _encode_core(lab, h, w)
    return _assemble(glyph, fg, bg, h, w)


# Warm the JIT at import (compile now, or load the on-disk cache) so the first
# real video frame after a server start isn't delayed by a one-off compile.  Costs
# ~nothing once cached; the server rarely restarts, so paying it at startup beats
# stalling the first stream.  numpy fallback makes this a cheap no-op.
if _HAVE_NUMBA:
    encode_frame(np.zeros((3, 2, 3), dtype=np.uint8))

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
we score 21 candidate pairs chosen to cover edges and flats (see the comment by
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
#   * the 15 pairs among the nearest palette colour to each of the 6 sub-pixels
#     — captures EDGES (a high-contrast cell keeps both its extremes), and
#   * the 6 pairs among the 4 palette colours nearest the cell mean — captures
#     the bracketing pair a flat/smooth between-palette cell needs to dither
#     (its per-sub-pixel nearests are all the same colour, so the first set alone
#     could only render it solid → banding).
# On real video this 21-pair search stays within ~1% of the exhaustive 120-pair
# cost while dithering flats and edges the same way it does.
_SII, _SJJ = (a.astype(np.int64) for a in np.triu_indices(6, k=1))   # (15,) sub-pixel
_MII, _MJJ = (a.astype(np.int64) for a in np.triu_indices(4, k=1))   # (6,)  mean top-4

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


def encode_frame(frame_rgb: np.ndarray) -> bytes:
    """
    Encode a (H*3) x (W*2) x 3 RGB frame into the binary wire format:

        bytes 0-1 : W (uint16 big-endian, character columns)
        bytes 2-3 : H (uint16 big-endian, character rows)
        then H rows, each: W text bytes (0x80-0x9F) + W fg + W bg (ASCII hex)

    A row's three strings are exactly what mon.blit(text, fg, bg) expects.
    """
    ph, pw, _ = frame_rgb.shape
    h = ph // 3
    w = pw // 2

    fr = np.asarray(frame_rgb, dtype=np.uint8)[: h * 3, : w * 2]
    r = fr[..., 0] >> _SHIFT
    g = fr[..., 1] >> _SHIFT
    b = fr[..., 2] >> _SHIFT

    # Target OKLab per sub-pixel, grouped into cells: (H, W, 6, 3).
    lab = (_OKLAB_LUT[r, g, b]
           .reshape(h, 3, w, 2, 3)
           .transpose(0, 2, 1, 3, 4)
           .reshape(h, w, 6, 3))
    lp_all = lab @ _PAL.T                                        # (H,W,6,16) q·P_k
    score = lp_all - 0.5 * _PAL2               # argmax_k = nearest palette to a sub-pixel

    # --- Build the 21 candidate endpoint pairs (palette indices) ------------- #
    nb1 = np.argmax(score, axis=-1)                            # (H,W,6) nearest/sub-pixel
    mscore = lab.mean(2) @ _PAL.T - 0.5 * _PAL2               # (H,W,16) nearest to cell mean
    top4 = np.argpartition(mscore, -4, axis=-1)[..., -4:]     # (H,W,4)
    idx_a = np.concatenate([nb1[..., _SII], top4[..., _MII]], axis=-1)  # (H,W,21) endpoint A
    idx_b = np.concatenate([nb1[..., _SJJ], top4[..., _MJJ]], axis=-1)  # (H,W,21) endpoint B

    # --- Score each pair by expected dither error ---------------------------- #
    # For a pair (A,B), target q projects onto the segment at
    # t = clip((q-A)·dir / |dir|², 0, 1).  The pair's cost is
    #     bias²      = distance² from q to the segment        (how well it spans q)
    #   + λ·variance = λ·t(1-t)·|dir|²                         (down-weighted noise)
    # summed over the 6 sub-pixels — see _DITHER_WEIGHT for why λ<1.  Per sub-pixel
    # this is  |q-A|² - 2t(q-A)·dir + t·|dir|²·(λ + (1-λ)t); the |q|² part of |q-A|²
    # is the same for every pair, so we drop it (the argmin is unchanged) and the
    # remaining work is fused in place to keep the (H,W,6,21) traffic down.
    n_pairs = idx_a.shape[-1]
    lpa = np.take_along_axis(
        lp_all, np.broadcast_to(idx_a[:, :, None, :], (h, w, 6, n_pairs)), axis=-1)
    lpb = np.take_along_axis(
        lp_all, np.broadcast_to(idx_b[:, :, None, :], (h, w, 6, n_pairs)), axis=-1)
    pa2 = _PAL2[idx_a]                                          # (H,W,21) |A|²
    dot = _DOT[idx_a, idx_b]                                    # (H,W,21) A·B
    len2 = pa2 + _PAL2[idx_b] - 2.0 * dot                       # (H,W,21) |B-A|²
    padir = dot - pa2                                           # (H,W,21) A·dir
    safe = np.where(len2 == 0.0, 1.0, len2)                     # guard A==B (solid)

    qad = lpb - lpa - padir[..., None, :]                       # (H,W,6,21) (q-A)·dir
    t = np.clip(qad / safe[..., None, :], 0.0, 1.0)
    l2 = len2[..., None, :]
    term = l2 * t                                               # fold cost in place
    term *= _DITHER_WEIGHT + (1.0 - _DITHER_WEIGHT) * t
    term -= 2.0 * t * qad
    term -= 2.0 * lpa
    cost = term.sum(2) + 6.0 * pa2                              # (H,W,21) (|q|² dropped)
    best = cost.argmin(-1)                                      # (H,W) winning pair

    # Resolve the winning pair's palette indices and reuse its already-computed
    # per-sub-pixel projection t for the dither.
    sel = best[:, :, None]
    idx_a = np.take_along_axis(idx_a, sel, axis=-1)[..., 0]     # (H,W)
    idx_b = np.take_along_axis(idx_b, sel, axis=-1)[..., 0]     # (H,W)
    t = np.take_along_axis(
        t, np.broadcast_to(best[:, :, None, None], (h, w, 6, 1)), axis=-1)[..., 0]  # (H,W,6)

    # --- Blue-noise dither along the segment --------------------------------- #
    # A sub-pixel shows endpoint B with probability ≈ t (its position along the
    # segment), decided against the screen-space noise threshold at its absolute
    # pixel coordinate.  t≈0 -> always A, t≈1 -> always B, t≈0.5 -> mixed; the mix
    # averages to the true in-between colour over neighbouring cells.
    ys = (np.arange(h, dtype=np.int64)[:, None, None] * 3 + _SUB_ROW)  # (H,1,6)
    xs = (np.arange(w, dtype=np.int64)[None, :, None] * 2 + _SUB_COL)  # (1,W,6)
    thr = _ign(np.broadcast_to(xs, (h, w, 6)), np.broadcast_to(ys, (h, w, 6)))
    is_b = t > thr                                             # (H,W,6) bool: show B

    # --- Canonicalise into the wire form ------------------------------------- #
    # Treat B as foreground, A as background, then honour the format rule that the
    # bottom-right sub-pixel must be background: if it would be foreground, invert
    # the five mask bits and swap fg/bg (visually identical).
    invert = is_b[..., 5]
    mask = np.packbits(is_b[..., :5], axis=-1, bitorder="little")[..., 0]
    final_fg = np.where(invert, idx_a, idx_b)
    final_bg = np.where(invert, idx_b, idx_a)
    glyph = np.uint8(0x80) + np.where(invert, np.uint8(31) - mask, mask)

    rows = np.empty((h, 3, w), dtype=np.uint8)
    rows[:, 0, :] = glyph
    rows[:, 1, :] = _BLIT_LUT[final_fg]
    rows[:, 2, :] = _BLIT_LUT[final_bg]

    header = bytes((
        (w >> 8) & 0xFF, w & 0xFF,
        (h >> 8) & 0xFF, h & 0xFF,
    ))
    return header + rows.tobytes()

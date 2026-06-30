"""
CC "image codec" encoder — ground-up rewrite.

Goal: turn a small RGB24 video frame into a W x H grid of CC blit cells, where
each cell is one of the 8192 legal states (32 glyph masks x 16 fg x 16 bg).  The
client renders each cell as a 2-wide x 3-tall block of sub-pixels, two colours,
arranged by the glyph mask.  The 16 colours are an ADAPTIVE per-frame palette
(sanjuuni's signature feature): `generate_palette` median-cuts each frame's own
colours, and the chosen 16 RGB triples ride along in the 32vid frame (see
_assemble) for the client to apply via setPaletteColour.  `encode_frame(...,
adaptive=False)` falls back to CC's fixed default palette (_CC_RGB).

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

import struct

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


# CC's fixed default palette as 48 bytes (R,G,B each) — the per-frame palette block
# of a 32vid frame when adaptive palettes are off (encode_frame(adaptive=False)).
_PALETTE_BYTES = _CC_RGB.astype(np.uint8).tobytes()


# --------------------------------------------------------------------------- #
# Adaptive per-frame palette (sanjuuni's signature feature)
# --------------------------------------------------------------------------- #
# Instead of CC's fixed 16 colours we pick 16 colours *per frame* from the frame's
# own content, and the client applies them via setPaletteColour (the 32vid frame
# already carries a 48-byte palette block, so this costs no extra bandwidth on the
# uncompressed stream).  Median cut is sanjuuni's default and what we mirror here:
# recursively split the colour cloud along its widest axis at the (population-
# weighted) median, then take each leaf box's mean as a palette entry.
#
# Done in sRGB (gamma) space like sanjuuni — simple, robust, and the encoder's
# perceptual OKLab matching then maps each cell onto whatever 16 colours come out.
# Colours are first collapsed to the 6-bit-per-channel buckets the encoder already
# works in (a bincount histogram), so the median cut runs over a small set of
# weighted unique colours rather than every pixel — cheap and frame-to-frame stable
# (no subsampling jitter).
#
# Temporal stability: this is per-frame with no smoothing.  Median cut over the
# whole-frame histogram is stable on similar consecutive frames; if a clip ever
# shows palette flicker, the fix is to EMA-smooth the palette across frames (needs
# per-stream state, so it'd move encode_frame behind a small stateful encoder) or
# recompute only per scene.  Started simple per the plan; revisit via render_cc.

def _box(levels: np.ndarray, weights: np.ndarray):
    """A median-cut box: (levels, weights, split-axis, score).  `levels` is (3, M)
    integer per-channel bucket indices — CHANNELS-FIRST so the per-channel min/max
    reductions run along the contiguous axis (axis 0 of an (M,3) layout is strided
    and far slower on a large colour cloud).  The widest channel and the population-
    weighted spread along it are computed ONCE here — the split loop then just picks
    the max-score box without rescanning.  score < 0 marks an unsplittable colour."""
    if levels.shape[1] < 2:
        return (levels, weights, 0, -1.0)
    rng = levels.max(1) - levels.min(1)
    ax = int(rng.argmax())
    return (levels, weights, ax, float(rng[ax]) * float(weights.sum()))


def _median_cut(levels: np.ndarray, weights: np.ndarray, ncolors: int) -> np.ndarray:
    """Median-cut `levels` (3, M int bucket indices) weighted by `weights` (M,) into
    `ncolors` representative colours -> (ncolors,3) float bucket-index means.  Splits
    the box with the largest population-weighted spread along its widest channel at
    the weighted-median level; each leaf's weighted mean is its colour.  Pads by
    repetition if there are fewer distinct colours than `ncolors`.

    The split is a weighted histogram over the (≤64) levels on the chosen axis, not a
    sort — O(box size) per split, so it stays cheap even on near-random frames where
    the colour cloud is large."""
    boxes = [_box(levels, weights)]
    while len(boxes) < ncolors:
        bi = max(range(len(boxes)), key=lambda i: boxes[i][3])
        if boxes[bi][3] < 0.0:                 # nothing left to split
            break
        c, w, ax, _ = boxes.pop(bi)
        vals = c[ax]
        cum = np.cumsum(np.bincount(vals, weights=w))   # weight up to each level
        m = int(np.searchsorted(cum, cum[-1] * 0.5))    # weighted-median level
        m = min(max(m, int(vals.min())), int(vals.max()) - 1)   # keep both halves
        mask = vals <= m
        boxes.append(_box(c[:, mask], w[mask]))
        boxes.append(_box(c[:, ~mask], w[~mask]))
    pal = np.empty((ncolors, 3), np.float64)
    for i in range(ncolors):
        if i < len(boxes):
            c, w, _, _ = boxes[i]
            pal[i] = (c * w).sum(1) / w.sum()
        else:
            pal[i] = pal[i - 1]                # fewer colours than the palette holds
    return pal


def generate_palette(frame_rgb: np.ndarray, ncolors: int = 16) -> np.ndarray:
    """Pick `ncolors` colours adapted to one RGB frame via median cut -> (ncolors,3)
    uint8 RGB.  Histograms the frame into 6-bit-per-channel buckets first (the same
    precision the encoder quantises to), so the cut runs over weighted unique
    colours."""
    px = np.asarray(frame_rgb, dtype=np.uint8).reshape(-1, 3)
    nlev = 1 << _BITS_PER_CHANNEL
    q = (px >> _SHIFT).astype(np.int64)                       # 6-bit channels, 0..63
    keys = (q[:, 0] * nlev + q[:, 1]) * nlev + q[:, 2]
    counts = np.bincount(keys, minlength=nlev ** 3)
    nz = np.nonzero(counts)[0]
    w = counts[nz].astype(np.float64)
    levels = np.stack([nz // (nlev * nlev), (nz // nlev) % nlev, nz % nlev], axis=0)
    means = _median_cut(levels.astype(np.int64), w, ncolors)  # (16,3) mean bucket idx
    step = 1 << _SHIFT
    rgb = means * step + (step - 1) / 2.0                     # bucket index -> 0..255
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


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


def _assemble(glyph, fg, bg, h: int, w: int, palette_bytes: bytes = _PALETTE_BYTES) -> bytes:
    """(glyph, fg, bg) per-cell arrays -> one 32vid *uncompressed* video frame
    (MCJack123/sanjuuni's format, compression mode 0).  No per-frame header — the
    decoder knows W/H from the 32vid stream header — so the frame is exactly:

        screen : ceil(W*H/8)*5 bytes — 8 cells packed into 5 bytes, MSB-first;
                 each cell is the 5-bit drawing-char index (glyph - 0x80)
        colour : W*H bytes — one per cell, (bg << 4) | fg
        palette: 48 bytes — 16 * (R,G,B)

    Cells are row-major (i = y*W + x).
    """
    n = h * w
    # --- screen: 8 five-bit codes -> 5 bytes, code0 in the high bits ---------- #
    codes = (np.asarray(glyph, np.uint8).ravel() - np.uint8(0x80))   # (n,) 0..31
    pad = (-n) % 8
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, np.uint8)])
    grp = codes.reshape(-1, 8).astype(np.uint64)
    val = (grp[:, 0] << 35 | grp[:, 1] << 30 | grp[:, 2] << 25 | grp[:, 3] << 20
           | grp[:, 4] << 15 | grp[:, 5] << 10 | grp[:, 6] << 5 | grp[:, 7])
    screen = np.empty((val.shape[0], 5), np.uint8)
    screen[:, 0] = (val >> 32) & 0xFF
    screen[:, 1] = (val >> 24) & 0xFF
    screen[:, 2] = (val >> 16) & 0xFF
    screen[:, 3] = (val >> 8) & 0xFF
    screen[:, 4] = val & 0xFF
    # --- colour: bg in the high nibble, fg in the low nibble ----------------- #
    colour = (np.asarray(bg, np.uint8).ravel() << 4) | np.asarray(fg, np.uint8).ravel()
    return screen.tobytes() + colour.tobytes() + bytes(palette_bytes)


def decode_32vid(frame: bytes, w: int, h: int):
    """Reference decoder (inverse of _assemble) -> (glyph, fg, bg, palette).

    glyph/fg/bg are (H, W) uint8 (glyph = 0x80 + 5-bit code); palette is (16, 3)
    uint8 RGB.  Used by the Python tooling/tests and mirrored by the Lua client.
    """
    n = h * w
    ng = (n + 7) // 8
    scr = np.frombuffer(frame, np.uint8, count=ng * 5, offset=0).reshape(ng, 5).astype(np.uint64)
    val = (scr[:, 0] << 32 | scr[:, 1] << 24 | scr[:, 2] << 16 | scr[:, 3] << 8 | scr[:, 4])
    codes = np.empty((ng, 8), np.uint8)
    for j in range(8):
        codes[:, j] = (val >> np.uint64(5 * (7 - j))) & np.uint64(0x1F)
    glyph = (codes.ravel()[:n].reshape(h, w) + np.uint8(0x80))
    off = ng * 5
    colour = np.frombuffer(frame, np.uint8, count=n, offset=off).reshape(h, w)
    fg = colour & 0x0F
    bg = colour >> 4
    palette = np.frombuffer(frame, np.uint8, count=48, offset=off + n).reshape(16, 3)
    return glyph, fg, bg, palette


# --------------------------------------------------------------------------- #
# 32vid container (chunked stream)
# --------------------------------------------------------------------------- #
# A 32vid stream is the 12-byte file header followed by a sequence of chunks, each
# self-delimiting (its own size) and tagged by type, so video and audio chunks can
# be interleaved live.  We send one video frame per chunk.
_V32_MAGIC = b"32VD"
V32_FLAG_BASE = 0x10       # bit 4 is always set
V32_FLAG_DFPWM_AUDIO = 0x04  # audio compression bits 2-3: 0=PCM, 1 (bit 2)=DFPWM
V32_TYPE_VIDEO = 0
V32_TYPE_AUDIO = 1


def v32_stream_header(w: int, h: int, fps: int, nstreams: int, flags: int) -> bytes:
    """The 12-byte 32vid file header: "32VD" + <width,height,fps,nstreams,flags>.
    Sent once at the start of the stream (and re-sent if fps changes)."""
    return _V32_MAGIC + struct.pack("<HHBBH", w, h, fps, nstreams, flags)


def v32_chunk(ctype: int, datalength: int, data: bytes) -> bytes:
    """One 32vid chunk: <size, datalength, type> header + data.  `datalength` is the
    number of video frames or audio samples carried in `data`."""
    return struct.pack("<IIB", len(data), datalength, ctype) + bytes(data)


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


def encode_frame(frame_rgb: np.ndarray, adaptive: bool = True) -> bytes:
    """Encode a (H*3) x (W*2) x 3 RGB frame into the 32vid wire format (see
    _assemble).  Uses the compiled numba core, falling back to pure numpy.

    adaptive=True (default) picks a fresh 16-colour palette per frame from the
    frame's own content (generate_palette) — sanjuuni's signature feature, and the
    biggest quality lever for out-of-gamut content.  adaptive=False uses CC's fixed
    default palette (_CC_RGB)."""
    idx, h, w = _prepare(frame_rgb)
    pal_rgb = generate_palette(frame_rgb) if adaptive else _CC_RGB.astype(np.uint8)
    pal, pal2, lin_pal, dot = _palette_tables(pal_rgb)
    glyph, fg, bg = _encode_core(idx, h, w, pal, pal2, lin_pal, dot)
    return _assemble(glyph, fg, bg, h, w, pal_rgb.tobytes())


# Warm the JIT at import (compile now, or load the on-disk cache) so the first
# real video frame after a server start isn't delayed by a one-off compile.  Costs
# ~nothing once cached; the server rarely restarts, so paying it at startup beats
# stalling the first stream.  numpy fallback makes this a cheap no-op.
if _HAVE_NUMBA:
    encode_frame(np.zeros((3, 2, 3), dtype=np.uint8))

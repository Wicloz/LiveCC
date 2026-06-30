"""
Perceptual image-difference metric for the CC encoder benchmarks/tests.

S-CIELAB (Zhang & Wandell, 1996/1997) — a spatial extension of CIELAB ΔE that
prepends a human contrast-sensitivity spatial filter before the colour-difference
calculation.  That matters here because the encoder deliberately trades per-pixel
accuracy for *dithered* detail the eye integrates over neighbouring sub-pixels: a
raw per-pixel PSNR/ΔE penalises exactly the dithering that looks good, while
S-CIELAB scores what the viewer actually perceives.  This replaces the crude "2x2
box blur then PSNR" proxy the benchmarks used before.

Pipeline (per image):  sRGB -> linear -> CIE XYZ -> Poirson&Wandell opponent
(luminance / red-green / blue-yellow) -> per-channel spatial filtering (luminance
kept sharp, chroma blurred more, matching the eye's lower chroma acuity) -> back to
XYZ -> CIELAB.  ΔE is then the CIE76 Euclidean distance in CIELAB, per pixel.

The opponent matrix and the per-channel Gaussian filter parameters are taken from
the reference MATLAB sources (wandell/SCIELAB-1996: cmatrix.m, separableFilters.m,
gauss.m).  Because both images pass through the *identical* filter before the
difference, the metric is insensitive to the exact kernel support/normalisation
(any consistent bias cancels) — what it encodes is the *differential* luminance-vs-
chroma blur, which is the perceptual content of S-CIELAB.

Reference: X. Zhang and B. A. Wandell, "A spatial extension of CIELAB for digital
colour-image reproduction", SID 1996 / Journal of the SID 1997.
"""

from __future__ import annotations

import numpy as np

# Samples per degree of visual angle: how many source sub-pixels span one degree
# when the monitor is viewed.  Sets the spatial-filter scale (lower = more pooling,
# i.e. viewing the chunky display from further away).  The metric's *relative*
# ordering is robust to this; it's exposed as a parameter for tuning.
SPD_DEFAULT = 14.0

# Linear sRGB -> CIE XYZ (D65), the standard sRGB primaries matrix.
_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)   # XYZ white (Y=1)

# CIE XYZ -> Poirson & Wandell opponent space (O1 luminance, O2 red-green,
# O3 blue-yellow).  Verbatim from cmatrix.m ('xyz2opp'); the inverse is computed.
_XYZ2OPP = np.array([
    [0.2787336, 0.7218031, -0.1065520],
    [-0.4487736, 0.2898056, 0.0771569],
    [0.0859513, -0.5899859, 0.5011089],
], dtype=np.float64)
_OPP2XYZ = np.linalg.inv(_XYZ2OPP)

# Per-channel spatial filters: a weighted sum of Gaussians whose half-widths are in
# DEGREES of visual angle (scaled to pixels by samples-per-degree).  Verbatim from
# separableFilters.m as (half_width_deg, weight) pairs; the weights per channel sum
# to 1, so the filter preserves mean colour.  Luminance gets the tightest Gaussian
# (sharp); the chroma channels are blurred more (lower chroma acuity).
_FILTERS = {
    "lum": [(0.0500, 1.00327), (0.2250, 0.114416), (7.0000, -0.117686)],
    "rg":  [(0.0685, 0.616725), (0.8260, 0.383275)],
    "by":  [(0.0920, 0.567885), (0.6451, 0.432115)],
}

_TWO_SQRT_LN2 = 2.0 * np.sqrt(np.log(2.0))


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _channel_kernel(terms, spd: float, cap: int) -> np.ndarray:
    """1-D separable kernel for one opponent channel: sum of the channel's Gaussians.

    gauss(x) = exp(-(alpha*x)^2) with alpha = 2*sqrt(ln2)/(halfwidth_px - 1) (the
    gauss.m definition; halfwidth is the full width at half maximum, in pixels).
    Each Gaussian is normalised to sum 1 before weighting, so the kernel sums to 1.
    Support covers 3 sigma of the widest Gaussian, capped to `cap` (the broad
    surround term is meant to be near-flat; truncating it is harmless since both
    images are filtered identically)."""
    sigmas = [max(hw * spd - 1.0, 1e-6) / _TWO_SQRT_LN2 for hw, _ in terms]
    width = int(2 * np.ceil(3.0 * max(sigmas)) + 1)
    width = min(width, cap)
    if width % 2 == 0:
        width += 1
    x = np.arange(width, dtype=np.float64) - width // 2
    k = np.zeros(width, dtype=np.float64)
    for hw_deg, w in terms:
        alpha = _TWO_SQRT_LN2 / max(hw_deg * spd - 1.0, 1e-6)
        g = np.exp(-((alpha * x) ** 2))
        k += w * (g / g.sum())
    return k


def _conv_axis(img: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """1-D convolution along `axis` with edge replication (so a unit-sum kernel
    preserves the mean right up to the border instead of darkening it)."""
    pad = k.shape[0] // 2
    pads = [(pad, pad) if i == axis else (0, 0) for i in range(img.ndim)]
    p = np.pad(img, pads, mode="edge")
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis, p)


def _filter_opponent(opp: np.ndarray, spd: float) -> np.ndarray:
    h, w = opp.shape[:2]
    cap = 2 * min(h, w) - 1                       # can't blur wider than the image
    out = np.empty_like(opp)
    for ci, key in enumerate(("lum", "rg", "by")):
        k = _channel_kernel(_FILTERS[key], spd, cap)
        ch = _conv_axis(opp[..., ci], k, 1)       # x (separable)
        out[..., ci] = _conv_axis(ch, k, 0)       # y
    return out


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    xn = xyz / _D65
    d = 6.0 / 29.0
    f = np.where(xn > d ** 3, np.cbrt(xn), xn / (3 * d * d) + 4.0 / 29.0)
    return np.stack([116.0 * f[..., 1] - 16.0,
                     500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], axis=-1)


def _to_filtered_lab(rgb: np.ndarray, spd: float) -> np.ndarray:
    xyz = _srgb_to_linear(rgb) @ _RGB2XYZ.T
    opp = xyz @ _XYZ2OPP.T
    return _xyz_to_lab(_filter_opponent(opp, spd) @ _OPP2XYZ.T)


def scielab_delta_e(rgb1: np.ndarray, rgb2: np.ndarray,
                    samples_per_degree: float = SPD_DEFAULT) -> np.ndarray:
    """Per-pixel S-CIELAB ΔE (CIE76) between two (H,W,3) uint8 sRGB images."""
    lab1 = _to_filtered_lab(rgb1, samples_per_degree)
    lab2 = _to_filtered_lab(rgb2, samples_per_degree)
    return np.sqrt(((lab1 - lab2) ** 2).sum(-1))


def mean_scielab(rgb1: np.ndarray, rgb2: np.ndarray,
                 samples_per_degree: float = SPD_DEFAULT) -> float:
    """Mean S-CIELAB ΔE — the headline perceptual error (lower = closer to source)."""
    return float(scielab_delta_e(rgb1, rgb2, samples_per_degree).mean())

"""
Shared benchmark/render harness: timing, test-frame generators, grid presets,
tables, and developer-media discovery/decoding.

Pure stdlib + numpy (numpy is already a server dependency), so the suite runs with
nothing extra installed.  Importing this also puts the server package dir on
sys.path, so the bench modules can `import cc_encoder` / `transcoder` regardless of
the current working directory.

(Named `harness.py`, not `_bench.py`, because the repo's root .gitignore ignores
`_*.py` as scratch files — a leading underscore here would silently un-track it.)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Make the server modules importable no matter where the bench is launched from.
_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

# Shared media/decode helpers (also used by tools/render_cc.py).  Re-exported so
# the bench modules can keep importing them from `harness`.
from cc_media import (  # noqa: E402  (needs the sys.path insert above)
    GRIDS, MEDIA_DIR, PREVIEW_DIR,
    find_media, have_ffmpeg, render_cells, sample_frames,
)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

def measure(fn, *, target_s: float = 0.5, warmup: int = 3,
            min_iters: int = 5, max_iters: int = 200_000) -> dict:
    """Time `fn` repeatedly and report mean / min / median in milliseconds.

    Iteration count auto-calibrates from a single trial so a 200 ms encode and a
    2 µs primitive both get a fair sample without hand-tuning.  `min` is the
    cleanest signal (least OS noise); `mean` reflects typical cost.
    """
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    fn()
    one = time.perf_counter() - t
    iters = int(max(min_iters, min(max_iters, target_s / max(one, 1e-9))))

    times = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        s = time.perf_counter()
        fn()
        times[i] = time.perf_counter() - s
    return {
        "mean_ms": float(times.mean()) * 1e3,
        "min_ms": float(times.min()) * 1e3,
        "median_ms": float(np.median(times)) * 1e3,
        "iters": iters,
    }


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #

def section(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def table(headers, rows) -> str:
    """A compact fixed-width table; first column left-aligned, rest right-aligned."""
    cells = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))

    def line(row):
        out = [row[0].ljust(widths[0])]
        out += [row[i].rjust(widths[i]) for i in range(1, len(row))]
        return "  ".join(out)

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(list(headers)), sep, *[line(r) for r in cells]])


def fmt(x, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
# Synthetic test frames.  Each generator returns an (H*3, W*2, 3) uint8 array —
# the raw rgb24 cell grid the encoder consumes — and is deterministic (seeded).
# --------------------------------------------------------------------------- #

def _px(w: int, h: int):
    return h * 3, w * 2


def solid(w: int, h: int) -> np.ndarray:
    ph, pw = _px(w, h)
    return np.tile(np.array((90, 140, 180), np.uint8), (ph, pw, 1))


def flat_between(w: int, h: int) -> np.ndarray:
    """Vertical bands of flat tones that sit *between* palette colours — the case
    that must dither rather than band."""
    ph, pw = _px(w, h)
    tones = [(196, 196, 196), (110, 110, 110), (180, 140, 90),
             (90, 140, 180), (200, 200, 140), (60, 90, 60)]
    f = np.zeros((ph, pw, 3), np.uint8)
    for i, t in enumerate(tones):
        f[:, i * pw // len(tones):(i + 1) * pw // len(tones)] = t
    return f


def gradient(w: int, h: int) -> np.ndarray:
    """Smooth 2-D colour gradient — heavy on dithering, light on edges."""
    ph, pw = _px(w, h)
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    r = 128 + 127 * np.sin(xx / 40)
    g = 128 + 127 * np.sin(yy / 30 + 1)
    b = 128 + 127 * np.sin((xx + yy) / 50 + 2)
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def gradient_edges(w: int, h: int) -> np.ndarray:
    """Gradient with a couple of hard-colour rectangles (edge endpoints)."""
    f = gradient(w, h)
    ph, pw, _ = f.shape
    f[ph // 3:ph // 3 + max(3, ph // 12), :] = (204, 76, 76)
    f[:, pw // 2:pw // 2 + max(2, pw // 16)] = (51, 102, 204)
    return f


def photo(w: int, h: int) -> np.ndarray:
    """Video-like: low-frequency colour field + film grain + a few object edges.
    The realistic mix of smooth and structured regions — use this for headline
    numbers."""
    ph, pw = _px(w, h)
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    base = np.stack([
        128 + 80 * np.sin(xx / pw * 3.0) + 40 * np.sin(yy / ph * 2.0),
        128 + 80 * np.sin(yy / ph * 4.0 + 1.0) + 30 * np.cos(xx / pw * 2.0),
        128 + 80 * np.sin((xx + yy) / pw * 3.0 + 2.0),
    ], -1)
    base += rng.normal(0, 8, base.shape)               # grain
    base[ph // 4:ph // 2, pw // 6:pw // 3] = (210, 60, 60)   # object
    base[2 * ph // 3:, 2 * pw // 3:] = (40, 60, 160)         # object
    return np.clip(base, 0, 255).astype(np.uint8)


def random(w: int, h: int) -> np.ndarray:
    """Uniform RGB noise — the encoder's worst case (no spatial coherence)."""
    ph, pw = _px(w, h)
    return np.random.default_rng(0).integers(0, 256, (ph, pw, 3), np.uint8)


CONTENT = {
    "solid": solid,
    "flat-between": flat_between,
    "gradient": gradient,
    "gradient+edges": gradient_edges,
    "photo": photo,
    "random": random,
}


"""
rANS + RLE plane codec — the entropy stage behind CCMF's ANS frame encoding.

This is an improvement on sanjuuni's 32vid ANS (docs/cc-media-format.md §4.5.3):
we keep its *design* — encode a raw frame as separate symbol planes, with a
per-frame frequency model and a run-length pre-pass for the flat regions that
dominate real video — but swap its bit-level table-ANS (tANS) for a
**byte-renormalized range-ANS (rANS)**.  The reason is the CC/Lua client, which
is the primary decode target: Lua 5.1 on CC has no native bit operators (only
`bit32.*` function calls) and interprets under a yield budget, so a bit reader
is exactly what made deflate non-viable.  rANS over these tiny alphabets needs
only integer `//`, `%`, `*` and a byte-at-a-time refill — all of which map to
cheap `math.floor` / `%` / `string.byte` in Lua, no bit twiddling anywhere.

Layout of one encoded plane (all little-endian, byte-aligned):

    [k u8]                       number of distinct run-value symbols present
    k × [sym u8][freq u16]       normalized frequencies, sum == M, freq desc.
    [rans_len u24]               byte length of the rANS value stream
    [rans bytes]                 4-byte final state (big-endian) + renorm bytes
    [length tokens ...]          one run length per run, until n cells filled

A plane is first run-length encoded into (value, length) runs.  The run VALUES
(skewed — a few colours/glyphs dominate) go through rANS; the run LENGTHS (close
to uniform in the log domain, poor entropy targets) are plain byte tokens: a
byte < 255 is the length itself, else 0xFF followed by a u16.  Flat content
(letterboxing, solid backgrounds, borders) collapses to a handful of runs, so
the decoder does far less work per keyframe than unpacking W·H bit-packed cells.

The rANS parameters (M = 2**12 total frequency, RANS_L = 2**16 low bound, byte
rendrmalization) keep every intermediate below 2**24 < 2**53, so the whole thing
is exact in Lua's doubles.  Symbols in the frequency table are ordered by
descending frequency, and the decoder finds a symbol by a short linear scan over
the cumulative table — for these skewed distributions the dominant symbol is
hit in one or two comparisons, which beats building a 2**12-entry lookup table
per keyframe on CC.
"""

from __future__ import annotations

import numpy as np

# rANS constants.  M (total frequency) is a power of two so `x % M` / `x // M`
# are a mask/shift; RANS_L with byte renorm keeps the state in [2**16, 2**24).
_SCALE_BITS = 12
_M = 1 << _SCALE_BITS            # 4096
_RANS_L = 1 << 16
_MASK = _M - 1


def _normalize_freqs(counts: np.ndarray) -> np.ndarray:
    """Scale non-negative integer `counts` to frequencies summing to exactly _M,
    giving every present symbol at least 1.  Returns an int array the same shape
    as counts (0 stays 0 for absent symbols)."""
    counts = counts.astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros_like(counts)
    freqs = np.zeros_like(counts)
    present = counts > 0
    # Proportional scale, floored, then floor-corrected to hit _M exactly.
    scaled = counts.astype(np.float64) * _M / total
    freqs[present] = np.maximum(1, np.floor(scaled[present]).astype(np.int64))
    diff = _M - int(freqs.sum())
    # Hand the leftover (or debt) to the highest-count symbols, one unit at a
    # time, never dropping a present symbol below 1.
    order = np.argsort(-counts)
    order = order[counts[order] > 0]
    i = 0
    step = 1 if diff > 0 else -1
    while diff != 0:
        s = order[i % len(order)]
        if step < 0 and freqs[s] <= 1:
            i += 1
            continue
        freqs[s] += step
        diff -= step
        i += 1
    return freqs


def _rle(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run-length encode a 1-D symbol array -> (values, lengths)."""
    n = plane.shape[0]
    if n == 0:
        return plane[:0], plane[:0].astype(np.int64)
    changes = np.nonzero(np.diff(plane))[0]
    starts = np.concatenate(([0], changes + 1))
    ends = np.concatenate((changes + 1, [n]))
    return plane[starts], (ends - starts).astype(np.int64)


def _pack_length(out: bytearray, length: int) -> None:
    """A run-length token: one byte if < 255, else 0xFF + u16 LE."""
    if length < 255:
        out.append(length)
    else:
        out.append(0xFF)
        out.append(length & 0xFF)
        out.append((length >> 8) & 0xFF)


def encode_plane(plane: np.ndarray, nsym: int) -> bytes:
    """Encode a 1-D array of symbols in [0, nsym) -> a self-delimiting plane blob."""
    plane = np.asarray(plane, dtype=np.uint8)
    values, lengths = _rle(plane)

    counts = np.bincount(values, minlength=nsym).astype(np.int64)
    freqs = _normalize_freqs(counts)

    # Frequency table, symbols ordered by descending frequency (fast linear-scan
    # decode).  cum[] is the exclusive prefix sum in that same order.
    present = np.nonzero(freqs)[0]
    present = present[np.argsort(-freqs[present])]
    freq = {int(s): int(freqs[s]) for s in present}
    cum: dict[int, int] = {}
    acc = 0
    for s in present:
        cum[int(s)] = acc
        acc += int(freqs[s])

    # rANS-encode the run values in reverse (decoder emits them forward).
    x = _RANS_L
    buf = bytearray()
    for s in values[::-1]:
        s = int(s)
        f = freq[s]
        x_max = f << (16 - _SCALE_BITS + 8)      # ((RANS_L >> SCALE) << 8) * f
        while x >= x_max:
            buf.append(x & 0xFF)
            x >>= 8
        x = (x // f) * _M + (x % f) + cum[s]
    for i in range(4):                            # final state, low byte first
        buf.append((x >> (8 * i)) & 0xFF)
    rans = bytes(reversed(buf))                   # reverse -> forward-readable

    head = bytearray()
    head.append(len(present))
    for s in present:
        s = int(s)
        head.append(s)
        head.append(freq[s] & 0xFF)
        head.append((freq[s] >> 8) & 0xFF)
    head.append(len(rans) & 0xFF)
    head.append((len(rans) >> 8) & 0xFF)
    head.append((len(rans) >> 16) & 0xFF)

    lens = bytearray()
    for L in lengths:
        _pack_length(lens, int(L))

    return bytes(head) + rans + bytes(lens)


def decode_plane(data: bytes, offset: int, n: int, nsym: int) -> tuple[np.ndarray, int]:
    """Decode one plane blob at `data[offset:]` for `n` cells -> (plane, next_offset).
    Mirrors the Lua/C++ decoders exactly (linear-scan symbol lookup, byte renorm)."""
    pos = offset
    k = data[pos]; pos += 1
    syms = [0] * k
    freqs = [0] * k
    cums = [0] * k
    acc = 0
    for i in range(k):
        s = data[pos]
        f = data[pos + 1] | (data[pos + 2] << 8)
        pos += 3
        syms[i] = s
        freqs[i] = f
        cums[i] = acc
        acc += f
    rans_len = data[pos] | (data[pos + 1] << 8) | (data[pos + 2] << 16)
    pos += 3
    rp = pos                                       # rANS byte cursor
    lp = pos + rans_len                            # length-token cursor
    pos = lp                                        # (block end updated below)

    # Initial state: 4 bytes big-endian (see encode_plane's reversal).
    x = (data[rp] << 24) | (data[rp + 1] << 16) | (data[rp + 2] << 8) | data[rp + 3]
    rp += 4

    out = np.empty(n, dtype=np.uint8)
    cells = 0
    while cells < n:
        slot = x & _MASK
        # Find the symbol whose cumulative interval contains slot (descending
        # frequency order -> the common symbols are found first).
        i = 0
        while i + 1 < k and cums[i + 1] <= slot:
            i += 1
        s = syms[i]
        x = freqs[i] * (x >> _SCALE_BITS) + slot - cums[i]
        while x < _RANS_L:
            x = (x << 8) | data[rp]
            rp += 1

        b = data[lp]; lp += 1
        if b < 255:
            length = b
        else:
            length = data[lp] | (data[lp + 1] << 8)
            lp += 2

        out[cells:cells + length] = s
        cells += length

    return out, lp

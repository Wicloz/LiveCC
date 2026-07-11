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

    [mode u8]                    0 = RLE+rANS, 1 = plain rANS
    [k u8]                       number of distinct symbols present
    k × [sym u8][freq u16]       normalized frequencies, sum == M, freq desc.
    [rans_len u24]               byte length of the rANS stream
    [rans bytes]                 4-byte final state (big-endian) + renorm bytes
    [length tokens ...]          (mode 0 only) one run length per run

Two per-plane modes, encoder picks the smaller:

  * **mode 0 (RLE+rANS)** — the plane is run-length encoded into (value, length)
    runs; the run VALUES (skewed — a few colours/glyphs dominate) go through
    rANS, and the run LENGTHS (near-uniform in the log domain, poor entropy
    targets) are plain byte tokens: a byte < 255 is the length, else 0xFF + u16.
    Flat content (letterboxing, solid fills) collapses to a handful of runs.
  * **mode 1 (plain rANS)** — every cell is rANS-coded directly, no run lengths.
    This wins on high-detail/dithered content where runs degenerate to length 1
    and mode 0's one-byte-per-run length tokens would dominate (there, mode 0 can
    exceed even the bit-packed `raw` plane; mode 1 stays near the entropy).

So flat planes keep the big RLE win and noisy planes fall back to pure entropy
coding — the plane is never bloated by run-length overhead it can't amortise.

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


def _build_table(counts: np.ndarray) -> tuple[dict, dict, bytes]:
    """Normalize `counts` to sum _M and serialize the frequency table (symbols
    frequency-descending) -> (freq{sym:f}, cum{sym:exclusive-prefix}, head bytes)."""
    freqs = _normalize_freqs(counts)
    present = np.nonzero(freqs)[0]
    present = present[np.argsort(-freqs[present])]
    freq = {int(s): int(freqs[s]) for s in present}
    cum: dict[int, int] = {}
    acc = 0
    head = bytearray([len(present)])
    for s in present:
        s = int(s)
        cum[s] = acc
        acc += freq[s]
        head.append(s)
        head.append(freq[s] & 0xFF)
        head.append((freq[s] >> 8) & 0xFF)
    return freq, cum, bytes(head)


def _rans_encode(symbols: np.ndarray, freq: dict, cum: dict) -> bytes:
    """rANS-encode a symbol array in reverse (decoder emits forward) -> the byte
    stream: 4-byte final state (big-endian, after the whole-buffer reversal) then
    renorm bytes."""
    x = _RANS_L
    buf = bytearray()
    for s in symbols[::-1]:
        s = int(s)
        f = freq[s]
        x_max = f << (16 - _SCALE_BITS + 8)      # ((RANS_L >> SCALE) << 8) * f
        while x >= x_max:
            buf.append(x & 0xFF)
            x >>= 8
        x = (x // f) * _M + (x % f) + cum[s]
    for i in range(4):                            # final state, low byte first
        buf.append((x >> (8 * i)) & 0xFF)
    return bytes(reversed(buf))                   # reverse -> forward-readable


def _rans_len_bytes(rans: bytes) -> bytes:
    return bytes([len(rans) & 0xFF, (len(rans) >> 8) & 0xFF, (len(rans) >> 16) & 0xFF])


def encode_plane(plane: np.ndarray, nsym: int) -> bytes:
    """Encode a 1-D array of symbols in [0, nsym) -> a self-delimiting plane blob,
    picking the smaller of mode 0 (RLE+rANS) and mode 1 (plain rANS)."""
    plane = np.asarray(plane, dtype=np.uint8)

    # mode 1: plain rANS over every cell (no run lengths).
    freq1, cum1, head1 = _build_table(np.bincount(plane, minlength=nsym).astype(np.int64))
    rans1 = _rans_encode(plane, freq1, cum1)
    m1 = b"\x01" + head1 + _rans_len_bytes(rans1) + rans1

    # mode 0: RLE, rANS over the run values, run lengths as byte tokens.
    values, lengths = _rle(plane)
    freq0, cum0, head0 = _build_table(np.bincount(values, minlength=nsym).astype(np.int64))
    rans0 = _rans_encode(values, freq0, cum0)
    lens = bytearray()
    for L in lengths:
        _pack_length(lens, int(L))
    m0 = b"\x00" + head0 + _rans_len_bytes(rans0) + rans0 + bytes(lens)

    return m0 if len(m0) <= len(m1) else m1


def _read_table(data: bytes, pos: int) -> tuple[list, list, list, int, int]:
    """Read a frequency table + rANS length -> (syms, freqs, cums, rans_start, next)."""
    k = data[pos]; pos += 1
    syms = [0] * k
    freqs = [0] * k
    cums = [0] * k
    acc = 0
    for i in range(k):
        syms[i] = data[pos]
        freqs[i] = data[pos + 1] | (data[pos + 2] << 8)
        cums[i] = acc
        acc += freqs[i]
        pos += 3
    rans_len = data[pos] | (data[pos + 1] << 8) | (data[pos + 2] << 16)
    pos += 3
    return syms, freqs, cums, pos, pos + rans_len


def decode_plane(data: bytes, offset: int, n: int, nsym: int) -> tuple[np.ndarray, int]:
    """Decode one plane blob at `data[offset:]` for `n` cells -> (plane, next_offset).
    Mirrors the Lua/C++ decoders exactly (linear-scan symbol lookup, byte renorm)."""
    mode = data[offset]
    syms, freqs, cums, rp, lp = _read_table(data, offset + 1)
    k = len(syms)

    # Initial state: 4 bytes big-endian (see _rans_encode's reversal).
    x = (data[rp] << 24) | (data[rp + 1] << 16) | (data[rp + 2] << 8) | data[rp + 3]
    rp += 4

    out = np.empty(n, dtype=np.uint8)
    if mode == 1:                                  # plain rANS: one symbol per cell
        for i in range(n):
            slot = x & _MASK
            j = 0
            while j + 1 < k and cums[j + 1] <= slot:
                j += 1
            x = freqs[j] * (x >> _SCALE_BITS) + slot - cums[j]
            out[i] = syms[j]
            while x < _RANS_L:
                x = (x << 8) | data[rp]
                rp += 1
        return out, lp                             # rANS stream end == lp

    cells = 0                                       # mode 0: RLE + rANS run values
    while cells < n:
        slot = x & _MASK
        j = 0
        while j + 1 < k and cums[j + 1] <= slot:
            j += 1
        s = syms[j]
        x = freqs[j] * (x >> _SCALE_BITS) + slot - cums[j]
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

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
    [filter u8]                  0 none · 1 sub (left) · 2 up — spatial predictor
    [mode u8]                    0 = RLE+rANS, 1 = plain rANS
    [k u8]                       number of distinct symbols present
    k × [sym u8][freq u16]       normalized frequencies, sum == M, freq desc.
    [rans_len u24]               byte length of the rANS stream
    [rans bytes]                 4-byte final state (big-endian) + renorm bytes
    [length tokens ...]          (mode 0 only) one run length per run

Before entropy coding, an optional per-plane **spatial predictor** (PNG's Sub /
Up filters, generalised to palette indices, which aren't ordered — so the
"residual" is a MATCH token when a cell equals its neighbour, else the literal):

  * **filter 1 (sub / left)** — MATCH (symbol value = nsym) where a cell equals
    the cell to its left (first column stays literal);
  * **filter 2 (up)** — MATCH where a cell equals the cell above (first row
    literal).

MATCH-dominated streams entropy-code the run *structure* far cheaper than mode
0's raw length bytes, so this shrinks structured keyframes ~8%.  The encoder
picks the filter by a cheap size ESTIMATE of each candidate (entropy + run
count — no trial encoding), which matches the true best-of choice, so encoding
stays ~one pass.  The decoder reverses the predictor after the entropy decode.

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

# Per-plane spatial predictor (a MATCH token, value == nsym, marks equality).
FILTER_NONE = 0
FILTER_SUB = 1                   # equals the cell to the left
FILTER_UP = 2                   # equals the cell above


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


def _encode_stream(stream: np.ndarray, alpha: int) -> bytes:
    """Entropy-code a symbol stream over [0, alpha) -> [mode][table][rans][lengths],
    picking the smaller of mode 0 (RLE+rANS) and mode 1 (plain rANS)."""
    # mode 1: plain rANS over every symbol (no run lengths).
    freq1, cum1, head1 = _build_table(np.bincount(stream, minlength=alpha).astype(np.int64))
    rans1 = _rans_encode(stream, freq1, cum1)
    m1 = b"\x01" + head1 + _rans_len_bytes(rans1) + rans1

    # mode 0: RLE, rANS over the run values, run lengths as byte tokens.
    values, lengths = _rle(stream)
    freq0, cum0, head0 = _build_table(np.bincount(values, minlength=alpha).astype(np.int64))
    rans0 = _rans_encode(values, freq0, cum0)
    lens = bytearray()
    for L in lengths:
        _pack_length(lens, int(L))
    m0 = b"\x00" + head0 + _rans_len_bytes(rans0) + rans0 + bytes(lens)

    return m0 if len(m0) <= len(m1) else m1


def _apply_filter(plane: np.ndarray, nsym: int, width: int, filt: int) -> np.ndarray:
    """Spatial predictor: replace a cell with a MATCH token (value nsym) where it
    equals its left (sub) or upper (up) neighbour.  `plane` is 1-D row-major."""
    grid = plane.reshape(-1, width)
    out = grid.astype(np.int16).copy()
    if filt == FILTER_SUB:
        out[:, 1:] = np.where(grid[:, 1:] == grid[:, :-1], nsym, grid[:, 1:])
    elif filt == FILTER_UP:
        out[1:, :] = np.where(grid[1:, :] == grid[:-1, :], nsym, grid[1:, :])
    return out.ravel().astype(np.uint8)


def _estimate_size(stream: np.ndarray, alpha: int) -> int:
    """Cheap size estimate of _encode_stream (entropy + run count, no rANS pass) —
    the heuristic that picks the filter without trial-encoding each candidate."""
    n = stream.shape[0]
    if n == 0:
        return 0
    counts = np.bincount(stream, minlength=alpha)
    nz = counts[counts > 0]
    p = nz / n
    plain = 1 + len(nz) * 3 + int(np.ceil(-(p * np.log2(p)).sum() * n / 8)) + 4
    changes = np.flatnonzero(np.diff(stream.astype(np.int16)))
    nruns = changes.size + 1
    rvals = stream[np.concatenate(([0], changes + 1))]
    rc = np.bincount(rvals, minlength=alpha)
    rnz = rc[rc > 0]
    rp = rnz / nruns
    rle = (1 + len(rnz) * 3 + int(np.ceil(-(rp * np.log2(rp)).sum() * nruns / 8))
           + 4 + nruns)
    return min(plain, rle)


def encode_plane(plane: np.ndarray, nsym: int, width: int = 0) -> bytes:
    """Encode a 1-D array of symbols in [0, nsym) -> a self-delimiting plane blob.

    `width` (the grid row length) enables the spatial predictors (Sub/Up); when
    it's 0/absent, or the plane is too small to have rows, the filter stays
    `none`.  The filter is chosen by a cheap size estimate (no trial encoding),
    then the winner is entropy-coded once."""
    plane = np.asarray(plane, dtype=np.uint8).ravel()
    filt, stream, alpha = FILTER_NONE, plane, nsym
    if width >= 2 and plane.shape[0] >= 2 * width:
        best = _estimate_size(plane, nsym)
        for cand in (FILTER_SUB, FILTER_UP):
            filtered = _apply_filter(plane, nsym, width, cand)
            est = _estimate_size(filtered, nsym + 1)
            if est < best:
                best, filt, stream, alpha = est, cand, filtered, nsym + 1
    return bytes([filt]) + _encode_stream(stream, alpha)


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


def _undo_filter(stream: np.ndarray, nsym: int, width: int, filt: int) -> np.ndarray:
    """Reverse _apply_filter: replace each MATCH token (value nsym) with the
    already-reconstructed left (sub) or upper (up) neighbour."""
    grid = stream.reshape(-1, width).astype(np.int16)
    if filt == FILTER_SUB:
        for x in range(1, width):                # intra-row dependency: left to right
            col = grid[:, x]
            grid[:, x] = np.where(col == nsym, grid[:, x - 1], col)
    elif filt == FILTER_UP:
        for y in range(1, grid.shape[0]):        # each row depends on the one above
            row = grid[y, :]
            grid[y, :] = np.where(row == nsym, grid[y - 1, :], row)
    return grid.ravel().astype(np.uint8)


def decode_plane(data: bytes, offset: int, n: int, nsym: int,
                 width: int = 0) -> tuple[np.ndarray, int]:
    """Decode one plane blob at `data[offset:]` for `n` cells -> (plane, next_offset).
    Mirrors the Lua/C++ decoders exactly (linear-scan symbol lookup, byte renorm)."""
    filt = data[offset]
    out, end = _decode_stream(data, offset + 1, n)
    if filt != FILTER_NONE:
        out = _undo_filter(out, nsym, width, filt)
    return out, end


def _decode_stream(data: bytes, offset: int, n: int) -> tuple[np.ndarray, int]:
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

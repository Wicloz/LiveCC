"""
DFPWM1a codec — a byte-exact port of CC:Tweaked's `cc.audio.dfpwm` [DFPWM].

Why encode in Python instead of ffmpeg: the spec (docs/cc-media-format.md §4.6)
requires the DECODER state to reset at every chunk so each chunk is
independently decodable, and the CC client does exactly that.  ffmpeg encodes
one continuous stream, so slicing its output into chunks leaves the decoder
resyncing against mid-stream encoder state at every chunk boundary (a periodic
artifact).  Encoding here, per chunk with fresh state, matches the contract —
and lets every channel role be cut from ONE shared PCM decode (sample-aligned)
instead of one ffmpeg pipeline per role.

encode()/decode() take/return the wire formats: unsigned 8-bit PCM (the CC
speaker amplitude + 128) in, packed DFPWM out, LSB = first sample.  Both start
from fresh codec state on every call — one call is one CCMF chunk.

The per-sample loops are sequential by nature (each bit depends on the last),
so they're plain loops, compiled with numba when available (a prod dependency,
same policy as cc_encoder) and still fast enough interpreted (~1 ms per 0.1 s
chunk) for the fallback to be a safety net rather than a trap.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:                          # pragma: no cover - numba is a prod dep
    _HAVE_NUMBA = False

# DFPWM1a predictor constants (cc.audio.dfpwm).
_PREC_POW = 1 << 10          # PREC = 10
_PREC_POW_HALF = 1 << 9
_STRENGTH_MIN = 1 << 3       # 2^(PREC - 8 + 1)


def _encode_kernel(levels: np.ndarray, out: np.ndarray) -> None:
    # levels: signed int32 (-128..127), length a multiple of 8; out: uint8, len/8.
    charge = 0
    strength = 0
    prev_bit = False
    byte = 0
    for i in range(levels.shape[0]):
        level = levels[i]
        bit = level > charge or (level == charge and level == 127)
        byte = (byte >> 1) + (128 if bit else 0)       # LSB-first packing

        target = 127 if bit else -128
        next_charge = charge + ((strength * (target - charge) + _PREC_POW_HALF)
                                >> 10)
        if next_charge == charge and next_charge != target:
            next_charge += 1 if bit else -1
        z = _PREC_POW - 1 if bit == prev_bit else 0
        next_strength = strength
        if next_strength != z:
            next_strength += 1 if bit == prev_bit else -1
        if next_strength < _STRENGTH_MIN:
            next_strength = _STRENGTH_MIN
        charge, strength, prev_bit = next_charge, next_strength, bit

        if i & 7 == 7:
            out[i >> 3] = byte
            byte = 0


def _decode_kernel(data: np.ndarray, out: np.ndarray) -> None:
    # data: uint8 DFPWM bytes; out: int32, len*8 (signed levels before clamping).
    charge = 0
    strength = 0
    prev_bit = False
    prev_charge = 0
    low_pass = 0
    for i in range(data.shape[0]):
        byte = int(data[i])
        for j in range(8):
            bit = byte & 1 != 0
            byte >>= 1

            target = 127 if bit else -128
            next_charge = charge + ((strength * (target - charge) + _PREC_POW_HALF)
                                    >> 10)
            if next_charge == charge and next_charge != target:
                next_charge += 1 if bit else -1
            z = _PREC_POW - 1 if bit == prev_bit else 0
            next_strength = strength
            if next_strength != z:
                next_strength += 1 if bit == prev_bit else -1
            if next_strength < _STRENGTH_MIN:
                next_strength = _STRENGTH_MIN

            antijerk = next_charge
            if bit != prev_bit:
                antijerk = (next_charge + prev_charge + 1) >> 1

            charge, strength, prev_bit = next_charge, next_strength, bit
            prev_charge = next_charge

            low_pass += ((antijerk - low_pass) * 140 + 128) >> 8
            out[i * 8 + j] = low_pass


if _HAVE_NUMBA:
    _encode_kernel = njit(cache=True, nogil=True)(_encode_kernel)
    _decode_kernel = njit(cache=True, nogil=True)(_decode_kernel)


def encode(pcm_u8: bytes) -> bytes:
    """Unsigned 8-bit PCM -> DFPWM, fresh encoder state (one call = one chunk).

    Input is padded to a multiple of 8 samples with silence (128), matching the
    Lua encoder's `input[i + j] or 0` — only ever relevant for a stream's final
    partial chunk.
    """
    levels = np.frombuffer(pcm_u8, np.uint8).astype(np.int32) - 128
    pad = (-levels.size) % 8
    if pad:
        levels = np.concatenate([levels, np.zeros(pad, np.int32)])
    out = np.empty(levels.size // 8, np.uint8)
    _encode_kernel(levels, out)
    return out.tobytes()


def decode(data: bytes) -> bytes:
    """DFPWM -> unsigned 8-bit PCM, fresh decoder state (reference/tests; the
    live decode happens on the CC client)."""
    packed = np.frombuffer(data, np.uint8)
    out = np.empty(packed.size * 8, np.int32)
    _decode_kernel(packed, out)
    return (np.clip(out, -128, 127) + 128).astype(np.uint8).tobytes()

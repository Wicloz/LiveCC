"""
DFPWM1a codec (dfpwm.py) — the Python port of CC:Tweaked's cc.audio.dfpwm.

The decisive test cross-validates against ffmpeg's dfpwm implementation (an
independent codebase): our encoder's bitstream must decode to the source
signal there, and our reference decoder must agree with ffmpeg sample-for-
sample.  If both sides only round-tripped against each other, a shared
constant error would pass silently and real CC clients would get noise.
"""

import shutil
import subprocess

import numpy as np
import pytest

import dfpwm


def _sine_u8(seconds=0.5, freq=440, rate=48000, amp=100) -> bytes:
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * freq * t) * amp + 128).astype(np.uint8).tobytes()


def _ffmpeg_decode_dfpwm(data: bytes) -> bytes | None:
    if shutil.which("ffmpeg") is None:
        return None
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "dfpwm", "-ar", "48000", "-ac", "1", "-i", "pipe:0",
         "-f", "u8", "pipe:1"],
        input=data, capture_output=True)
    return proc.stdout if proc.returncode == 0 else None


def test_encode_packs_one_bit_per_sample():
    out = dfpwm.encode(bytes([255] * 64))
    assert len(out) == 8


def test_encode_pads_a_partial_final_chunk():
    assert len(dfpwm.encode(bytes([128] * 12))) == 2   # 12 samples -> 2 bytes


def test_first_sample_lands_in_the_low_bit():
    # LSB-first packing (the CC decoder reads bit 0 first): a loud first
    # sample against fresh state (charge 0) must set bit 0 of byte 0.
    out = dfpwm.encode(bytes([255] + [0] * 7))
    assert out[0] & 1


def test_own_roundtrip_tracks_the_source():
    src = _sine_u8()
    out = dfpwm.decode(dfpwm.encode(src))
    assert len(out) == len(src)
    a = np.frombuffer(src, np.uint8)[500:].astype(np.float64)
    b = np.frombuffer(out, np.uint8)[500:].astype(np.float64)
    assert np.corrcoef(a, b)[0, 1] > 0.95


@pytest.mark.skipif(_ffmpeg_decode_dfpwm(b"\x00" * 8) is None,
                    reason="ffmpeg without dfpwm support")
def test_bitstream_is_valid_on_an_independent_implementation():
    src = _sine_u8(1.0)
    enc = dfpwm.encode(src)
    ff = _ffmpeg_decode_dfpwm(enc)
    assert ff is not None

    ref = np.frombuffer(src, np.uint8).astype(np.float64) - 128
    got = np.frombuffer(ff, np.uint8).astype(np.float64) - 128
    n = min(len(ref), len(got))
    assert np.corrcoef(ref[500:n], got[500:n])[0, 1] > 0.95

    # And our reference decoder agrees with ffmpeg's sample-for-sample.
    own = np.frombuffer(dfpwm.decode(enc), np.uint8).astype(np.float64) - 128
    assert np.array_equal(own[:n], got[:n])

"""
Chunk-compression benchmark over the developer clips in media/.

The container lets a producer compress each chunk payload (spec §4.1.2); the CC
client can only inflate a **byte-oriented** codec (no native bit ops), which is
why LZ4 is the streaming default and deflate/zstd are native-only.  This bench
answers two questions on *real* payloads (not synthetic data, which badly
misrepresents them):

  1. How do the practical compressors actually do, as a DISTRIBUTION across
     chunks — LZ4 / LZ4HC (CC-decodable) vs deflate / zstd / brotli / lzma / bz2
     (native-only ceiling)?
  2. Is there another CC-suitable (byte-level) option worth having?  We already
     ship a byte-renormalized rANS decoder on the CC client (the ANS keyframes,
     spec §4.5.3), so a general **order-0 / order-1 byte-rANS** chunk codec is
     the natural candidate.  rANS reaches within <0.1% of the model's entropy,
     so we report its achievable size from the measured order-0/1 entropy (plus
     the order-0 frequency-table overhead) — an honest bound without shipping a
     second encoder just to benchmark it.

Payloads are gathered by TYPE, because they behave completely differently:
  * video (packed GOPs)  — bit-packed keyframes + delta span-lists
  * video (ANS GOPs)     — already entropy-coded (should barely compress)
  * audio PCM8           — 8-bit samples
  * audio DFPWM          — already an adaptive 1-bit codec (near-incompressible)
Raw clips are run through the real front-end (sample_frames + GopEncoder / an
ffmpeg audio decode); any existing .ccmf files are read directly (parse_chunk
inflates, so we re-measure on the raw payload).  Needs ffmpeg for the raw clips;
the .ccmf path and the whole bench self-skip cleanly without it.

Run:  python benchmarks/bench_compression.py
"""

from __future__ import annotations

import math
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np

import harness
from harness import fmt, section, table

import ccmf
import dfpwm
from cc_encoder import GopEncoder, VideoConfig

import lz4.block                                     # a server dependency

# Optional stronger compressors — the native-only ceiling.  bz2/lzma are stdlib;
# zstd/brotli may be absent (the suite is "stdlib + numpy"), so guard them.
try:
    import bz2
except Exception:                                    # pragma: no cover
    bz2 = None
try:
    import lzma
except Exception:                                    # pragma: no cover
    lzma = None
try:
    import zstandard
    _ZSTD = zstandard.ZstdCompressor(level=19)
except Exception:                                    # pragma: no cover
    _ZSTD = None
try:
    import brotli
except Exception:                                    # pragma: no cover
    brotli = None


_GRID = (51, 19)                 # the default terminal grid (spec §5.4)
_FPS = 24
_FRAMES_PER_CLIP = 72            # ~3 s -> a couple of GOPs with real deltas
_AUDIO_CHUNK = 96000             # 2 s at 48 kHz, matching AUDIO_CHUNK_SECONDS
_MAX_AUDIO_CHUNKS = 8


# --------------------------------------------------------------------------- #
# Compressors: real (measured bytes) + byte-rANS (entropy-derived achievable).
# --------------------------------------------------------------------------- #

def _rans_table_bytes(counts: np.ndarray) -> int:
    """Overhead of an order-0 rANS frequency table in our own plane format:
    [k u8] then [sym u8][freq u16] per present symbol (rans.py)."""
    return 1 + int((counts > 0).sum()) * 3


def rans0_size(data: bytes) -> int:
    """Achievable size of an order-0 byte-rANS: table + ceil(N·H0/8) + 4-byte
    state.  rANS codes within <0.1% of H0, so this is a faithful bound."""
    n = len(data)
    if n == 0:
        return 0
    counts = np.bincount(np.frombuffer(data, np.uint8), minlength=256).astype(np.float64)
    p = counts[counts > 0] / n
    h0 = float(-(p * np.log2(p)).sum())              # bits/byte
    return _rans_table_bytes(counts) + math.ceil(n * h0 / 8) + 4


def rans1_size(data: bytes) -> int:
    """Achievable size of an ADAPTIVE order-1 byte-rANS (context = previous
    byte): ceil(N·H1/8) + 4.  Adaptive => no per-context table on the wire, but
    the decoder must update counts per symbol (heavier on CC than order-0)."""
    n = len(data)
    if n < 2:
        return rans0_size(data)
    a = np.frombuffer(data, np.uint8).astype(np.int64)
    joint = np.bincount(a[:-1] * 256 + a[1:], minlength=256 * 256).reshape(256, 256).astype(np.float64)
    ctx = joint.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(joint > 0, joint / ctx, 1.0)
        bits = float(-(joint * np.log2(cond)).sum())  # total bits for bytes 2..N
    return math.ceil((bits + 8) / 8) + 4              # +8: first byte ~raw


# name -> (fn(bytes)->size, cc_suitable, is_estimate)
def _real(fn):
    return lambda b: len(fn(b))


def _lz4(b: bytes) -> bytes:
    return lz4.block.compress(b, store_size=False)


COMPRESSORS: dict[str, tuple] = {
    "lz4":        (_real(_lz4), True, False),
    "lz4hc":      (_real(lambda b: lz4.block.compress(b, mode="high_compression",
                                                      compression=12, store_size=False)), True, False),
    "rANS0":      (rans0_size, True, True),
    # LZ4 (matching) THEN order-0 rANS on its block — both byte-level, so the CC
    # client can inflate both passes.  This is the practical CC-suitable codec
    # that adds the entropy coding LZ4 lacks (a byte-oriented cousin of deflate).
    "lz4>rANS0":  (lambda b: rans0_size(_lz4(b)), True, True),
    # Order-1 (context = previous byte).  Reported as an informational FLOOR: an
    # adaptive coder would pay cold-start on each independent RAP chunk and a
    # static one a 64K-entry table, so this ideal isn't achievable per chunk.
    "rANS1":      (rans1_size, True, True),
    "deflate":    (_real(lambda b: zlib.compress(b, 9)), False, False),
}
if _ZSTD is not None:
    COMPRESSORS["zstd19"] = (_real(_ZSTD.compress), False, False)
if brotli is not None:
    COMPRESSORS["brotli"] = (_real(lambda b: brotli.compress(b, quality=11)), False, False)
if lzma is not None:
    COMPRESSORS["lzma"] = (_real(lambda b: lzma.compress(b, preset=9)), False, False)
if bz2 is not None:
    COMPRESSORS["bz2"] = (_real(lambda b: bz2.compress(b, 9)), False, False)


# --------------------------------------------------------------------------- #
# Payload gathering
# --------------------------------------------------------------------------- #

def _video_payloads(path: Path, use_ans: bool) -> list[bytes]:
    """Real GOP payloads from a clip: decode frames, run the actual GopEncoder
    (compression NONE, so the emitted chunk carries the raw payload)."""
    frames = harness.sample_frames(path, *_GRID, fps=_FPS, limit=_FRAMES_PER_CLIP)
    if not frames:
        return []
    gop = GopEncoder(config=VideoConfig(use_ans=use_ans))
    out: list[bytes] = []
    for i, f in enumerate(frames):
        done = gop.add(round(i * ccmf.SAMPLE_RATE / _FPS), f)
        if done is not None:
            out.append(ccmf.parse_chunk(done[1])[2])
    done = gop.flush(round(len(frames) * ccmf.SAMPLE_RATE / _FPS))
    if done is not None:
        out.append(ccmf.parse_chunk(done[1])[2])
    return out


def _decode_pcm(path: Path) -> bytes:
    """Decode a clip's audio to unsigned-8-bit mono 48 kHz (the PCM8 wire form)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn",
           "-ac", "1", "-ar", "48000", "-f", "u8", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=120).stdout
    except Exception:
        return b""


def _audio_payloads(path: Path) -> tuple[list[bytes], list[bytes]]:
    """(pcm8_payloads, dfpwm_payloads) for the clip's first few audio chunks."""
    pcm = _decode_pcm(path)
    pcm8, dfp = [], []
    for i in range(0, len(pcm) - 1, _AUDIO_CHUNK):
        if len(pcm8) >= _MAX_AUDIO_CHUNKS:
            break
        block = pcm[i:i + _AUDIO_CHUNK]
        pcm8.append(ccmf.audio_payload(ccmf.CODEC_PCM8, block))
        dfp.append(ccmf.audio_payload(ccmf.CODEC_DFPWM, dfpwm.encode(block)))
    return pcm8, dfp


def _ccmf_payloads(path: Path, buckets: dict) -> None:
    """Classify the chunks of an existing .ccmf file into buckets (parse_chunk
    inflates, so we bucket the RAW payloads)."""
    data = path.read_bytes()
    for _pts, ctype, payload in ccmf.iter_chunks(data):
        if ctype == ccmf.TYPE_VIDEO:
            # rebuild the on-wire chunk bytes to reuse the ANS peek
            key = "video-ANS" if _payload_is_ans(payload) else "video-packed"
            buckets[key].append(payload)
        elif ctype == ccmf.TYPE_AUDIO:
            codec, _ch, _ = ccmf.parse_audio_payload(payload)
            buckets["audio-dfpwm" if codec == ccmf.CODEC_DFPWM else "audio-pcm8"].append(payload)


def _payload_is_ans(payload: bytes) -> bool:
    """Does a (decompressed) video payload's first frame unit use ENC_RAW_ANS?"""
    pos = 4
    while pos < len(payload):
        flags = payload[pos]; pos += 1
        if not flags & 0x80:
            pos += 48
            continue
        return ((flags >> 4) & 0x07) == ccmf.ENC_RAW_ANS
    return False


def gather() -> dict[str, list[bytes]]:
    buckets: dict[str, list[bytes]] = {
        "video-packed": [], "video-ANS": [], "audio-pcm8": [], "audio-dfpwm": []}
    have_ff = harness.have_ffmpeg()
    for path in harness.find_media():
        streams = _media_streams(path)
        if have_ff and "video" in streams:
            buckets["video-packed"] += _video_payloads(path, use_ans=False)
            buckets["video-ANS"] += _video_payloads(path, use_ans=True)
        if have_ff and "audio" in streams:
            pcm8, dfp = _audio_payloads(path)
            buckets["audio-pcm8"] += pcm8
            buckets["audio-dfpwm"] += dfp
    # Existing .ccmf outputs (e.g. showcase.ccmf) — real production chunks, no
    # ffmpeg needed.
    if harness.MEDIA_DIR.is_dir():
        for path in sorted(harness.MEDIA_DIR.glob("*.ccmf")):
            _ccmf_payloads(path, buckets)
    return {k: v for k, v in buckets.items() if v}


def _media_streams(path: Path) -> set:
    from cc_media import media_streams
    try:
        return media_streams(path)
    except Exception:
        return set()


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _pct(values) -> tuple:
    """(p10, median, p90) of a list, as percentages already scaled by caller."""
    a = np.asarray(values, np.float64)
    return (float(np.percentile(a, 10)), float(np.median(a)), float(np.percentile(a, 90)))


def _report_type(name: str, payloads: list[bytes]) -> None:
    raw = sum(len(p) for p in payloads)
    section(f"{name}   ({len(payloads)} chunks, {raw/1024:.0f} KiB raw)")

    # Per compressor: ratio-vs-raw distribution + total, and total-vs-lz4.
    lz4_total = sum(COMPRESSORS["lz4"][0](p) for p in payloads)
    rows = []
    for cname, (fn, cc, est) in COMPRESSORS.items():
        ratios = [100.0 * fn(p) / len(p) for p in payloads if p]
        total = sum(fn(p) for p in payloads)
        p10, med, p90 = _pct(ratios)
        tag = "byte/CC" if cc else "native"
        rows.append([cname + ("*" if est else ""), tag,
                     fmt(med, 1), f"{fmt(p10,0)}-{fmt(p90,0)}",
                     fmt(100.0 * total / raw, 1), fmt(100.0 * total / lz4_total, 0)])
    print(table(["codec", "class", "med%", "p10-p90", "all%", "vs lz4"], rows))


def _report_throughput(payloads: list[bytes]) -> None:
    """Compress throughput (MB/s) for the REAL codecs — a rough guide to native
    encode cost (decode is faster; CC decode is a separate, algorithmic concern)."""
    blob = b"".join(payloads)[: 4 << 20]
    if not blob:
        return
    rows = []
    for cname, (fn, cc, est) in COMPRESSORS.items():
        if est:
            continue
        t = time.perf_counter()
        n = 0
        while time.perf_counter() - t < 0.3:
            fn(blob); n += 1
        mbps = n * len(blob) / (time.perf_counter() - t) / 1e6
        rows.append([cname, "byte/CC" if cc else "native", fmt(mbps, 0)])
    section("compress throughput  (all payloads concatenated, MB/s)")
    print(table(["codec", "class", "MB/s"], rows))


def main() -> None:
    if not harness.have_ffmpeg() and not list(harness.MEDIA_DIR.glob("*.ccmf")
                                              if harness.MEDIA_DIR.is_dir() else []):
        print("bench_compression: need ffmpeg (for raw clips) or a .ccmf file in "
              f"{harness.MEDIA_DIR} — skipping.")
        return
    buckets = gather()
    if not buckets:
        print(f"bench_compression: no usable media in {harness.MEDIA_DIR} — skipping.")
        return

    print("Per-chunk size as % of the raw payload (lower is better); 'vs lz4' is "
          "total size vs LZ4.\nclass byte/CC = inflatable in pure Lua on the CC "
          "client (no bit ops); native = not.\nrANS rows are entropy-derived "
          "achievable sizes (byte-rANS reaches within <0.1% of the\nmodel): rANS0 "
          "/ lz4>rANS0 include the order-0 table and ARE realizable per chunk; "
          "rANS1\nis an idealized order-1 FLOOR, not achievable per independent "
          "chunk (cold-start / table).")
    order = ["video-packed", "video-ANS", "audio-pcm8", "audio-dfpwm"]
    for name in order:
        if buckets.get(name):
            _report_type(name, buckets[name])
    # Throughput on the most interesting bucket (packed video).
    if buckets.get("video-packed"):
        _report_throughput(buckets["video-packed"])


if __name__ == "__main__":
    main()

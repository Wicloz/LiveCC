"""
Shared dev/test helpers around the developer-provided media/ folder and the CC
codec's reference *decoder*.

Not part of the server runtime (the CC client does the decoding live) — this is
support code for the benchmarks and the sample tests, which need to find sample
clips, pull real frames through the front-end, and turn encoded blit frames back
into the pixels a monitor would show.

Lives at the server package root so both `benchmarks/` and `tools/` can import it
once the server dir is on sys.path.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parent.parent       # repo root (server/..)
MEDIA_DIR = _ROOT_DIR / "media"

# Realistic CC display sizes (character grid W x H) for the benchmarks/renderer.
# Built-in screens are a fixed size; monitors come from the CC:Tweaked formula at
# text scale 0.5 — termW = round((blocksW*64-20)/3), termH = round((blocksH*64-20)/4.5)
# — for block layouts close to 16:9 (monitor blocks are square in-world, so the
# physical aspect is blocksW:blocksH, and ~16:9 is what players build for video).
#
# The monitor block cap is uncapped in our config up to 16x9 blocks, so the larger
# tiers below are the closest-to-16:9 layout at each block height 6..9 (width =
# round(h*16/9)), culminating in an exact 16:9 at 16x9 — the new max.
#
#   device        blocks  aspect          cells (WxH)
#   pocket        builtin                 26x20
#   terminal      builtin                 51x19
#   mon 4x2       2.00                    79x24
#   mon 5x3       1.67                    100x38
#   mon 7x4       1.75  (closest to 16:9) 143x52
#   mon 8x5       1.60                    164x67
#   mon 11x6      1.83                    228x81
#   mon 12x7      1.71                    249x95
#   mon 14x8      1.75                    292x109
#   mon 16x9      1.78  (exact 16:9, max) 335x124
GRIDS = [
    ("pocket", 26, 20),
    ("terminal", 51, 19),
    ("mon4x2", 79, 24),
    ("mon5x3", 100, 38),
    ("mon7x4", 143, 52),
    ("mon8x5", 164, 67),
    ("mon11x6", 228, 81),
    ("mon12x7", 249, 95),
    ("mon14x8", 292, 109),
    ("mon16x9", 335, 124),
]


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


@functools.lru_cache(maxsize=None)
def media_streams(path: Path) -> frozenset[str]:
    """The stream types present in a media file (e.g. {"video", "audio"}), probed
    with ffprobe.  A `video` stream flagged `attached_pic` (cover art / thumbnail,
    as music files carry) does NOT count as video — it's a still image, not a clip.
    Empty if the file has no usable streams.  If ffprobe isn't installed we can't
    tell, so we assume both — nothing gets filtered out."""
    if shutil.which("ffprobe") is None:
        return frozenset({"video", "audio"})
    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "stream=codec_type:stream_disposition=attached_pic",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    streams = set()
    for line in out.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 2:
            continue
        codec_type, attached_pic = parts[0].strip(), parts[1].strip()
        if codec_type == "video" and attached_pic == "1":
            continue                       # cover art, not a real video stream
        if codec_type in ("video", "audio"):
            streams.add(codec_type)
    return frozenset(streams)


def find_media(stream: str | None = None) -> list[Path]:
    """Developer-provided samples in media/ (sorted).

    With no argument: *every* file except internal bookkeeping — dotfiles like
    .gitignore and README.md — and any subdirectory (drops out naturally, since
    only files pass the is_file() filter).  Deliberately no extension allowlist: whatever a
    developer drops here is a clip they expect to work, so a file a pipeline can't
    handle is a real bug to surface, not something to silently skip.

    With stream="video" (or "audio"): only files that actually contain such a
    stream, decided by probing the file — not by guessing from its extension.  This
    routes each sample to the pipeline that fits it (e.g. an audio-only clip is
    exercised by the audio path and isn't spuriously failed by the video path).
    """
    if not MEDIA_DIR.is_dir():
        return []
    files = [p for p in sorted(MEDIA_DIR.iterdir())
             if p.is_file()
             and not p.name.startswith(".")          # .gitignore and other dotfiles
             and p.name.lower() != "readme.md"]
    if stream is None:
        return files
    return [p for p in files if stream in media_streams(p)]


def sample_frames(path, w: int, h: int, fps: int = 24, limit: int = 8) -> list:
    """Decode up to `limit` real frames from `path` through the *actual* transcode
    front-end (the same ffmpeg scale/letterbox + frame splitter the server uses),
    returning (H*3, W*2, 3) uint8 arrays ready for encode_frame.

    Reads incrementally and stops at `limit`, so it stays cheap on long clips.
    """
    from transcoder import _FrameSplitter, _video_ffmpeg_cmd   # lazy: pulls numpy

    px_w, px_h = w * 2, h * 3
    cmd = _video_ffmpeg_cmd(px_w, px_h, fps, source=str(path))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    splitter = _FrameSplitter(px_w, px_h)
    frames: list = []
    try:
        while len(frames) < limit:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            frames.extend(splitter.push(chunk))
    finally:
        proc.kill()
        proc.wait()
    return frames[:limit]


def render_cells(glyph: np.ndarray, fg: np.ndarray, bg: np.ndarray,
                 palette: np.ndarray) -> np.ndarray:
    """(glyph, fg, bg) grids + a (16,3) palette -> (H*3, W*2, 3) uint8 image,
    exactly what the CC client paints (each cell = its two colours in the glyph
    pattern).  Takes cc_encoder.encode_frame's output, or a ccmf.DecodedFrame's
    fields, so both the encoder and the wire format can be eyeballed."""
    h, w = glyph.shape
    mask = glyph.astype(np.intp) - 0x80
    fg_idx, bg_idx = fg.astype(np.intp), bg.astype(np.intp)

    idx = np.empty((h, w, 6), np.intp)
    for s in range(5):                                  # s0..s4 from the mask bits
        idx[..., s] = np.where((mask >> s) & 1, fg_idx, bg_idx)
    idx[..., 5] = bg_idx                                # bottom-right is always bg
    rgb = palette[idx].astype(np.uint8)                 # (H,W,6,3), frame palette
    return (rgb.reshape(h, w, 3, 2, 3)
            .transpose(0, 2, 1, 3, 4)
            .reshape(h * 3, w * 2, 3))

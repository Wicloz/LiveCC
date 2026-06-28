"""
Shared dev/test helpers around the developer-provided media/ folder and the CC
codec's reference *decoder*.

Not part of the server runtime (the CC client does the decoding live) — this is
support code for the benchmarks, the sample tests, and the preview renderer, which
all need to find sample clips, pull real frames through the front-end, and turn
encoded blit frames back into the pixels a monitor would show.

Lives at the server package root so both `benchmarks/` and `tools/` can import it
once the server dir is on sys.path.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parent.parent       # repo root (server/..)
MEDIA_DIR = _ROOT_DIR / "media"
PREVIEW_DIR = MEDIA_DIR / "cc_preview"                    # renderer output (git-ignored)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".gif", ".m4v", ".flv", ".ts"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac"}

# CC display sizes (character grid W x H): pocket computer up to the largest 8x6
# monitor at text scale 0.5.
GRIDS = [
    ("pocket   26x20", 26, 20),
    ("terminal 51x19", 51, 19),
    ("monitor  82x41", 82, 41),
    ("max     164x81", 164, 81),
]


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def find_media(kind: str = "video") -> list[Path]:
    """Sample files present in media/ right now (sorted).  kind: video|audio|any.
    The render-output subfolder is skipped."""
    if not MEDIA_DIR.is_dir():
        return []
    exts = {"video": VIDEO_EXTS, "audio": AUDIO_EXTS,
            "any": VIDEO_EXTS | AUDIO_EXTS}[kind]
    return [p for p in sorted(MEDIA_DIR.iterdir())
            if p.is_file() and p.suffix.lower() in exts]


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


# Blit wire bytes -> palette indices -> RGB.  The inverse of encode_frame; used by
# the fidelity benchmark and the preview renderer to see what a CC monitor shows.
_HEXMAP = np.zeros(256, np.intp)
for _i, _c in enumerate(b"0123456789abcdef"):
    _HEXMAP[_c] = _i


def decode_frame(buf: bytes) -> np.ndarray:
    """Reverse encode_frame: blit wire bytes -> (H*3, W*2, 3) uint8 image, exactly
    what the CC client paints (each cell = its two colours in the glyph pattern)."""
    from cc_encoder import _CC_RGB

    w = buf[0] << 8 | buf[1]
    h = buf[2] << 8 | buf[3]
    body = np.frombuffer(buf, np.uint8, offset=4).reshape(h, 3, w)
    glyph, fg, bg = body[:, 0, :], body[:, 1, :], body[:, 2, :]
    mask = glyph.astype(np.intp) - 0x80
    fg_idx, bg_idx = _HEXMAP[fg], _HEXMAP[bg]

    idx = np.empty((h, w, 6), np.intp)
    for s in range(5):                                  # s0..s4 from the mask bits
        idx[..., s] = np.where((mask >> s) & 1, fg_idx, bg_idx)
    idx[..., 5] = bg_idx                                # bottom-right is always bg
    rgb = _CC_RGB[idx].astype(np.uint8)                 # (H,W,6,3)
    return (rgb.reshape(h, w, 3, 2, 3)
            .transpose(0, 2, 1, 3, 4)
            .reshape(h * 3, w * 2, 3))

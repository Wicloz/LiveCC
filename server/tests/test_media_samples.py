"""
Sample-driven pipeline tests.

These exercise the *real* transcode pipeline (ffmpeg decode/scale -> frame split
-> encode_frame, and ffmpeg -> PCM audio) over whatever developer-provided clips
sit in the repo-root media/ folder.  They are intentionally data-driven: with an
empty media/ folder (e.g. CI) every case skips, and they light up automatically
once a developer drops clips in.  ffmpeg is required.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

import transcoder

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_HAS_FFPROBE = shutil.which("ffprobe") is not None

_MEDIA_DIR = Path(__file__).resolve().parents[2] / "media"
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".gif", ".m4v", ".flv", ".ts"}
_BLIT_RANGE = range(0x80, 0xA0)
_HEX = set(b"0123456789abcdef")


def _video_samples() -> list[Path]:
    if not _MEDIA_DIR.is_dir():
        return []
    return [p for p in sorted(_MEDIA_DIR.iterdir())
            if p.is_file() and p.suffix.lower() in _VIDEO_EXTS]


def _has_audio(path: Path) -> bool:
    if not _HAS_FFPROBE:
        return False
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    return bool(out.stdout.strip())


_SAMPLES = _video_samples()
_AUDIO_SAMPLES = [p for p in _SAMPLES if _has_audio(p)] if _HAS_FFMPEG else []

_skip_no_samples = pytest.mark.skipif(
    not (_HAS_FFMPEG and _SAMPLES),
    reason="no media/ samples or ffmpeg not installed")


async def _collect_video(path: Path, w: int, h: int, limit: int) -> list:
    out = []
    agen = transcoder.iter_video("ignored", term_w=w, term_h=h, fps=10,
                                 source_path=str(path))
    try:
        async for pts, frame in agen:
            out.append((pts, frame))
            if len(out) >= limit:
                break
    finally:
        await agen.aclose()
    return out


@_skip_no_samples
@pytest.mark.parametrize("sample", _SAMPLES or [None],
                         ids=[p.name for p in _SAMPLES] or ["none"])
def test_sample_video_encodes_valid_frames(sample):
    # The full front-end must turn a real clip into well-formed blit frames.
    w, h = 20, 8
    items = asyncio.run(_collect_video(sample, w, h, limit=3))
    assert items, f"no frames produced from {sample.name}"

    pts = [p for p, _ in items]
    assert pts == sorted(pts)                     # PTS monotonically increases
    assert all(p >= 0 for p in pts)

    for _, frame in items:
        assert frame[0:4] == bytes((0, w, 0, h))  # header matches requested grid
        assert len(frame) == 4 + h * 3 * w        # header + H rows x 3 strings x W
        body = frame[4:]
        for row in range(h):
            base = row * 3 * w
            text = body[base:base + w]
            fg = body[base + w:base + 2 * w]
            bg = body[base + 2 * w:base + 3 * w]
            assert all(b in _BLIT_RANGE for b in text)   # valid 2x3 glyphs
            assert all(b in _HEX for b in fg) and all(b in _HEX for b in bg)


@_skip_no_samples
@pytest.mark.parametrize("sample", _SAMPLES or [None],
                         ids=[p.name for p in _SAMPLES] or ["none"])
def test_sample_video_grid_sizes(sample):
    # A couple of representative grids both produce correctly-sized frames.
    for w, h in [(10, 6), (40, 20)]:
        items = asyncio.run(_collect_video(sample, w, h, limit=1))
        assert items, f"{sample.name} produced nothing at {w}x{h}"
        assert items[0][1][0:4] == bytes((0, w, 0, h))


@pytest.mark.skipif(
    not (_HAS_FFMPEG and _AUDIO_SAMPLES),
    reason="no media/ samples with audio (or ffmpeg/ffprobe missing)")
def test_sample_audio_produces_pcm():
    # Use the first sample known to have audio so we never block on a silent clip.
    sample = _AUDIO_SAMPLES[0]

    async def go():
        agen = transcoder.iter_audio("ignored", sample_rate=48000,
                                     source_path=str(sample))
        try:
            return await asyncio.wait_for(agen.__anext__(), timeout=20)
        finally:
            await agen.aclose()

    chunk = asyncio.run(go())
    assert isinstance(chunk, (bytes, bytearray)) and len(chunk) > 0

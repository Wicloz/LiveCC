"""
Sample-driven pipeline tests.

These exercise the *real* transcode pipeline (ffmpeg decode/scale -> frame split
-> GOP encode into CCMF chunks, and ffmpeg -> PCM audio) over whatever
developer-provided clips sit in the repo-root media/ folder.  They are
intentionally data-driven: with an empty media/ folder (e.g. CI) every case
skips, and they light up automatically once a developer drops clips in.  ffmpeg
is required.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

import ccmf
import transcoder

from cc_media import find_media, media_streams

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_HAS_FFPROBE = shutil.which("ffprobe") is not None

# Every developer-provided clip is expected to work; if one doesn't, the test for
# it FAILS (not skips) — that's a genuine bug, since the file was put here on
# purpose.  Each file is routed to the pipeline matching its actual streams (probed,
# not guessed from the extension), so a video clip is checked by the video tests, an
# audio clip by the audio test.  Only an empty media/ folder (or missing ffmpeg)
# skips wholesale.
_SAMPLES = find_media()
_VIDEO_SAMPLES = find_media("video")
_AUDIO_SAMPLES = find_media("audio") if _HAS_FFMPEG else []


def _ids(samples):
    return [p.name for p in samples] or ["none"]


_skip_no_video = pytest.mark.skipif(
    not (_HAS_FFMPEG and _VIDEO_SAMPLES),
    reason="no media/ video samples or ffmpeg not installed")


async def _collect_video(path: Path, w: int, h: int, limit: int) -> list:
    out = []
    agen = transcoder.iter_video("ignored", term_w=w, term_h=h, fps=10,
                                 source_path=str(path))
    try:
        async for pts, chunk in agen:
            out.append((pts, chunk))
            if len(out) >= limit:
                break
    finally:
        await agen.aclose()
    return out


def _parse_video_chunk(chunk: bytes):
    pts, ctype, payload, _end = ccmf.parse_chunk(chunk)
    assert ctype == ccmf.TYPE_VIDEO
    return pts, ccmf.parse_video_payload(payload)


@_skip_no_video
@pytest.mark.parametrize("sample", _VIDEO_SAMPLES or [None], ids=_ids(_VIDEO_SAMPLES))
def test_sample_video_encodes_valid_chunks(sample):
    # The full front-end must turn a real clip into well-formed CCMF GOP chunks.
    w, h = 20, 8
    items = asyncio.run(_collect_video(sample, w, h, limit=3))
    assert items, f"no chunks produced from {sample.name}"

    pts = [p for p, _ in items]
    assert pts == sorted(pts)                     # PTS monotonically increases
    assert all(p >= 0 for p in pts)

    for ipts, chunk in items:
        cpts, (dw, dh, frames) = _parse_video_chunk(chunk)
        assert cpts == ipts                       # yielded pts matches the header
        assert (dw, dh) == (w, h)
        assert frames[0].encoding == ccmf.ENC_RAW  # every chunk is a RAP
        for f in frames:
            assert ((f.glyph >= 0x80) & (f.glyph <= 0x9F)).all()  # valid 2x3 chars
            assert (f.fg <= 15).all() and (f.bg <= 15).all()      # palette indices
            assert f.palette.shape == (16, 3)
            assert 0 < f.duration <= ccmf.MAX_DURATION


@_skip_no_video
@pytest.mark.parametrize("sample", _VIDEO_SAMPLES or [None], ids=_ids(_VIDEO_SAMPLES))
def test_sample_video_grid_sizes(sample):
    # A couple of representative grids both produce correctly-sized chunks.
    for w, h in [(10, 6), (40, 20)]:
        items = asyncio.run(_collect_video(sample, w, h, limit=1))
        assert items, f"{sample.name} produced nothing at {w}x{h}"
        _pts, (dw, dh, _frames) = _parse_video_chunk(items[0][1])
        assert (dw, dh) == (w, h)


@pytest.mark.skipif(
    not (_HAS_FFMPEG and _AUDIO_SAMPLES),
    reason="no media/ samples with audio (or ffmpeg/ffprobe missing)")
@pytest.mark.parametrize("sample", _AUDIO_SAMPLES or [None], ids=_ids(_AUDIO_SAMPLES))
def test_sample_audio_produces_pcm(sample):
    # Every clip that carries audio (a music file, or a video's soundtrack) must
    # resample to PCM through the real audio pipeline.
    async def go():
        agen = transcoder.iter_audio("ignored", sample_rate=48000,
                                     source_path=str(sample))
        try:
            return await asyncio.wait_for(agen.__anext__(), timeout=20)
        finally:
            await agen.aclose()

    chunk = asyncio.run(go())
    assert isinstance(chunk, (bytes, bytearray)) and len(chunk) > 0


@pytest.mark.skipif(
    not (_HAS_FFPROBE and _SAMPLES),
    reason="no media/ samples or ffprobe not installed")
@pytest.mark.parametrize("sample", _SAMPLES or [None], ids=_ids(_SAMPLES))
def test_sample_is_usable_media(sample):
    # A file dropped in media/ that has neither a video nor an audio stream is junk
    # (corrupt, or not a media file) — surface it instead of silently ignoring it.
    streams = media_streams(sample)
    assert "video" in streams or "audio" in streams, \
        f"{sample.name}: no decodable audio/video stream"

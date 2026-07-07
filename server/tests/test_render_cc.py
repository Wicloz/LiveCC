"""
render_cc CLI — argument wiring and the PTS merge/write, without needing
ffmpeg/yt-dlp: the async producers (iter_video / iter_audio_roles) and the
source probes are stubbed, so this only exercises render_cc's own logic
(grid/channel parsing, output-path defaulting, and interleaving finished
video/audio chunks into one file in PTS order).
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

_RENDER_CC = Path(__file__).resolve().parent.parent / "tools" / "render_cc.py"
_spec = importlib.util.spec_from_file_location("render_cc", _RENDER_CC)
render_cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_cc)

import ccmf  # noqa: E402  (server dir already on sys.path via render_cc's import)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def test_resolve_grid_preset():
    assert render_cc._resolve_grid("pocket", None, None) == (26, 20)


def test_resolve_grid_wxh():
    assert render_cc._resolve_grid("40x15", None, None) == (40, 15)


def test_resolve_grid_width_height_override():
    assert render_cc._resolve_grid("pocket", 10, None) == (10, 20)
    assert render_cc._resolve_grid("pocket", 10, 5) == (10, 5)


def test_resolve_grid_bad_spec():
    with pytest.raises(SystemExit):
        render_cc._resolve_grid("bogus", None, None)


def test_parse_channels_presets():
    assert render_cc._parse_channels("mono") == ccmf.CAP_CHANNEL_MONO
    assert render_cc._parse_channels("STEREO") == (
        ccmf.CAP_CHANNEL_FRONT_LEFT | ccmf.CAP_CHANNEL_FRONT_RIGHT)


def test_parse_channels_custom_roles():
    assert render_cc._parse_channels("fl,fr,lfe") == (
        ccmf.CAP_CHANNEL_FRONT_LEFT | ccmf.CAP_CHANNEL_FRONT_RIGHT
        | ccmf.CAP_CHANNEL_LFE)


def test_parse_channels_bad_role():
    with pytest.raises(SystemExit):
        render_cc._parse_channels("fl,nonsense")


def test_sanitize_filename_strips_unsafe_chars():
    assert render_cc._sanitize_filename('a/b:c*d"e') == "a_b_c_d_e"


def test_default_out_local_file(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    assert render_cc._default_out(str(clip), is_url=False) == clip.with_suffix(".ccmf")


# --------------------------------------------------------------------------- #
# _render(): CLI wiring + PTS-ordered merge, producers stubbed
# --------------------------------------------------------------------------- #

def _make_args(tmp_path, clip, extra=None):
    argv = [str(clip), "-o", str(tmp_path / "out.ccmf"), "--grid", "10x4",
            "--fps", "10"]
    if extra:
        argv += extra
    return render_cc.build_argparser().parse_args(argv)


async def _fake_iter_video_ok(*a, **kw):
    yield 0, ccmf.chunk(0, ccmf.TYPE_VIDEO, b"v0")
    yield 4000, ccmf.chunk(4000, ccmf.TYPE_VIDEO, b"v1")


async def _fake_iter_video_empty(*a, **kw):
    return
    yield  # pragma: no cover  (makes this an async generator)


async def _fake_iter_audio_roles_ok(*a, **kw):
    yield 0, {0: b"\x80\x81\x82"}
    yield 2000, {0: b"\x83\x84\x85"}


async def _fake_iter_audio_roles_empty(*a, **kw):
    return
    yield  # pragma: no cover


def _decode(out_path):
    """Walk the written file with the reference decoder -> [(type, pts), ...]."""
    data = out_path.read_bytes()
    return [(ctype, pts) for pts, ctype, _payload in ccmf.iter_chunks(data)]


def test_render_merges_video_and_audio_in_pts_order(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(render_cc, "iter_video", _fake_iter_video_ok)
    monkeypatch.setattr(render_cc, "iter_audio_roles", _fake_iter_audio_roles_ok)

    args = _make_args(tmp_path, clip)
    rc = asyncio.run(render_cc._render(args))
    assert rc == 0

    out_path = tmp_path / "out.ccmf"
    entries = _decode(out_path)
    assert len(entries) == 4
    assert sorted(entries, key=lambda e: e[1]) == entries   # non-decreasing PTS
    assert {t for t, _ in entries} == {ccmf.TYPE_VIDEO, ccmf.TYPE_AUDIO}


def test_render_no_video_produces_error(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(render_cc, "iter_video", _fake_iter_video_empty)
    monkeypatch.setattr(render_cc, "iter_audio_roles", _fake_iter_audio_roles_ok)

    args = _make_args(tmp_path, clip)
    rc = asyncio.run(render_cc._render(args))
    assert rc == 1
    assert not (tmp_path / "out.ccmf").exists()


def test_render_audio_only_warns_but_succeeds(tmp_path, monkeypatch, capsys):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(render_cc, "iter_video", _fake_iter_video_ok)
    monkeypatch.setattr(render_cc, "iter_audio_roles", _fake_iter_audio_roles_empty)

    args = _make_args(tmp_path, clip)
    rc = asyncio.run(render_cc._render(args))
    assert rc == 0
    assert (tmp_path / "out.ccmf").exists()
    assert "warning" in capsys.readouterr().out.lower()


def test_render_rejects_both_streams_disabled(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    args = _make_args(tmp_path, clip, extra=["--no-video", "--no-audio"])
    rc = asyncio.run(render_cc._render(args))
    assert rc == 1


def test_render_rejects_bad_time_range(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    args = _make_args(tmp_path, clip, extra=["--start", "5", "--end", "3"])
    rc = asyncio.run(render_cc._render(args))
    assert rc == 1


def test_main_requires_ffmpeg(monkeypatch, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(render_cc, "have_ffmpeg", lambda: False)
    rc = render_cc.main([str(clip)])
    assert rc == 1

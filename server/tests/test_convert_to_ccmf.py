"""
convert_to_ccmf CLI — argument wiring and the PTS merge/write, without needing
ffmpeg/yt-dlp: the async producers (iter_video / iter_audio_roles) and the
source probes are stubbed, so this only exercises convert_to_ccmf's own logic
(grid/channel parsing, output-path defaulting, and interleaving finished
video/audio chunks into one file in PTS order).
"""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
from pathlib import Path

import pytest

_CONVERT_TO_CCMF = Path(__file__).resolve().parent.parent / "tools" / "convert_to_ccmf.py"
_spec = importlib.util.spec_from_file_location("convert_to_ccmf", _CONVERT_TO_CCMF)
convert_to_ccmf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert_to_ccmf)

import ccmf  # noqa: E402  (server dir already on sys.path via convert_to_ccmf's import)
from cc_media import find_media  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_probe_source_stats(monkeypatch):
    """_render() probes the source's dimensions/duration before producing
    (for the bounding-box grid computation and the progress bars' totals);
    stub it so every test in this file stays hermetic (no real ffprobe/
    yt-dlp call) regardless of whether it happens to reach that code path."""
    async def _fake(*_a, **_kw):
        return (64, 48), 10.0
    monkeypatch.setattr(convert_to_ccmf, "_probe_source_stats", _fake)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def test_resolve_grid_bound_preset():
    args = convert_to_ccmf.build_argparser().parse_args(["clip.mp4", "--grid", "pocket"])
    assert convert_to_ccmf._resolve_grid_bound(args) == (26, 20)


def test_resolve_grid_bound_wxh():
    args = convert_to_ccmf.build_argparser().parse_args(["clip.mp4", "--grid", "40x15"])
    assert convert_to_ccmf._resolve_grid_bound(args) == (40, 15)


def test_resolve_grid_bound_width_only_has_no_height_bound():
    # Giving --width alone means "derive height from aspect ratio", NOT
    # "keep --grid's default height" -- see _resolve_grid_bound's doc.
    args = convert_to_ccmf.build_argparser().parse_args(["clip.mp4", "--width", "100"])
    assert convert_to_ccmf._resolve_grid_bound(args) == (100, None)


def test_resolve_grid_bound_height_only():
    args = convert_to_ccmf.build_argparser().parse_args(["clip.mp4", "--height", "40"])
    assert convert_to_ccmf._resolve_grid_bound(args) == (None, 40)


def test_resolve_grid_bound_width_and_height_ignores_grid():
    args = convert_to_ccmf.build_argparser().parse_args(
        ["clip.mp4", "--grid", "pocket", "--width", "10", "--height", "5"])
    assert convert_to_ccmf._resolve_grid_bound(args) == (10, 5)


def test_resolve_grid_bound_bad_spec():
    args = convert_to_ccmf.build_argparser().parse_args(["clip.mp4", "--grid", "bogus"])
    with pytest.raises(SystemExit):
        convert_to_ccmf._resolve_grid_bound(args)


# --------------------------------------------------------------------------- #
# _compute_output_grid: aspect-preserving, no letterboxing
# --------------------------------------------------------------------------- #

def test_compute_output_grid_both_bounds_fits_inside_landscape():
    # A 16:9 source is narrower (per cell aspect) than a 51x19 bound, so
    # height is the constraining dimension and width comes out under 51.
    assert convert_to_ccmf._compute_output_grid(1920, 1080, 51, 19) == (50, 19)


def test_compute_output_grid_both_bounds_fits_inside_portrait():
    assert convert_to_ccmf._compute_output_grid(1080, 1920, 51, 19) == (16, 19)


def test_compute_output_grid_width_only_derives_height():
    assert convert_to_ccmf._compute_output_grid(1920, 1080, 100, None) == (100, 38)


def test_compute_output_grid_height_only_derives_width():
    assert convert_to_ccmf._compute_output_grid(1920, 1080, None, 40) == (107, 40)


def test_compute_output_grid_square_source_and_bound():
    # Cells are 2x3 px (not square), so a square source doesn't map to a
    # square cell grid: width is the constraining bound here (scale=0.4 from
    # width; 0.6 from height), landing exactly on it while height comes out
    # under its own bound.
    assert convert_to_ccmf._compute_output_grid(100, 100, 20, 20) == (20, 13)


def test_compute_output_grid_rejects_no_bounds():
    with pytest.raises(ValueError):
        convert_to_ccmf._compute_output_grid(1920, 1080, None, None)


def test_compute_output_grid_rejects_bad_source_dims():
    with pytest.raises(ValueError):
        convert_to_ccmf._compute_output_grid(0, 1080, 51, 19)


# --------------------------------------------------------------------------- #
# _probe_source_stats_blocking: real ffprobe, over whatever media/ samples are
# present (self-skips otherwise, matching test_media_samples.py's convention).
# --------------------------------------------------------------------------- #

_HAS_FFPROBE = shutil.which("ffprobe") is not None
_VIDEO_SAMPLES = find_media("video") if _HAS_FFPROBE else []


@pytest.mark.skipif(not _VIDEO_SAMPLES, reason="no media/ video samples or ffprobe not installed")
def test_probe_source_stats_blocking_reads_real_dimensions_and_duration():
    sample = _VIDEO_SAMPLES[0]
    dims, duration = convert_to_ccmf._probe_source_stats_blocking(str(sample), is_url=False)

    assert dims is not None
    w, h = dims
    assert w > 0 and h > 0

    assert duration is not None
    assert duration > 0


def test_probe_source_stats_blocking_missing_file_returns_none_none():
    dims, duration = convert_to_ccmf._probe_source_stats_blocking(
        "definitely_not_a_real_file.mp4", is_url=False)
    assert dims is None
    assert duration is None


def test_parse_channels_presets():
    assert convert_to_ccmf._parse_channels("mono") == ccmf.CAP_CHANNEL_MONO
    assert convert_to_ccmf._parse_channels("STEREO") == (
        ccmf.CAP_CHANNEL_FRONT_LEFT | ccmf.CAP_CHANNEL_FRONT_RIGHT)


def test_parse_channels_custom_roles():
    assert convert_to_ccmf._parse_channels("fl,fr,lfe") == (
        ccmf.CAP_CHANNEL_FRONT_LEFT | ccmf.CAP_CHANNEL_FRONT_RIGHT
        | ccmf.CAP_CHANNEL_LFE)


def test_parse_channels_bad_role():
    with pytest.raises(SystemExit):
        convert_to_ccmf._parse_channels("fl,nonsense")


def test_sanitize_filename_strips_unsafe_chars():
    assert convert_to_ccmf._sanitize_filename('a/b:c*d"e') == "a_b_c_d_e"


def test_default_out_local_file(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    assert convert_to_ccmf._default_out(str(clip), is_url=False) == clip.with_suffix(".ccmf")


# --------------------------------------------------------------------------- #
# _render(): CLI wiring + PTS-ordered merge, producers stubbed
# --------------------------------------------------------------------------- #

def _make_args(tmp_path, clip, extra=None):
    argv = [str(clip), "-o", str(tmp_path / "out.ccmf"), "--grid", "10x4",
            "--fps", "10"]
    if extra:
        argv += extra
    return convert_to_ccmf.build_argparser().parse_args(argv)


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


def test_render_bounds_a_fast_producer_via_backpressure(tmp_path, monkeypatch):
    """A stream that finishes far faster than the other (the common case:
    audio decode is cheap, video encode is expensive) must not buffer its
    entire output in memory while waiting -- it should block once it's built
    up _MAX_QUEUED_CHUNKS items ahead, resuming only as _merge_write drains
    them. Proven here by holding video to a single item (an async Event it
    won't get past) while audio tries to race through 200, and checking how
    many audio actually managed to produce during that stall.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")

    video_gate = asyncio.Event()
    audio_yielded: list[int] = []

    async def _fake_iter_video_one_then_wait(*_a, **_kw):
        yield 0, ccmf.chunk(0, ccmf.TYPE_VIDEO, b"v0")
        await video_gate.wait()
        yield 999_999, ccmf.chunk(999_999, ccmf.TYPE_VIDEO, b"v1")

    async def _fake_iter_audio_roles_fast_many(*_a, **_kw):
        for i in range(200):
            audio_yielded.append(i)
            yield i * 1000, {0: bytes([i % 256])}

    monkeypatch.setattr(convert_to_ccmf, "iter_video", _fake_iter_video_one_then_wait)
    monkeypatch.setattr(convert_to_ccmf, "iter_audio_roles", _fake_iter_audio_roles_fast_many)

    args = _make_args(tmp_path, clip)

    async def _drive() -> int:
        task = asyncio.ensure_future(convert_to_ccmf._render(args))
        await asyncio.sleep(0.3)  # let the event loop settle into its blocked steady-state
        assert not task.done(), "render finished without ever needing video_gate"
        stalled_count = len(audio_yielded)
        # The real assertion: audio could NOT race through anywhere near all
        # 200 items while video was stuck on its first -- only up to roughly
        # one queue's worth (a little slack for the item already popped into
        # _merge_write's `heads` and any in-flight coroutine step).
        assert stalled_count < 200
        assert stalled_count <= convert_to_ccmf._MAX_QUEUED_CHUNKS + 2, (
            f"audio produced {stalled_count} items while video was stalled -- "
            f"backpressure isn't bounding it to ~{convert_to_ccmf._MAX_QUEUED_CHUNKS}")

        video_gate.set()  # let video finish; the render can now complete naturally
        return await task

    rc = asyncio.run(_drive())
    assert rc == 0
    entries = _decode(tmp_path / "out.ccmf")
    assert len(entries) == 202  # 2 video + 200 audio
    assert sorted(entries, key=lambda e: e[1]) == entries


def test_render_merges_video_and_audio_in_pts_order(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(convert_to_ccmf, "iter_video", _fake_iter_video_ok)
    monkeypatch.setattr(convert_to_ccmf, "iter_audio_roles", _fake_iter_audio_roles_ok)

    args = _make_args(tmp_path, clip)
    rc = asyncio.run(convert_to_ccmf._render(args))
    assert rc == 0

    out_path = tmp_path / "out.ccmf"
    entries = _decode(out_path)
    assert len(entries) == 4
    assert sorted(entries, key=lambda e: e[1]) == entries   # non-decreasing PTS
    assert {t for t, _ in entries} == {ccmf.TYPE_VIDEO, ccmf.TYPE_AUDIO}


def test_render_no_video_produces_error(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(convert_to_ccmf, "iter_video", _fake_iter_video_empty)
    monkeypatch.setattr(convert_to_ccmf, "iter_audio_roles", _fake_iter_audio_roles_ok)

    args = _make_args(tmp_path, clip)
    rc = asyncio.run(convert_to_ccmf._render(args))
    assert rc == 1
    assert not (tmp_path / "out.ccmf").exists()


def test_render_audio_only_warns_but_succeeds(tmp_path, monkeypatch, capsys):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(convert_to_ccmf, "iter_video", _fake_iter_video_ok)
    monkeypatch.setattr(convert_to_ccmf, "iter_audio_roles", _fake_iter_audio_roles_empty)

    args = _make_args(tmp_path, clip)
    rc = asyncio.run(convert_to_ccmf._render(args))
    assert rc == 0
    assert (tmp_path / "out.ccmf").exists()
    assert "warning" in capsys.readouterr().out.lower()


def test_render_rejects_both_streams_disabled(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    args = _make_args(tmp_path, clip, extra=["--no-video", "--no-audio"])
    rc = asyncio.run(convert_to_ccmf._render(args))
    assert rc == 1


def test_render_rejects_bad_time_range(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    args = _make_args(tmp_path, clip, extra=["--start", "5", "--end", "3"])
    rc = asyncio.run(convert_to_ccmf._render(args))
    assert rc == 1


def test_render_rejects_grid_over_cell_budget(tmp_path, capsys):
    # 64x48 source (the stubbed probe) into a 9999x9999 bounding box scales to
    # 9999x4999 -- ~50M cells, way past CCMF's 65535-cell delta-span limit.
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    args = _make_args(tmp_path, clip, extra=["--width", "9999", "--height", "9999"])
    rc = asyncio.run(convert_to_ccmf._render(args))
    assert rc == 1
    assert not (tmp_path / "out.ccmf").exists()
    assert "cell" in capsys.readouterr().out.lower()


def test_main_requires_ffmpeg(monkeypatch, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(convert_to_ccmf, "have_ffmpeg", lambda: False)
    rc = convert_to_ccmf.main([str(clip)])
    assert rc == 1

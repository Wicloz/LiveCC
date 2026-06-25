import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

import transcoder
from transcoder import _FrameSplitter, _kill_wait, _ytdlp_cmd, parse_timestamp


# --------------------------------------------------------------------------- #
# _FrameSplitter
# --------------------------------------------------------------------------- #

def _px_dims(term_w, term_h):
    return term_w * 2, term_h * 3   # 2x3 sub-pixels per cell


def test_splitter_emits_one_complete_frame():
    px_w, px_h = _px_dims(4, 2)
    s = _FrameSplitter(px_w, px_h)
    frames = list(s.push(bytes([17]) * (px_w * px_h * 3)))   # solid
    assert len(frames) == 1
    assert s.count == 1
    # push() now yields the raw (H, W, 3) array; encoding is done separately.
    assert frames[0].shape == (px_h, px_w, 3)
    # encode_frame still produces the right header: W=4, H=2.
    assert transcoder.encode_frame(frames[0])[0:4] == bytes((0, 4, 0, 2))


def test_splitter_buffers_partial_frames():
    px_w, px_h = _px_dims(2, 1)        # frame_bytes = 4*3*3 = 36
    s = _FrameSplitter(px_w, px_h)
    fb = px_w * px_h * 3
    assert list(s.push(bytes(fb - 1))) == []
    assert s.count == 0
    frames = list(s.push(bytes(1)))
    assert len(frames) == 1
    assert s.count == 1


def test_splitter_handles_multiple_frames_across_byte_chunks():
    px_w, px_h = _px_dims(2, 1)
    s = _FrameSplitter(px_w, px_h)
    two = bytes(px_w * px_h * 3 * 2)
    out = []
    for b in two:
        out.extend(s.push(bytes([b])))
    assert len(out) == 2
    assert s.count == 2


# --------------------------------------------------------------------------- #
# Command builders
# --------------------------------------------------------------------------- #

def test_ytdlp_cmd_streams_to_stdout():
    cmd = _ytdlp_cmd("https://example.com/v", transcoder._VIDEO_FMT)
    assert cmd[0] == "yt-dlp"
    assert "-o" in cmd and cmd[cmd.index("-o") + 1] == "-"
    assert cmd[-1] == "https://example.com/v"
    assert "--no-playlist" in cmd
    assert "--download-sections" not in cmd       # no offset by default


def test_ytdlp_cmd_section_start_only():
    cmd = _ytdlp_cmd("https://example.com/v", transcoder._VIDEO_FMT, start=125)
    assert cmd[cmd.index("--download-sections") + 1] == "*125-inf"
    assert cmd[-1] == "https://example.com/v"


def test_ytdlp_cmd_section_start_and_end():
    cmd = _ytdlp_cmd("https://example.com/v", transcoder._VIDEO_FMT, start=30, end=90)
    assert cmd[cmd.index("--download-sections") + 1] == "*30-90"


def test_audio_fmt_prefers_best_source():
    # The DFPWM encoder is 1-bit; feeding it the *worst* YouTube stream stacks a
    # second lossy pass on an already-crushed one (audible clipping/hiss).  Pin
    # bestaudio so the encoder always gets a clean source.
    fmt = transcoder._AUDIO_FMT
    assert "bestaudio" in fmt
    assert "worstaudio" not in fmt
    # webm/opus still preferred first for streaming-pipe compatibility.
    assert fmt.startswith("bestaudio[ext=webm]")


@pytest.mark.parametrize("value,expected", [
    ("90", 90),
    ("90s", 90),
    ("1m30s", 90),
    ("3h2m", 10920),
    ("1h2m3s", 3723),
    ("25234s", 25234),
    ("", 0),
    (None, 0),
    ("garbage", 0),
])
def test_parse_timestamp(value, expected):
    assert parse_timestamp(value) == expected


def test_video_ffmpeg_cmd_reads_pipe_outputs_rgb24():
    cmd = transcoder._video_ffmpeg_cmd(10, 8, 12)
    assert "pipe:0" in cmd and "pipe:1" in cmd
    assert "rgb24" in cmd
    assert "fps=12" in " ".join(cmd)
    assert "-stream_loop" not in cmd


def test_video_ffmpeg_cmd_loops_a_file():
    cmd = transcoder._video_ffmpeg_cmd(10, 8, 12, source="/tmp/s.mkv")
    assert cmd[cmd.index("-stream_loop") + 1] == "-1"
    assert "/tmp/s.mkv" in cmd
    assert "0:v:0" in cmd
    assert "pipe:0" not in cmd


def test_video_ffmpeg_cmd_aligns_letterbox_to_cell_grid():
    # Regression: content edges must snap to the 2x3 cell grid, otherwise
    # boundary cells mix content with black bar and the corners go black.
    vf = " ".join(transcoder._video_ffmpeg_cmd(20, 8, 10))
    assert "trunc(iw/2)*2:trunc(ih/3)*3" in vf      # content snapped to grid
    assert "trunc((ow-iw)/4)*2" in vf               # even x pad offset
    assert "trunc((oh-ih)/6)*3" in vf               # multiple-of-3 y pad offset


def test_download_cmd_merges_to_mkv():
    # Regression: yt-dlp rejects "matroska"; the merge format must be "mkv".
    cmd = transcoder._download_cmd("https://x", "/tmp/d", 0, None, True)
    assert cmd[cmd.index("--merge-output-format") + 1] == "mkv"


def test_requirements_bundle_ejs_solver():
    # Regression: yt-dlp needs the bundled JS challenge solver (yt-dlp-ejs) to
    # solve YouTube's n/sig; the [default] extra pulls it in.
    req = (Path(__file__).parent.parent / "requirements.txt").read_text()
    assert "yt-dlp[default]" in req


def test_audio_ffmpeg_cmd_is_dfpwm_mono():
    cmd = transcoder._audio_ffmpeg_cmd(48000)
    assert "dfpwm" in cmd             # CC-native DFPWM1a
    assert cmd[cmd.index("-c:a") + 1] == "dfpwm"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-ar") + 1] == "48000"


def test_audio_ffmpeg_cmd_loops_a_file():
    cmd = transcoder._audio_ffmpeg_cmd(48000, source="/tmp/s.mkv")
    assert cmd[cmd.index("-stream_loop") + 1] == "-1"
    assert "0:a:0?" in cmd            # optional audio map (source may be video-only)


def test_audio_ffmpeg_cmd_has_no_server_side_filtering():
    # Server-side audio filtering was reported to make audio worse, so the source
    # is fed straight into the DFPWM encoder — no -af chain on either path.
    for cmd in (transcoder._audio_ffmpeg_cmd(48000),
                transcoder._audio_ffmpeg_cmd(48000, source="/tmp/s.mkv")):
        assert "-af" not in cmd
        assert "-filter:a" not in cmd


# --------------------------------------------------------------------------- #
# _kill_wait
# --------------------------------------------------------------------------- #

def test_kill_wait_is_safe_on_finished_process_and_none():
    p = subprocess.Popen(
        ["python", "-c", "pass"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    p.wait()
    _kill_wait(p, None)   # must tolerate both a dead process and None


# --------------------------------------------------------------------------- #
# Integration: real ffmpeg pipeline (skipped if ffmpeg is unavailable)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_rawvideo_feeds_splitter():
    term_w, term_h, fps = 8, 4, 5
    px_w, px_h = _px_dims(term_w, term_h)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=size=64x48:rate={fps}:duration=1",
        "-vf", f"scale={px_w}:{px_h}",
        "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.returncode == 0
    assert len(proc.stdout) == px_w * px_h * 3 * fps

    s = _FrameSplitter(px_w, px_h)
    raw_frames = list(s.push(proc.stdout))
    assert len(raw_frames) == fps
    enc = transcoder.encode_frame(raw_frames[0])
    assert enc[0:4] == bytes((0, term_w, 0, term_h))
    # each encoded frame is header + term_h rows * 3 strings * term_w bytes
    assert len(enc) == 4 + term_h * 3 * term_w


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_dfpwm_encode_is_available_in_ffmpeg():
    # Guard that this ffmpeg can actually produce DFPWM (CC speaker audio) with the
    # same output args the audio command uses — a missing encoder would only
    # surface at runtime otherwise.  Push a sine through to dfpwm.
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=48000",
        "-ar", "48000", "-ac", "1", "-c:a", "dfpwm", "-f", "dfpwm", "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert len(proc.stdout) > 0


# --------------------------------------------------------------------------- #
# Integration: real live streams (regression guard)
# --------------------------------------------------------------------------- #

# NASA's 24/7 live channels.  These exercise the full live pipeline end to end.
NASA_LIVE_STREAMS = [
    "https://www.youtube.com/watch?v=FuuC4dpSQ1M",
    "https://www.youtube.com/watch?v=uwXgcTc8oY8",
]

# YouTube needs a JS runtime (deno/node) to resolve formats; without one the
# pipeline can't fetch anything, so skip rather than report a bogus failure.
_HAS_JS_RUNTIME = any(shutil.which(x) for x in ("deno", "node", "bun"))


@pytest.mark.youtube
@pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("yt-dlp") and _HAS_JS_RUNTIME),
    reason="live test needs ffmpeg, yt-dlp and a JS runtime (deno/node) on PATH",
)
@pytest.mark.parametrize("url", NASA_LIVE_STREAMS)
def test_live_stream_produces_frames(url):
    """A real live stream must produce frames through the video pipeline.

    Confirms the stream is actually live first; if it isn't (taken down, or no
    network right now) the test skips, so it only fails on a genuine regression.
    """
    state = {"frames": 0}

    async def go():
        if not await transcoder.probe_is_live(url):
            pytest.skip(f"not live right now (or unreachable): {url}")

        async def collect():
            agen = transcoder.iter_video(url, term_w=20, term_h=8, fps=5)
            try:
                async for _frame in agen:
                    state["frames"] += 1
                    if state["frames"] >= 5:
                        break
            finally:
                await agen.aclose()

        try:
            await asyncio.wait_for(collect(), timeout=60)
        except asyncio.TimeoutError:
            pass

    asyncio.run(go())
    assert state["frames"] > 0, f"live stream produced no frames: {url}"

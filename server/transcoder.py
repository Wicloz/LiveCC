"""
YouTube -> CC producers.

Two modes:
  * Streaming (default): yt-dlp (-o -) -> OS pipe -> ffmpeg (stdin) -> frames/PCM.
  * Looping (--loop):    download the [start,end] section once to a temp file,
                         then ffmpeg `-stream_loop -1` replays it.  Both video
                         and audio read the *same* file, so they loop at the
                         identical period and stay in sync across loop edges.

Pacing, buffering and A/V sync are handled downstream by session.StreamSession,
which timestamps each item by its output position on a shared media timeline.

yt-dlp owns the download (auth, headers, JS challenge solving via deno).
WebM/VP9 + Opus are preferred for the streaming pipe (parseable from a pipe).
"""

from __future__ import annotations

import asyncio
import collections
import glob
import logging
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Deque, Iterator, Optional

import numpy as np

from cc_palette import encode_frame

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

log = logging.getLogger("livecc")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    log.addHandler(_h)
    log.propagate = False

_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="pipe-reader")

# yt-dlp format selectors — webm first for streaming-pipe compatibility.
# Video picks the smallest stream (frames are downscaled to the cell grid anyway).
# Audio picks the *best* stream: it's transcoded to 1-bit DFPWM, so feeding the
# encoder a clean source avoids stacking a second lossy pass on an already-crushed
# one.  Audio streams are tiny next to video, so the bandwidth cost is negligible.
_VIDEO_FMT = "worstvideo[ext=webm]/worstvideo[vcodec^=vp9]/worstvideo/worst"
_AUDIO_FMT = "bestaudio[ext=webm]/bestaudio[acodec^=opus]/bestaudio/best"

# Watchdog timeouts (seconds): wait for the first byte, then max gap between reads.
_FIRST_OUTPUT_TIMEOUT = 45
_STALL_TIMEOUT = 30
_DOWNLOAD_TIMEOUT = 1800   # full section download for --loop

# Audio is DFPWM1a: 1 bit/sample, decoded natively on the client by
# cc.audio.dfpwm (fast, no Lua parse).  8 samples per byte.
SAMPLES_PER_BYTE = 8
AUDIO_READ_BYTES = 6000   # DFPWM -> 48000 samples -> 1.0 s per chunk at 48 kHz

# Encode-side filtering for the 1-bit DFPWM codec, kept static and predictable.
# Two artifacts to manage: a ~Nyquist/2 whine intrinsic to 1-bit modulation, and
# broadband hiss the predictor generates from high frequencies it can't track.
# The lowpass is the most effective hiss reducer — rolling off highs *before* the
# encoder means the predictor never has to chase transitions it would turn into
# noise.  We avoid dynamic gain (dynaudnorm/loudnorm): the codec has a fixed noise
# floor, so gain riding just pulls that hiss up in quiet passages.  In order:
#   highpass  drop sub-40 Hz rumble that wastes the codec's limited range
#   lowpass   roll off highs the predictor can't track -> much less hiss/whine
#   volume    a fixed -3 dB cut for headroom (the predictor overshoots transients)
# No hard limiter: the -3 dB cut keeps most peaks in range, and the occasional
# clip on the loudest transients is left as part of the codec's lo-fi charm.
# The decode-side one-pole postfilter in player.lua then cleans up the residual.
# LOWPASS_HZ is the main hiss/brightness knob: lower = less hiss but duller.
_LOWPASS_HZ = 10000
_AUDIO_FILTERS = (
    "highpass=f=40,"
    f"lowpass=f={_LOWPASS_HZ},"
    "volume=-3dB"
)


# --------------------------------------------------------------------------- #
# Timestamp parsing  ("90", "90s", "1m30s", "3h2m", "25234s")
# --------------------------------------------------------------------------- #

def parse_timestamp(value: str | None) -> int:
    """Parse a duration into whole seconds; default unit is seconds.

    Accepts plain seconds ("90"), an explicit "s" ("90s"), and the h/m/s form
    ("3h2m", "1m30s", "1h2m3s").  Returns 0 for empty/unrecognised input.
    """
    if not value:
        return 0
    value = value.strip().lower()
    if value.isdigit():
        return int(value)
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", value)
    if not m or not any(m.groups()):
        return 0
    h, mn, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mn * 60 + s


def _sections_arg(start: int, end: Optional[int]) -> list[str]:
    """yt-dlp --download-sections for a [start, end] window (seconds)."""
    if start <= 0 and end is None:
        return []
    s = max(0, start)
    e = "inf" if end is None else max(s, end)
    return ["--download-sections", f"*{s}-{e}"]


# --------------------------------------------------------------------------- #
# Subprocess helpers
# --------------------------------------------------------------------------- #

def _spawn_stderr_drain(proc: subprocess.Popen, name: str) -> Deque[str]:
    """Drain proc.stderr in a daemon thread (an unread PIPE can deadlock the child)."""
    tail: Deque[str] = collections.deque(maxlen=30)

    def _drain() -> None:
        if proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                log.info("[%s] %s", name, line)
                tail.append(line)

    threading.Thread(target=_drain, name=f"stderr-{name}", daemon=True).start()
    return tail


def _kill_wait(*procs: subprocess.Popen | None) -> None:
    for p in procs:
        if p is not None:
            try:
                p.kill()
            except OSError:
                pass
    for p in procs:
        if p is not None:
            try:
                p.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass


async def _read_with_timeout(proc: subprocess.Popen, size: int, timeout: float) -> bytes:
    """Read up to `size` bytes from proc.stdout, bounded by `timeout` seconds."""
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(_executor, proc.stdout.read, size), timeout
    )


# --------------------------------------------------------------------------- #
# Frame splitting (pure logic — unit-testable without subprocesses)
# --------------------------------------------------------------------------- #

class _FrameSplitter:
    """Accumulate raw rgb24 bytes and emit encoded 2x3 frames per complete frame."""

    def __init__(self, px_w: int, px_h: int) -> None:
        self.px_w = px_w
        self.px_h = px_h
        self.frame_bytes = px_w * px_h * 3
        self._buf = bytearray()
        self._n = 0

    def push(self, chunk: bytes) -> Iterator[bytes]:
        self._buf.extend(chunk)
        while len(self._buf) >= self.frame_bytes:
            raw = bytes(self._buf[: self.frame_bytes])
            del self._buf[: self.frame_bytes]
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.px_h, self.px_w, 3)
            yield encode_frame(arr)
            self._n += 1

    @property
    def count(self) -> int:
        return self._n


# --------------------------------------------------------------------------- #
# Command builders
# --------------------------------------------------------------------------- #

def _ytdlp_cmd(youtube_url: str, fmt: str, start: int = 0,
               end: Optional[int] = None) -> list[str]:
    cmd = ["yt-dlp", "-f", fmt, "--no-playlist", "--no-progress"]
    cmd += _sections_arg(start, end)   # server-side seek via range requests
    cmd += ["-o", "-", youtube_url]
    return cmd


def _video_ffmpeg_cmd(px_w: int, px_h: int, fps: int,
                      source: Optional[str] = None) -> list[str]:
    # Letterbox aligned to the 2x3 character grid: snap the scaled content to
    # even width / multiple-of-3 height and pad at even-x / multiple-of-3-y
    # offsets.  Otherwise the content edge lands mid-cell, so boundary cells mix
    # content with black bar and the content's corners quantize to black.
    scale = (
        f"scale={px_w}:{px_h}:force_original_aspect_ratio=decrease,"
        f"scale=trunc(iw/2)*2:trunc(ih/3)*3,"
        f"pad={px_w}:{px_h}:trunc((ow-iw)/4)*2:trunc((oh-ih)/6)*3:black,"
        f"fps={fps}"
    )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats"]
    if source:                                   # loop a local file
        cmd += ["-stream_loop", "-1", "-i", source, "-map", "0:v:0"]
    else:                                        # stream from yt-dlp pipe
        cmd += ["-i", "pipe:0"]
    cmd += ["-vf", scale, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    return cmd


def _audio_ffmpeg_cmd(sample_rate: int, source: Optional[str] = None) -> list[str]:
    # DFPWM1a mono — CC's native speaker format (cc.audio.dfpwm decodes it).
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats"]
    if source:
        cmd += ["-stream_loop", "-1", "-i", source, "-map", "0:a:0?"]
    else:
        cmd += ["-i", "pipe:0"]
    cmd += ["-vn", "-af", _AUDIO_FILTERS, "-ar", str(sample_rate), "-ac", "1",
            "-c:a", "dfpwm", "-f", "dfpwm", "pipe:1"]
    return cmd


# --------------------------------------------------------------------------- #
# is_live probe
# --------------------------------------------------------------------------- #

def _probe_is_live_blocking(url: str) -> bool:
    try:
        out = subprocess.run(
            ["yt-dlp", "--no-warnings", "--quiet", "--no-playlist",
             "--print", "%(is_live)s", url],
            capture_output=True, text=True, timeout=40,
        )
        return out.stdout.strip().lower() == "true"
    except Exception:
        log.exception("is_live probe failed; assuming VOD")
        return False


async def probe_is_live(url: str) -> bool:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _probe_is_live_blocking, url)


# --------------------------------------------------------------------------- #
# Section download (for --loop)
# --------------------------------------------------------------------------- #

def _download_cmd(url: str, out_dir: str, start: int, end: Optional[int],
                  want_audio: bool) -> list[str]:
    fmt = "worstvideo+worstaudio/worst" if want_audio else "worstvideo/worst"
    template = os.path.join(out_dir, "source.%(ext)s")
    cmd = ["yt-dlp", "-f", fmt, "--no-playlist", "--no-progress",
           "--merge-output-format", "mkv"]
    cmd += _sections_arg(start, end)
    cmd += ["-o", template, url]
    return cmd


async def download_source(url: str, out_dir: str, start: int, end: Optional[int],
                          want_audio: bool) -> Optional[str]:
    """Download the [start,end] section to a temp file for --loop.

    Cancellable: if the awaiting session is cancelled (client disconnect), the
    yt-dlp process is killed instead of finishing the whole download orphaned.
    """
    proc = subprocess.Popen(
        _download_cmd(url, out_dir, start, end, want_audio),
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    _spawn_stderr_drain(proc, "yt-dlp/loop")
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    try:
        while proc.poll() is None:
            if loop.time() - t0 > _DOWNLOAD_TIMEOUT:
                log.warning("loop: download timed out")
                break
            await asyncio.sleep(0.2)
    except asyncio.CancelledError:
        proc.kill()
        raise
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
    files = sorted(glob.glob(os.path.join(out_dir, "source.*")))
    if not files or os.path.getsize(files[0]) == 0:
        return None
    return files[0]


# --------------------------------------------------------------------------- #
# Producer iterators
# --------------------------------------------------------------------------- #

async def iter_video(youtube_url: str, term_w: int, term_h: int, fps: int,
                     start: int = 0, end: Optional[int] = None,
                     source_path: Optional[str] = None) -> AsyncGenerator[bytes, None]:
    """Yield encoded 2x3 binary frames.  Loops forever when source_path is set."""
    px_w, px_h = term_w * 2, term_h * 3
    ytdlp: subprocess.Popen | None = None
    ffmpeg: subprocess.Popen | None = None
    splitter = _FrameSplitter(px_w, px_h)
    try:
        if source_path:
            ffmpeg = subprocess.Popen(
                _video_ffmpeg_cmd(px_w, px_h, fps, source=source_path),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            _spawn_stderr_drain(ffmpeg, "ffmpeg")
        else:
            ytdlp = subprocess.Popen(
                _ytdlp_cmd(youtube_url, _VIDEO_FMT, start, end),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            ffmpeg = subprocess.Popen(
                _video_ffmpeg_cmd(px_w, px_h, fps),
                stdin=ytdlp.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            ytdlp.stdout.close()
            _spawn_stderr_drain(ytdlp, "yt-dlp")
            _spawn_stderr_drain(ffmpeg, "ffmpeg")

        while True:
            timeout = _FIRST_OUTPUT_TIMEOUT if splitter.count == 0 else _STALL_TIMEOUT
            try:
                chunk = await _read_with_timeout(ffmpeg, 65536, timeout)
            except TimeoutError:
                log.warning("video: no output for %ss — stopping", timeout)
                break
            if not chunk:
                break
            for frame in splitter.push(chunk):
                yield frame
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("video: pipeline error")
    finally:
        _kill_wait(ffmpeg, ytdlp)
        log.info("video: produced %d frame(s)", splitter.count)


async def iter_audio(youtube_url: str, sample_rate: int = 48000,
                     start: int = 0, end: Optional[int] = None,
                     source_path: Optional[str] = None) -> AsyncGenerator[bytes, None]:
    """Yield DFPWM1a audio chunks (mono).  Loops forever when source_path is set."""
    ytdlp: subprocess.Popen | None = None
    ffmpeg: subprocess.Popen | None = None
    sent = 0
    try:
        if source_path:
            ffmpeg = subprocess.Popen(
                _audio_ffmpeg_cmd(sample_rate, source=source_path),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            _spawn_stderr_drain(ffmpeg, "ffmpeg/audio")
        else:
            ytdlp = subprocess.Popen(
                _ytdlp_cmd(youtube_url, _AUDIO_FMT, start, end),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            ffmpeg = subprocess.Popen(
                _audio_ffmpeg_cmd(sample_rate),
                stdin=ytdlp.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            ytdlp.stdout.close()
            _spawn_stderr_drain(ytdlp, "yt-dlp/audio")
            _spawn_stderr_drain(ffmpeg, "ffmpeg/audio")

        while True:
            timeout = _FIRST_OUTPUT_TIMEOUT if sent == 0 else _STALL_TIMEOUT
            try:
                chunk = await _read_with_timeout(ffmpeg, AUDIO_READ_BYTES, timeout)
            except TimeoutError:
                log.warning("audio: no output for %ss — stopping", timeout)
                break
            if not chunk:
                break
            sent += len(chunk)
            yield chunk
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("audio: pipeline error")
    finally:
        _kill_wait(ffmpeg, ytdlp)
        log.info("audio: streamed %d bytes", sent)

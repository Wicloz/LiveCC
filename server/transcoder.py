"""
YouTube -> CC producers.

Two modes:
  * Streaming (default): yt-dlp (-o -) -> OS pipe -> ffmpeg (stdin) -> frames/PCM.
  * Looping (--loop):    download the [start,end] section once to a temp file,
                         then ffmpeg `-stream_loop -1` replays it.  Both video
                         and audio read the *same* file, so they loop at the
                         identical period and stay in sync across loop edges.

Timeline: every producer emits absolute PTS on ONE shared session timeline
(spec §4.2).  Video and audio are separate fetches of the same source, and on
a live stream each fetch joins the live edge wherever it happens to land — so
counting output (frame N -> N/fps, byte K -> K/rate) alone would put the two
streams up to several seconds apart.  Each pipeline therefore reports the
SOURCE timestamp of its first output (ffmpeg showinfo/ashowinfo, parsed from
stderr) to a per-session SourceTimeline, which converts the counted positions
onto the common timeline.  When timestamps are unavailable or implausible it
falls back to raw counters (correct for whole-file VOD, where both fetches
start at source zero anyway).

Audio: ffmpeg only ever DECODES (to raw u8 PCM, the CC speaker's native form);
iter_audio_roles cuts every negotiated channel role — including the mono
downmix — from ONE decode pass, so roles can never drift against each other.
DFPWM is encoded here (dfpwm.py), per chunk with fresh state, matching the
spec's per-chunk decoder reset (§4.6).

Pacing and buffering are handled downstream by session.StreamSession.

yt-dlp owns the download (auth, headers, JS challenge solving via deno).
WebM/VP9 + Opus are preferred for the streaming pipe (parseable from a pipe).
"""

from __future__ import annotations

import asyncio
import collections
import glob
import logging
import math
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Deque, Iterable, Iterator, Optional

import numpy as np

from cc_encoder import GopEncoder

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
# Audio picks the *best* stream: it's resampled down to 8-bit PCM, so feeding the
# resampler a clean source avoids stacking a second lossy pass on an already-
# crushed one.  Source audio is tiny next to video, so the download cost is negligible.
_VIDEO_FMT = "worstvideo[ext=webm]/worstvideo[vcodec^=vp9]/worstvideo/worst"
_AUDIO_FMT = "bestaudio[ext=webm]/bestaudio[acodec^=opus]/bestaudio/best"

# Watchdog timeouts (seconds): wait for the first byte, then max gap between reads.
_FIRST_OUTPUT_TIMEOUT = 45
_STALL_TIMEOUT = 30
_DOWNLOAD_TIMEOUT = 1800   # full section download for --loop

# Audio codecs.  The client advertises what it decodes in its CAPS message; the
# server prefers raw unsigned 8-bit PCM — the CC speaker's native format (1 byte/
# sample, no client decode, no 1-bit noise) — and falls back to DFPWM (1 bit/
# sample, ~8x less bandwidth, lower fidelity) when that's all the client takes.
# ffmpeg always decodes to PCM; DFPWM is packed per chunk by dfpwm.encode()
# with fresh state, because the spec (§4.6) makes every chunk independently
# decodable — a continuous ffmpeg DFPWM stream sliced into chunks would leave
# the client's per-chunk decoder resyncing at every boundary.
#
# Chunks stay short (~0.1 s): the client processes each chunk inline on the same
# coroutine that renders video, so a large chunk would stall rendering for the
# whole decode/unpack.  Sample count per chunk is the same for both codecs, so the
# A/V buffering downstream is codec-independent.
SAMPLE_RATE = 48000
AUDIO_CHUNK_SECONDS = 0.1
AUDIO_CHUNK_SAMPLES = int(SAMPLE_RATE * AUDIO_CHUNK_SECONDS)   # 4800

# Target span of one video GOP chunk (a palette + keyframe + delta/repeat units).
# Longer GOPs amortise the keyframe better but add that much latency to a live
# stream (frames are batched per GOP) and make each chunk a bigger burst.
GOP_SECONDS = 1.0
GOP_SAMPLES = int(SAMPLE_RATE * GOP_SECONDS)

# samples_per_byte describes the WIRE format (what the client unpacks);
# ffmpeg's decode output is always 1 byte/sample u8 PCM regardless.
AudioCodec = collections.namedtuple("AudioCodec", "name samples_per_byte")

PCM = AudioCodec("pcm", 1)        # preferred
DFPWM = AudioCodec("dfpwm", 8)    # negotiated fallback (packed by dfpwm.encode)
AUDIO_CODECS = {c.name: c for c in (PCM, DFPWM)}

# No server-side audio filtering.  Earlier encode-side processing (highpass /
# lowpass / volume, and before that dynaudnorm / loudnorm / limiter) was reported
# to make audio worse, so we resample the source straight to PCM with no filters.
# (Positional extraction below uses `pan` purely to pick a channel, not to filter.)

# Discrete source-channel index (0-based) ffmpeg exposes for each positional
# role, following the standard multichannel decode order FL FR FC LFE BL BR
# [SL SR] -- the same order the CCMF role IDs are numbered in (spec §4.6).
# Role 0 (mono) has no entry: it's always the full downmix of every source
# channel (see _audio_ffmpeg_cmd's source_channel=None case), not one discrete
# channel.
ROLE_SOURCE_CHANNEL = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}


def negotiate_channel_roles(requested_channels: int, source_channels: int) -> list[int]:
    """Pick which CCMF channel roles (spec §4.6) to actually produce.

    `requested_channels` is a CAPS `channels` bitmask -- for a private session
    that's simply the one client's request; for a shared (sync) room it's the
    OR of every current subscriber's request, so e.g. one client wanting
    mono+lfe and another wanting front_left+front_right together produce all
    four distinct roles (session.StreamSession.reconfigure_channels re-derives
    this union and calls back in whenever a sync room's membership changes).

    Every individual role bit set in `requested_channels` is produced as long
    as the source can actually supply it: role 0 (mono) always can (it's a
    downmix of everything), a positional role N needs the source to have more
    than ROLE_SOURCE_CHANNEL[N] discrete channels. There's no bundling
    requirement (e.g. asking for just front_left without front_right is
    honoured as just front_left) -- a room's union may well be a combination
    no single client would ever request on its own.  Falls back to [0] (mono)
    if the request has nothing the source can supply, since mono is the
    universal fallback (spec §5.4) and always available.
    """
    roles = [0] if requested_channels & 1 else []   # bit0 = mono (ccmf.CAP_CHANNEL_MONO)
    for role, idx in ROLE_SOURCE_CHANNEL.items():
        if requested_channels & (1 << role) and idx < source_channels:
            roles.append(role)
    return sorted(roles) or [0]


# --------------------------------------------------------------------------- #
# Section selection (timestamps arrive from the client in ms, as seconds here)
# --------------------------------------------------------------------------- #

def _sections_arg(start: float, end: Optional[float]) -> list[str]:
    """yt-dlp --download-sections for a [start, end] window (seconds)."""
    if start <= 0 and end is None:
        return []
    s = max(0, start)
    e = "inf" if end is None else format(max(s, end), "g")
    return ["--download-sections", f"*{s:g}-{e}"]


# --------------------------------------------------------------------------- #
# Subprocess helpers
# --------------------------------------------------------------------------- #

def _spawn_stderr_drain(proc: subprocess.Popen, name: str,
                        probe: "Optional[_FirstPtsProbe]" = None) -> Deque[str]:
    """Drain proc.stderr in a daemon thread (an unread PIPE can deadlock the child).

    `probe` consumes showinfo/ashowinfo frame reports (source-PTS capture); those
    lines are dropped from the log — one per frame would drown everything else.
    """
    tail: Deque[str] = collections.deque(maxlen=30)

    def _drain() -> None:
        if proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line and not (probe is not None and probe.feed(line)):
                log.info("[%s] %s", name, line)
                tail.append(line)

    threading.Thread(target=_drain, name=f"stderr-{name}", daemon=True).start()
    return tail


class _FirstPtsProbe:
    """Source timestamp of a pipeline's FIRST output frame, scraped from ffmpeg's
    showinfo (video) / ashowinfo (audio) stderr reports.

    Only the first frame report counts: the pipelines' local counters advance
    from output position 0, so the shared-timeline mapping is exactly "where in
    the source was output 0" (SourceTimeline).  If that report carries no
    usable pts (`pts_time:NOPTS`), value stays None — a later frame's pts would
    map the wrong output position, so it must not be used instead.
    """

    _PTS_RE = re.compile(r"pts_time:(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

    def __init__(self) -> None:
        self.seen = False                 # a frame report arrived (parsed or not)
        self.value: Optional[float] = None

    def feed(self, line: str) -> bool:
        """Consume `line` if it's showinfo noise; True means "don't log it"."""
        if "showinfo" not in line:        # "ashowinfo" contains "showinfo" too
            return False
        if not self.seen and "pts_time:" in line:
            self.seen = True
            m = self._PTS_RE.search(line)
            if m:
                self.value = float(m.group(1))
        return True


async def _first_pts(probe: Optional[_FirstPtsProbe],
                     grace: float = 2.0) -> Optional[float]:
    """The probe's captured pts, waiting up to `grace` for the stderr line to
    race in behind the stdout bytes that triggered the caller."""
    if probe is None:
        return None
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    while not probe.seen and loop.time() - t0 < grace:
        await asyncio.sleep(0.05)
    return probe.value


class SourceTimeline:
    """Rendezvous that puts independently-fetched pipelines on ONE timeline.

    Each producer reports the source time of its output zero (offset_samples);
    once every expected pipeline has reported — or the timeout expires — the
    earliest reported base becomes the session's zero and each pipeline gets
    the sample offset that places its counter on the shared timeline.

    Fallback is deliberate and total: if ANY pipeline couldn't read a source
    timestamp, or the bases are implausibly far apart (different timestamp
    epochs, e.g. an HLS/DASH mix), every offset is 0 — identical behaviour to
    plain counting, which is already correct for whole-file VOD.  A partial
    correction could be worse than none.
    """

    # A live fetch pair should land within a few segments of each other; bases
    # further apart than this are two different clocks, not a fetch skew.
    MAX_SKEW = 30.0

    def __init__(self, expected: Iterable[str], timeout: float = 20.0) -> None:
        self._expected = set(expected)
        self._timeout = timeout
        self._bases: dict[str, Optional[float]] = {}
        self._decided = asyncio.Event()
        self._zero: Optional[float] = None

    async def offset_samples(self, name: str, base: Optional[float]) -> int:
        """Report `base` (source seconds of this pipeline's output 0, or None
        if unknown) and wait for the group decision; -> samples to ADD to this
        pipeline's counted PTS."""
        if not self._decided.is_set():
            self._bases[name] = base
            if set(self._bases) >= self._expected:
                self._decide()
            else:
                try:
                    await asyncio.wait_for(self._decided.wait(), self._timeout)
                except asyncio.TimeoutError:
                    log.warning("timeline: %s reported, still missing %s after "
                                "%.0fs — falling back to raw counters",
                                sorted(self._bases),
                                sorted(self._expected - set(self._bases)),
                                self._timeout)
                    self._decide()
        if self._zero is None or base is None:
            return 0
        return max(0, round((base - self._zero) * SAMPLE_RATE))

    def report(self, name: str, base: Optional[float]) -> None:
        """Non-blocking offset_samples for a pipeline that is going away: it
        records the base (deciding if that completes the set) but never waits.
        Used in producer teardown, where awaiting the group decision could
        stall generator close until the timeout on a cancelled session."""
        if not self._decided.is_set():
            self._bases[name] = base
            if set(self._bases) >= self._expected:
                self._decide()

    def _decide(self) -> None:
        if self._decided.is_set():
            return
        vals = [b for b in self._bases.values() if b is not None]
        aligned = (set(self._bases) >= self._expected
                   and len(vals) == len(self._bases)
                   and vals
                   and max(vals) - min(vals) <= self.MAX_SKEW)
        if aligned:
            self._zero = min(vals)
            if len(self._expected) > 1:
                log.info("timeline: aligned %s (zero=%.3fs, spread=%.3fs)",
                         {k: round(v, 3) for k, v in self._bases.items()},
                         self._zero, max(vals) - min(vals))
        elif len(self._expected) > 1:
            log.warning("timeline: cannot align %s — falling back to raw "
                        "counters", self._bases)
        self._decided.set()


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
    """Accumulate raw rgb24 bytes and emit each complete frame as an (H, W, 3) array.

    Splitting only — the CPU-heavy encode (GopEncoder.add) is applied separately
    (in iter_video) so it can run off the event loop in a worker thread.  Keeping
    the two apart is what lets the pacing scheduler stay responsive; see iter_video.
    """

    def __init__(self, px_w: int, px_h: int) -> None:
        self.px_w = px_w
        self.px_h = px_h
        self.frame_bytes = px_w * px_h * 3
        self._buf = bytearray()
        self._n = 0

    def push(self, chunk: bytes) -> Iterator[np.ndarray]:
        self._buf.extend(chunk)
        while len(self._buf) >= self.frame_bytes:
            raw = bytes(self._buf[: self.frame_bytes])
            del self._buf[: self.frame_bytes]
            self._n += 1
            yield np.frombuffer(raw, dtype=np.uint8).reshape(self.px_h, self.px_w, 3)

    @property
    def count(self) -> int:
        return self._n


# --------------------------------------------------------------------------- #
# Command builders
# --------------------------------------------------------------------------- #

def _ytdlp_cmd(youtube_url: str, fmt: str, start: float = 0,
               end: Optional[float] = None) -> list[str]:
    cmd = ["yt-dlp", "-f", fmt, "--no-playlist", "--no-progress"]
    cmd += _sections_arg(start, end)   # server-side seek via range requests
    cmd += ["-o", "-", youtube_url]
    return cmd


def _video_ffmpeg_cmd(px_w: int, px_h: int, fps: int,
                      source: Optional[str] = None, loop: bool = False) -> list[str]:
    # Letterbox aligned to the 2x3 character grid: snap the scaled content to
    # even width / multiple-of-3 height and pad at even-x / multiple-of-3-y
    # offsets.  Otherwise the content edge lands mid-cell, so boundary cells mix
    # content with black bar and the content's corners quantize to black.
    # Area filter for the content scale: when the source is larger than the cell
    # grid it box-integrates the pixels each sub-pixel covers (the "partial
    # overlap" the encoder wants) instead of point-sampling; on upscale it behaves
    # like bilinear.  The encoder makes its colour decisions in linear OKLab.
    # showinfo last (after fps): its first report is the source timestamp of
    # output frame 0 exactly — the SourceTimeline base (captured off stderr by
    # _FirstPtsProbe; rawvideo itself carries no timestamps).
    scale = (
        f"scale={px_w}:{px_h}:force_original_aspect_ratio=decrease:flags=area,"
        f"scale=trunc(iw/2)*2:trunc(ih/3)*3:flags=area,"
        f"pad={px_w}:{px_h}:trunc((ow-iw)/4)*2:trunc((oh-ih)/6)*3:black,"
        f"fps={fps},showinfo"
    )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats"]
    if source:                                   # decode a local (seekable) file
        if loop:                                 # --loop: replay the section forever
            cmd += ["-stream_loop", "-1"]
        cmd += ["-i", source, "-map", "0:v:0"]
    else:                                        # stream from yt-dlp pipe
        cmd += ["-i", "pipe:0"]
    cmd += ["-vf", scale, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    return cmd


def _audio_ffmpeg_cmd(sample_rate: int,
                      source_channel: Optional[int] = None,
                      source: Optional[str] = None, loop: bool = False) -> list[str]:
    # Decode-only: always raw u8 PCM at the speaker rate (the wire PCM format;
    # DFPWM, when negotiated, is packed from this in Python — see dfpwm.py).
    # source_channel=None: downmix every source channel to mono (role 0, the
    # default/only mode until a client asks for positional roles).
    # source_channel=N: extract discrete source channel N with `pan` -- no
    # mixing -- so a positional role (e.g. front_left) doesn't bleed into its
    # neighbour (see ROLE_SOURCE_CHANNEL).
    # ashowinfo is inspection only (first-frame source PTS for SourceTimeline),
    # not filtering; it taps the stream before any channel work.
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats"]
    if source:
        if loop:
            cmd += ["-stream_loop", "-1"]
        cmd += ["-i", source, "-map", "0:a:0?"]
    else:
        cmd += ["-i", "pipe:0"]
    cmd += ["-vn", "-ar", str(sample_rate)]
    if source_channel is None:
        cmd += ["-af", "ashowinfo", "-ac", "1"]
    else:
        cmd += ["-af", f"ashowinfo,pan=mono|c0=c{source_channel}"]
    cmd += ["-c:a", "pcm_u8", "-f", "u8", "pipe:1"]
    return cmd


# --------------------------------------------------------------------------- #
# is_live probe
# --------------------------------------------------------------------------- #

# Containers (the ISOBMFF / MP4 family) whose index (moov atom) may sit at the end
# of the file.  ffmpeg can't demux those from a non-seekable pipe — it would have
# to read the whole stream to reach moov, by which point mdat is gone — so such a
# VOD is checked with the moov probe and, if moov-at-end, decoded from a downloaded
# (seekable) temp file instead.  All share the moov/mdat box layout, so the same
# _scan_moov_position() handles every one.  WebM / Matroska stream fine from a pipe.
_SEEKABLE_REQUIRED_EXTS = {
    "mp4", "m4v", "m4a", "m4b",   # MPEG-4 (video / audio / audiobook)
    "mov", "qt",                  # QuickTime
    "3gp", "3g2",                 # 3GPP / 3GPP2
    "f4v",                        # Flash MP4
    "mj2", "mjp2",                # Motion JPEG 2000
}


def _probe_source_blocking(url: str) -> tuple[bool, str]:
    """Return (is_live, video_ext) for the format we'd actually stream.

    The ext is resolved against _VIDEO_FMT so it reflects the selected stream
    (e.g. YouTube prefers webm), not the default best format.
    """
    try:
        out = subprocess.run(
            ["yt-dlp", "--no-warnings", "--quiet", "--no-playlist",
             "-f", _VIDEO_FMT, "--print", "%(is_live)s\n%(ext)s", url],
            capture_output=True, text=True, timeout=40,
        )
        lines = out.stdout.strip().splitlines()
        is_live = bool(lines) and lines[0].strip().lower() == "true"
        ext = lines[1].strip().lower() if len(lines) > 1 else ""
        return is_live, ext
    except Exception:
        log.exception("source probe failed; assuming VOD, streamable")
        return False, ""


async def probe_source_info(url: str) -> tuple[bool, str]:
    """(is_live, video_ext) — see _probe_source_blocking."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _probe_source_blocking, url)


def needs_seekable_source(ext: str) -> bool:
    """True if a VOD in container `ext` *might* hide its index at the end.

    Only a coarse container gate — pairs with moov_at_end() to decide for real,
    so a faststart MP4 (moov up front) still streams instead of downloading.
    """
    return ext.lower() in _SEEKABLE_REQUIRED_EXTS


# Formats that can't be decoded from the non-seekable yt-dlp pipe and must be
# downloaded to a file first — unconditionally (no moov-style probe).  GIF is the
# known case: the server's ffmpeg (bookworm 5.1) demuxes a GIF fine from a file
# but produces no frames from a pipe, so a streamed GIF failed with "Failed to
# load video".  Downloaded, it decodes normally (and plays once unless --loop).
_DOWNLOAD_REQUIRED_EXTS = {"gif"}


def needs_download(ext: str) -> bool:
    """True if `ext` can't be pipe-streamed and must be downloaded to a file first."""
    return ext.lower() in _DOWNLOAD_REQUIRED_EXTS


# Cap on how much of the file head to scan for the moov/mdat order.  The decisive
# top-level box header is almost always in the first few KB; this is just a bound.
_MOOV_PROBE_BYTES = 256 * 1024


def _scan_moov_position(buf: bytes) -> Optional[bool]:
    """Walk ISOBMFF top-level boxes in `buf`.

    Returns True if `mdat` is reached before `moov` (moov-at-end, not streamable),
    False if `moov` comes first (faststart, streamable), or None if `buf` doesn't
    yet contain enough to decide.  Only box headers are read; payloads are skipped
    by size, so a huge leading mdat is identified without reading it.
    """
    pos = 0
    while pos + 8 <= len(buf):
        size = int.from_bytes(buf[pos:pos + 4], "big")
        btype = buf[pos + 4:pos + 8]
        if btype == b"moov":
            return False
        if btype == b"mdat":
            return True
        if size == 1:                              # 64-bit largesize after the type
            if pos + 16 > len(buf):
                return None
            size = int.from_bytes(buf[pos + 8:pos + 16], "big")
        if size < 8:                               # 0 (extends to EOF) or malformed
            return None
        pos += size
    return None


def _probe_moov_at_end_blocking(url: str, timeout: float = 30.0) -> bool:
    """Stream the file head via yt-dlp and report whether moov sits after mdat.

    Defaults to True (treat as moov-at-end -> download) on any uncertainty, so a
    file we can't classify is still handled correctly, just without the streaming
    optimisation.
    """
    proc = subprocess.Popen(
        _ytdlp_cmd(url, _VIDEO_FMT), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    killer = threading.Timer(timeout, proc.kill)
    killer.start()
    try:
        buf = b""
        while len(buf) < _MOOV_PROBE_BYTES:
            chunk = proc.stdout.read(8192)
            if not chunk:
                break
            buf += chunk
            verdict = _scan_moov_position(buf)
            if verdict is not None:
                return verdict
        verdict = _scan_moov_position(buf)
        return True if verdict is None else verdict
    except Exception:
        log.exception("moov probe failed; assuming moov-at-end")
        return True
    finally:
        killer.cancel()
        _kill_wait(proc)


async def probe_moov_at_end(url: str) -> bool:
    """Async wrapper for _probe_moov_at_end_blocking."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _probe_moov_at_end_blocking, url)


async def probe_is_live(url: str) -> bool:
    is_live, _ext = await probe_source_info(url)
    return is_live


def _probe_audio_channels_blocking(url: str) -> int:
    """Discrete channel count of the selected audio format (best-effort).

    Defaults to 1 (mono) on any uncertainty -- the previously-only behaviour --
    so a probe failure just means no positional audio, not a broken stream.
    """
    try:
        out = subprocess.run(
            ["yt-dlp", "--no-warnings", "--quiet", "--no-playlist",
             "-f", _AUDIO_FMT, "--print", "%(audio_channels)s", url],
            capture_output=True, text=True, timeout=40,
        )
        line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        return int(line) if line.isdigit() else 1
    except Exception:
        log.exception("audio channel probe failed; assuming mono")
        return 1


async def probe_audio_channels(url: str) -> int:
    """Async wrapper for _probe_audio_channels_blocking."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _probe_audio_channels_blocking, url)


# --------------------------------------------------------------------------- #
# Section download (for --loop)
# --------------------------------------------------------------------------- #

def _download_cmd(url: str, out_dir: str, start: float, end: Optional[float],
                  want_audio: bool) -> list[str]:
    fmt = "worstvideo+worstaudio/worst" if want_audio else "worstvideo/worst"
    template = os.path.join(out_dir, "source.%(ext)s")
    cmd = ["yt-dlp", "-f", fmt, "--no-playlist", "--no-progress",
           "--merge-output-format", "mkv"]
    cmd += _sections_arg(start, end)
    cmd += ["-o", template, url]
    return cmd


async def download_source(url: str, out_dir: str, start: float, end: Optional[float],
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
# Adaptive frame pacing
# --------------------------------------------------------------------------- #
# Per-frame encode cost (GopEncoder.add) scales with the cell grid; on a big
# monitor a frame can take longer than its slot at the requested fps.  Rather
# than out-run the encoder (frames pile up, the buffer drains, playback stalls
# into constant re-buffering), we encode only every Nth source frame so the
# EFFECTIVE fps falls to a steady rate the CPU sustains — low but smooth beats
# stuttery.  N tracks a smoothed encode time, so it adapts to the host and to
# load from other streams sharing the worker pool.  Emitted frames carry their
# true source PTS (as their durations inside the GOP), so audio and the playback
# clock stay in sync regardless of N.
_PACE_SAFETY = 1.15      # leave ~15% headroom over the measured encode time
_PACE_EMA = 0.2          # weight of the newest encode-time sample in the average


def _encode_stride(enc_seconds: float, fps: int) -> int:
    """How many source frames each encoded frame should span to keep up.

    Keeping up needs encode_time <= stride/fps, i.e. stride >= encode_time*fps;
    clamped to [1, fps] (never finer than every frame, never below 1 fps).
    """
    return max(1, min(fps, math.ceil(enc_seconds * fps * _PACE_SAFETY)))


# --------------------------------------------------------------------------- #
# Producer iterators
# --------------------------------------------------------------------------- #

async def iter_video(youtube_url: str, term_w: int, term_h: int, fps: int,
                     start: float = 0, end: Optional[float] = None,
                     source_path: Optional[str] = None,
                     loop: bool = False,
                     timeline: Optional[SourceTimeline] = None,
                     ) -> AsyncGenerator[tuple[int, bytes], None]:
    """Yield (pts_samples, CCMF video chunk) pairs — each chunk one self-contained
    GOP (~GOP_SECONDS of palette + raw/delta/repeat units, see cc_encoder.GopEncoder).

    pts is the chunk's first frame in 48 kHz samples (source_index/fps of that
    frame plus the SourceTimeline offset), so the consumer can pace it against
    the shared clock even when adaptive pacing skips source frames.

    source_path set => decode that local (seekable) file; loop=True replays it
    forever (--loop).  Otherwise stream from the yt-dlp pipe.
    """
    px_w, px_h = term_w * 2, term_h * 3
    ytdlp: subprocess.Popen | None = None
    ffmpeg: subprocess.Popen | None = None
    splitter = _FrameSplitter(px_w, px_h)
    gop = GopEncoder(gop_samples=GOP_SAMPLES,
                     nominal_duration=round(SAMPLE_RATE / fps))
    ev_loop = asyncio.get_running_loop()
    probe = _FirstPtsProbe()
    base_off = 0                # samples; resolved against timeline at first frame
    resolved = timeline is None
    # Adaptive pacing state: src_i counts source frames, next_i is the next source
    # index we'll actually encode, enc_ema smooths the encode wall-time, encoded
    # counts what we emitted.  Initialised before the try so finally can log them.
    src_i = next_i = encoded = 0
    enc_ema = 0.0
    try:
        if source_path:
            ffmpeg = subprocess.Popen(
                _video_ffmpeg_cmd(px_w, px_h, fps, source=source_path, loop=loop),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            _spawn_stderr_drain(ffmpeg, "ffmpeg", probe=probe)
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
            _spawn_stderr_drain(ffmpeg, "ffmpeg", probe=probe)

        while True:
            timeout = _FIRST_OUTPUT_TIMEOUT if src_i == 0 else _STALL_TIMEOUT
            try:
                chunk = await _read_with_timeout(ffmpeg, 65536, timeout)
            except TimeoutError:
                log.warning("video: no output for %ss — stopping", timeout)
                break
            if not chunk:
                break
            for arr in splitter.push(chunk):
                if not resolved:
                    base_off = await timeline.offset_samples(
                        "video", await _first_pts(probe))
                    resolved = True
                idx = src_i
                src_i += 1
                if idx < next_i:
                    continue                         # shed load: skip this frame
                # Encode off the event loop.  GopEncoder.add() is CPU-heavy numpy;
                # run inline it would block the single-threaded pacing scheduler
                # (session.run) between frames, so chunks go out in bursts and the
                # client stutters.  numpy releases the GIL, so offloading lets the
                # loop release buffered chunks on a steady cadence while this worker
                # encodes.  Frames are awaited one at a time, preserving order.
                #
                # Deferred redesign: move the whole video pipeline (read + encode)
                # onto its own thread for cleaner isolation.  Bigger change — it
                # reintroduces backpressure (TimedBuffer.put) and cancellation
                # (agen.aclose) as things we'd hand-roll across a sync/async queue.
                # See memory note [[video-encode-offload]].  Not worth it until this
                # targeted offload proves insufficient.
                pts = round(idx * SAMPLE_RATE / fps) + base_off
                t0 = ev_loop.time()
                done = await ev_loop.run_in_executor(_executor, gop.add, pts, arr)
                enc = ev_loop.time() - t0
                # Smoothed encode time -> how many source frames to span next, so
                # the effective fps tracks what the CPU can actually sustain.
                enc_ema = enc if encoded == 0 else \
                    (1 - _PACE_EMA) * enc_ema + _PACE_EMA * enc
                next_i = idx + _encode_stride(enc_ema, fps)
                encoded += 1
                if done is not None:                 # this frame opened a new GOP
                    yield done
        done = await ev_loop.run_in_executor(_executor, gop.flush)
        if done is not None:                         # trailing partial GOP
            yield done
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("video: pipeline error")
    finally:
        _kill_wait(ffmpeg, ytdlp)
        if not resolved:
            # Never produced a frame: report so the audio pipeline stops waiting
            # for us (it falls back to raw counters rather than hanging).
            timeline.report("video", None)
        eff = encoded / splitter.count * fps if splitter.count else fps
        log.info("video: encoded %d of %d source frame(s) (~%.1f fps effective)",
                 encoded, splitter.count, eff)


async def iter_audio(youtube_url: str, sample_rate: int = 48000,
                     start: float = 0, end: Optional[float] = None,
                     source_path: Optional[str] = None,
                     loop: bool = False,
                     source_channel: Optional[int] = None,
                     probe: Optional[_FirstPtsProbe] = None,
                     ) -> AsyncGenerator[bytes, None]:
    """Yield ~0.1 s chunks of raw u8 PCM for ONE stream: the mono downmix
    (source_channel=None) or one discrete source channel (spec §4.6, see
    ROLE_SOURCE_CHANNEL).  Codec packing and PTS live in iter_audio_roles —
    this is just the decode tap.

    source_path set => decode that local (seekable) file; loop=True replays it
    forever (--loop).  Otherwise stream from the yt-dlp pipe.
    """
    ytdlp: subprocess.Popen | None = None
    ffmpeg: subprocess.Popen | None = None
    sent = 0
    try:
        if source_path:
            ffmpeg = subprocess.Popen(
                _audio_ffmpeg_cmd(sample_rate, source_channel=source_channel,
                                  source=source_path, loop=loop),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            _spawn_stderr_drain(ffmpeg, "ffmpeg/audio", probe=probe)
        else:
            ytdlp = subprocess.Popen(
                _ytdlp_cmd(youtube_url, _AUDIO_FMT, start, end),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            ffmpeg = subprocess.Popen(
                _audio_ffmpeg_cmd(sample_rate, source_channel=source_channel),
                stdin=ytdlp.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            ytdlp.stdout.close()
            _spawn_stderr_drain(ytdlp, "yt-dlp/audio")
            _spawn_stderr_drain(ffmpeg, "ffmpeg/audio", probe=probe)

        while True:
            timeout = _FIRST_OUTPUT_TIMEOUT if sent == 0 else _STALL_TIMEOUT
            try:
                chunk = await _read_with_timeout(ffmpeg, AUDIO_CHUNK_SAMPLES, timeout)
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


def _multichannel_pcm_cmd(sample_rate: int, n_channels: int,
                          source: Optional[str] = None, loop: bool = False) -> list[str]:
    """Raw interleaved PCM8 at n_channels, undownmixed -- ffmpeg passes discrete
    source channels through unchanged when -ac equals the source's channel
    count, which the caller guarantees (decode_channels is the probed source
    width).  ashowinfo taps the first-frame source PTS for SourceTimeline."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats"]
    if source:
        if loop:
            cmd += ["-stream_loop", "-1"]
        cmd += ["-i", source, "-map", "0:a:0?"]
    else:
        cmd += ["-i", "pipe:0"]
    cmd += ["-vn", "-ar", str(sample_rate), "-af", "ashowinfo",
            "-ac", str(n_channels), "-c:a", "pcm_u8", "-f", "u8", "pipe:1"]
    return cmd


# Decode index of the LFE channel in the standard order FL FR FC LFE BL BR
# [SL SR] — excluded from the mono downmix (it's band-limited rumble; folding
# it in at equal weight just muddies the mix).
_LFE_INDEX = 3


def mono_downmix(interleaved: np.ndarray) -> bytes:
    """(n, C) u8 PCM -> mono u8 PCM: equal-weight average of the non-LFE
    channels.  ffmpeg's downmix matrix weights channels slightly differently;
    through a CC speaker the difference is inaudible, and computing it here
    keeps mono cut from the SAME decode pass as the positional roles."""
    n, c = interleaved.shape
    cols = [i for i in range(c) if c < 4 or i != _LFE_INDEX]
    sel = interleaved[:, cols].astype(np.uint32)
    return ((sel.sum(axis=1) + len(cols) // 2) // len(cols)).astype(np.uint8).tobytes()


async def iter_audio_roles(youtube_url: str, sample_rate: int, roles: list[int],
                           decode_channels: int = 1, start: float = 0,
                           end: Optional[float] = None,
                           source_path: Optional[str] = None,
                           loop: bool = False,
                           timeline: Optional[SourceTimeline] = None,
                           ) -> AsyncGenerator[tuple[int, dict[int, bytes]], None]:
    """Yield (pts_samples, {role: u8 PCM chunk}) covering every role in `roles`
    — the ONE audio producer a session runs, whatever its channel layout.

    Everything is cut from a single fetch + decode pass:
      * positional roles are de-interleaved from the discrete channels,
      * mono (role 0) is downmixed from the same frames,
    so every role's chunk at a given pts holds the exact same source samples.
    Independent per-role (or separate mono/positional) pipelines are how
    channels end up offset from each other on a live source — two fetches of
    "the stream, right now" don't join at the same sample.

    pts is absolute on the shared session timeline (SourceTimeline), same as
    iter_video's, and contiguous: chunk N+1's pts = chunk N's pts + samples.

    decode_channels == 1 (mono-only sessions) skips the de-interleave and uses
    the plain downmix pipeline.  Wire-codec packing (DFPWM) is the caller's
    job — these chunks are the PCM truth.
    """
    probe = _FirstPtsProbe()
    resolved = timeline is None
    base_off = 0
    samples = 0

    async def _resolve() -> int:
        return await timeline.offset_samples("audio", await _first_pts(probe))

    try:
        if decode_channels <= 1:
            role = roles[0] if roles else 0
            agen = iter_audio(youtube_url, sample_rate, start, end,
                              source_path, loop,
                              source_channel=ROLE_SOURCE_CHANNEL.get(role),
                              probe=probe)
            try:
                async for data in agen:
                    if not resolved:
                        base_off = await _resolve()
                        resolved = True
                    yield samples + base_off, {role: data}
                    samples += len(data)
            finally:
                await agen.aclose()
            return

        frame_bytes = AUDIO_CHUNK_SAMPLES * decode_channels
        ytdlp: subprocess.Popen | None = None
        ffmpeg: subprocess.Popen | None = None
        try:
            if source_path:
                ffmpeg = subprocess.Popen(
                    _multichannel_pcm_cmd(sample_rate, decode_channels,
                                          source=source_path, loop=loop),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                _spawn_stderr_drain(ffmpeg, "ffmpeg/audio", probe=probe)
            else:
                ytdlp = subprocess.Popen(
                    _ytdlp_cmd(youtube_url, _AUDIO_FMT, start, end),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                ffmpeg = subprocess.Popen(
                    _multichannel_pcm_cmd(sample_rate, decode_channels),
                    stdin=ytdlp.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                ytdlp.stdout.close()
                _spawn_stderr_drain(ytdlp, "yt-dlp/audio")
                _spawn_stderr_drain(ffmpeg, "ffmpeg/audio", probe=probe)

            while True:
                timeout = _FIRST_OUTPUT_TIMEOUT if samples == 0 else _STALL_TIMEOUT
                try:
                    raw = await _read_with_timeout(ffmpeg, frame_bytes, timeout)
                except TimeoutError:
                    log.warning("audio: no output for %ss — stopping", timeout)
                    break
                if not raw:
                    break
                usable = len(raw) - (len(raw) % decode_channels)  # torn trailing frame
                if usable == 0:
                    continue
                if not resolved:
                    base_off = await _resolve()
                    resolved = True
                arr = np.frombuffer(raw, np.uint8,
                                    count=usable).reshape(-1, decode_channels)
                chunks = {role: arr[:, ROLE_SOURCE_CHANNEL[role]].tobytes()
                          for role in roles if role != 0}
                if 0 in roles:
                    chunks[0] = mono_downmix(arr)
                yield samples + base_off, chunks
                samples += arr.shape[0]
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("audio: pipeline error")
        finally:
            _kill_wait(ffmpeg, ytdlp)
            log.info("audio: streamed %d samples (%d channels)",
                     samples, decode_channels)
    finally:
        if not resolved:
            # Never produced a chunk: report so the video pipeline stops
            # waiting for us (falls back to raw counters, doesn't hang).
            timeline.report("audio", None)

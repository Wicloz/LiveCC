"""
render_cc — convert a media file into a .ccmf file.

Stand-alone CLI wrapped around the *real* producer code (transcoder.py,
cc_encoder.GopEncoder, ccmf.py, dfpwm.py): the same GOP encoding, palette
generation, and audio-role/codec packing a live LiveCC session uses, just run
to completion against a file instead of a WebSocket.

Two differences from the live path:

  * No adaptive pacing. A live session skips source frames when the encoder
    can't keep up with real time (transcoder._encode_stride); a render has no
    "real time" to keep up with, so every source frame is encoded and the
    output plays at the full requested --fps regardless of how long the
    encode takes.
  * No streaming/buffering. session.StreamSession's TimedBuffers and release
    clock exist to pace delivery to a live client; here video GOPs and audio
    chunks are produced concurrently and simply merged into the output file
    in ascending PTS order as they complete.

Source can be a local file (any container ffmpeg can open) or a yt-dlp URL;
the same source-selection logic the server uses picks a streaming pipe vs. a
one-shot download (moov-at-end MP4, GIF) — see transcoder.probe_source_info /
needs_download / needs_seekable_source. Live sources are rejected: an
unbounded stream has no natural file length.

Examples:
  python tools/render_cc.py clip.mp4
  python tools/render_cc.py clip.mp4 --grid mon7x4 --fps 30 --channels stereo
  python tools/render_cc.py https://youtu.be/XXXXXXXXXXX --start 30 --duration 20
  python tools/render_cc.py clip.mkv --audio-codec dfpwm --channels 5.1 -o out.ccmf
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# Make the server modules importable when run as a standalone script.
_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

import ccmf  # noqa: E402  (needs the sys.path insert above)
import dfpwm  # noqa: E402
from cc_media import GRIDS, have_ffmpeg  # noqa: E402
from transcoder import (  # noqa: E402
    AUDIO_CHUNK_SECONDS,
    GOP_SECONDS,
    SourceTimeline,
    _ASSUMED_CHANNELS,
    _ffprobe_channels,
    download_source,
    iter_audio_roles,
    iter_video,
    needs_download,
    needs_seekable_source,
    negotiate_channel_roles,
    probe_audio_channels,
    probe_moov_at_end,
    probe_source_info,
)

log = logging.getLogger("livecc")

# --------------------------------------------------------------------------- #
# CLI vocabulary: grid presets (reused from cc_media) and channel layouts
# --------------------------------------------------------------------------- #

_GRID_PRESETS = {label: (w, h) for label, w, h in GRIDS}

_ROLE_NAMES = {
    "mono": ccmf.CHANNEL_MONO,
    "fl": ccmf.CHANNEL_FRONT_LEFT,
    "fr": ccmf.CHANNEL_FRONT_RIGHT,
    "c": ccmf.CHANNEL_CENTER,
    "center": ccmf.CHANNEL_CENTER,
    "lfe": ccmf.CHANNEL_LFE,
    "sl": ccmf.CHANNEL_SURROUND_LEFT,
    "sr": ccmf.CHANNEL_SURROUND_RIGHT,
    "rl": ccmf.CHANNEL_REAR_LEFT,
    "rr": ccmf.CHANNEL_REAR_RIGHT,
}

_CHANNEL_PRESETS = {
    "mono": ccmf.CAP_CHANNEL_MONO,
    "stereo": ccmf.CAP_CHANNEL_FRONT_LEFT | ccmf.CAP_CHANNEL_FRONT_RIGHT,
    "5.1": (ccmf.CAP_CHANNEL_FRONT_LEFT | ccmf.CAP_CHANNEL_FRONT_RIGHT
            | ccmf.CAP_CHANNEL_CENTER | ccmf.CAP_CHANNEL_LFE
            | ccmf.CAP_CHANNEL_SURROUND_LEFT | ccmf.CAP_CHANNEL_SURROUND_RIGHT),
    "7.1": (ccmf.CAP_CHANNEL_FRONT_LEFT | ccmf.CAP_CHANNEL_FRONT_RIGHT
            | ccmf.CAP_CHANNEL_CENTER | ccmf.CAP_CHANNEL_LFE
            | ccmf.CAP_CHANNEL_SURROUND_LEFT | ccmf.CAP_CHANNEL_SURROUND_RIGHT
            | ccmf.CAP_CHANNEL_REAR_LEFT | ccmf.CAP_CHANNEL_REAR_RIGHT),
    "all": sum(1 << role for role in _ROLE_NAMES.values()),
}


def _resolve_grid(spec: str, width: Optional[int], height: Optional[int]) -> tuple[int, int]:
    if width and height:
        return width, height
    key = spec.strip().lower()
    if key in _GRID_PRESETS:
        w, h = _GRID_PRESETS[key]
    elif "x" in key:
        wt, ht = key.split("x", 1)
        w, h = int(wt), int(ht)
    else:
        raise SystemExit(f"render_cc: bad --grid '{spec}' "
                         f"(use WxH, or a preset: {', '.join(_GRID_PRESETS)})")
    return width or w, height or h


def _parse_channels(spec: str) -> int:
    key = spec.strip().lower()
    if key in _CHANNEL_PRESETS:
        return _CHANNEL_PRESETS[key]
    mask = 0
    for tok in key.split(","):
        tok = tok.strip()
        if tok not in _ROLE_NAMES:
            raise SystemExit(
                f"render_cc: bad --channels '{tok}' "
                f"(presets: {', '.join(_CHANNEL_PRESETS)}; "
                f"or a comma list of roles: {', '.join(_ROLE_NAMES)})")
        mask |= 1 << _ROLE_NAMES[tok]
    return mask


# --------------------------------------------------------------------------- #
# Output path defaulting
# --------------------------------------------------------------------------- #

_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_filename(name: str) -> str:
    cleaned = _UNSAFE_FILENAME.sub("_", name).strip(" .")
    return (cleaned or "output")[:120]


def _probe_title(url: str) -> Optional[str]:
    """Best-effort source title (for the default output filename); None on
    any failure — yt-dlp missing, network error, an extractor with no title."""
    if shutil.which("yt-dlp") is None:
        return None
    try:
        out = subprocess.run(
            ["yt-dlp", "--no-warnings", "--quiet", "--no-playlist",
             "--print", "%(title)s", url],
            capture_output=True, text=True, timeout=20)
        title = out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
        return title or None
    except Exception:
        return None


def _default_out(source: str, is_url: bool) -> Path:
    if not is_url:
        return Path(source).with_suffix(".ccmf")
    title = _probe_title(source)
    return Path.cwd() / f"{_sanitize_filename(title) if title else 'output'}.ccmf"


# --------------------------------------------------------------------------- #
# PTS-ordered merge: interleave finished video GOPs and audio chunks into one
# file as they're produced, without buffering the whole render in memory.
# --------------------------------------------------------------------------- #

async def _merge_write(out_f, queues: dict[str, "asyncio.Queue"]) -> dict[str, int]:
    heads: dict[str, tuple[int, bytes]] = {}
    exhausted: set[str] = set()
    counts = {name: 0 for name in queues}

    async def _refill(name: str) -> None:
        if name in exhausted:
            return
        item = await queues[name].get()
        if item is None:
            exhausted.add(name)
        else:
            heads[name] = item

    for name in queues:
        await _refill(name)
    while heads:
        name = min(heads, key=lambda n: heads[n][0])
        _pts, data = heads.pop(name)
        out_f.write(data)
        counts[name] += 1
        await _refill(name)
    return counts


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #

async def _render(args: argparse.Namespace) -> int:
    want_video = not args.no_video
    want_audio = not args.no_audio
    if not want_video and not want_audio:
        print("render_cc: nothing to render (both --no-video and --no-audio given).")
        return 1
    if args.start < 0:
        print("render_cc: --start must be >= 0.")
        return 1
    if args.duration is not None and args.duration <= 0:
        print("render_cc: --duration must be > 0.")
        return 1
    if args.end is not None and args.end <= args.start:
        print("render_cc: --end must be greater than --start.")
        return 1

    source = args.source
    is_url = not Path(source).is_file()
    if is_url and shutil.which("yt-dlp") is None:
        print(f"render_cc: '{source}' isn't a local file, and yt-dlp isn't on PATH.")
        return 1

    w, h = _resolve_grid(args.grid, args.width, args.height)
    channels_mask = _parse_channels(args.channels) if want_audio else 0
    codec_id = ccmf.CODEC_PCM8 if args.audio_codec == "pcm" else ccmf.CODEC_DFPWM
    roles = negotiate_channel_roles(channels_mask) if want_audio else []

    end = args.end
    if args.duration is not None:
        end = args.start + args.duration

    out_path = Path(args.out) if args.out else _default_out(source, is_url)

    # Where each producer reads from, and how it's trimmed:
    #  * genuinely local file  -> decode it directly; trim with an input-side
    #    ffmpeg seek (trim_start/trim_duration).
    #  * URL, streamable       -> decode from the yt-dlp pipe; trim via the
    #    yt-dlp fetch window (pipe_start/pipe_end), same as a live session.
    #  * URL needing a download (moov-at-end MP4, GIF) -> download exactly
    #    [start, end] to a temp file, then decode it untrimmed.
    tmpdir: Optional[str] = None
    source_path: Optional[str] = None
    pipe_start, pipe_end = 0.0, None
    trim_start, trim_duration = 0.0, None

    if is_url:
        is_live, ext = await probe_source_info(source)
        if is_live:
            print(f"render_cc: '{source}' is a live stream; render_cc only "
                 "converts on-demand (VOD) sources.")
            return 1
        need_download = want_video and (needs_download(ext) or (
            needs_seekable_source(ext) and await probe_moov_at_end(source)))
        if need_download:
            tmpdir = tempfile.mkdtemp(prefix="render_cc_")
            source_path = await download_source(source, tmpdir, args.start, end, want_audio)
            if not source_path:
                print("render_cc: failed to download the source.")
                shutil.rmtree(tmpdir, ignore_errors=True)
                return 1
        else:
            pipe_start, pipe_end = args.start, end
    else:
        source_path = str(Path(source).resolve())
        trim_start = args.start
        trim_duration = None if end is None else max(0.0, end - args.start)

    source_channels = 1
    if want_audio and channels_mask != ccmf.CAP_CHANNEL_MONO:
        if is_url:
            source_channels = await probe_audio_channels(source)
        else:
            source_channels = _ffprobe_channels(source_path) or _ASSUMED_CHANNELS

    timeline = SourceTimeline(
        [n for n, wanted in (("video", want_video), ("audio", want_audio)) if wanted],
        live=False)

    gop_samples = round(args.gop_seconds * ccmf.SAMPLE_RATE)
    chunk_samples = round(args.audio_chunk_seconds * ccmf.SAMPLE_RATE)

    video_q: asyncio.Queue = asyncio.Queue()
    audio_q: asyncio.Queue = asyncio.Queue()
    queues: dict[str, asyncio.Queue] = {}
    tasks: list[asyncio.Task] = []

    async def _run_video() -> None:
        agen = None
        try:
            agen = iter_video(source, w, h, args.fps,
                              start=pipe_start, end=pipe_end,
                              source_path=source_path, timeline=timeline,
                              adaptive=False, trim_start=trim_start,
                              trim_duration=trim_duration, gop_samples=gop_samples)
            async for pts, chunk in agen:
                await video_q.put((pts, chunk))
        except Exception:
            log.exception("render_cc: video pipeline error")
        finally:
            if agen is not None:
                await agen.aclose()
            await video_q.put(None)

    async def _run_audio() -> None:
        agen = None
        ev_loop = asyncio.get_running_loop()
        try:
            agen = iter_audio_roles(source, ccmf.SAMPLE_RATE, roles=roles,
                                    decode_channels=source_channels,
                                    start=pipe_start, end=pipe_end,
                                    source_path=source_path, timeline=timeline,
                                    trim_start=trim_start, trim_duration=trim_duration,
                                    chunk_samples=chunk_samples)
            async for pts, chunks in agen:
                encoded: dict[bytes, bytes] = {}
                for role, data in chunks.items():
                    wire = data
                    if codec_id == ccmf.CODEC_DFPWM:
                        wire = encoded.get(data)
                        if wire is None:
                            wire = await ev_loop.run_in_executor(None, dfpwm.encode, data)
                            encoded[data] = wire
                    payload = ccmf.audio_payload(codec_id, wire, channel=role)
                    await audio_q.put((pts, ccmf.chunk(pts, ccmf.TYPE_AUDIO, payload)))
        except Exception:
            log.exception("render_cc: audio pipeline error")
        finally:
            if agen is not None:
                await agen.aclose()
            await audio_q.put(None)

    if want_video:
        queues["video"] = video_q
        tasks.append(asyncio.create_task(_run_video()))
    if want_audio:
        queues["audio"] = audio_q
        tasks.append(asyncio.create_task(_run_audio()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    t0 = time.perf_counter()
    try:
        with open(tmp_out, "wb") as f:
            counts = await _merge_write(f, queues)
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    primary = counts.get("video", 0) if want_video else counts.get("audio", 0)
    if primary == 0:
        print(f"render_cc: failed to produce any {'video' if want_video else 'audio'} "
             "— check the source and grid/codec settings.")
        tmp_out.unlink(missing_ok=True)
        return 1
    if want_video and want_audio and counts.get("audio", 0) == 0:
        print("render_cc: warning — no audio chunks were produced; the file has video only.")

    tmp_out.replace(out_path)
    elapsed = time.perf_counter() - t0
    print(f"render_cc: wrote {out_path} "
         f"({counts.get('video', 0)} video GOP(s), {counts.get('audio', 0)} audio chunk(s)) "
         f"in {elapsed:.1f}s")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Convert a media file (or yt-dlp URL) into a .ccmf file, "
                     "at full transcode speed (no live adaptive pacing).")
    ap.add_argument("source", help="local media file path, or a yt-dlp-supported URL")
    ap.add_argument("-o", "--out",
                    help="output .ccmf path (default: <source>.ccmf, or "
                         "./<title>.ccmf for a URL)")
    ap.add_argument("--grid", default="terminal",
                    help=f"character grid: WxH, or a preset "
                         f"({', '.join(_GRID_PRESETS)}) (default: terminal)")
    ap.add_argument("--width", type=int, help="grid cell width (overrides --grid)")
    ap.add_argument("--height", type=int, help="grid cell height (overrides --grid)")
    ap.add_argument("--fps", type=int, default=24, help="output frame rate (default: 24)")
    ap.add_argument("--start", type=float, default=0.0,
                    help="seconds into the source to start at (default: 0)")
    end_grp = ap.add_mutually_exclusive_group()
    end_grp.add_argument("--end", type=float, help="seconds into the source to stop at")
    end_grp.add_argument("--duration", type=float, help="seconds to render, from --start")
    ap.add_argument("--no-video", action="store_true", help="skip the video stream")
    ap.add_argument("--no-audio", action="store_true", help="skip the audio stream")
    ap.add_argument("--audio-codec", choices=["pcm", "dfpwm"], default="pcm",
                    help="wire audio codec (default: pcm; dfpwm trades fidelity "
                         "for ~8x less space)")
    ap.add_argument("--channels", default="mono",
                    help=f"speaker layout: {', '.join(_CHANNEL_PRESETS)}, or a "
                         f"comma list of roles ({', '.join(_ROLE_NAMES)}) (default: mono)")
    ap.add_argument("--gop-seconds", type=float, default=GOP_SECONDS,
                    help=f"video GOP span in seconds (default: {GOP_SECONDS:g})")
    ap.add_argument("--audio-chunk-seconds", type=float, default=AUDIO_CHUNK_SECONDS,
                    help=f"audio chunk span in seconds (default: {AUDIO_CHUNK_SECONDS:g})")
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if not have_ffmpeg():
        print("render_cc: ffmpeg not found on PATH.")
        return 1
    try:
        return asyncio.run(_render(args))
    except KeyboardInterrupt:
        print("\nrender_cc: interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

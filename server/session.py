"""
StreamSession — server-side buffering, A/V sync, and pacing.

Two producers (video, audio) fill bounded TimedBuffers; each item is tagged with
a media-time PTS derived from its output position (frame i -> i/fps; audio byte
k -> k/rate) on one shared timeline.  A single scheduler advances a media clock
and releases items whose PTS is due, sending them over one WebSocket:

    video frame -> binary message  b"\\x01" + frame
    audio chunk -> binary message  b"\\x02" + pcm
    status      -> text message    "META ..." / "BUFFERING" / "PLAYING" / "ERROR ..."

Because both streams are released against the same clock, they stay in sync.

Underrun / hiccup policy:
  * VOD  — buffers backpressure the producer; on underrun the clock PAUSES and
           re-buffers, so every frame is shown (just delayed).
  * LIVE — buffers drop their oldest item when full; when the clock falls behind
           the buffered head it SKIPS forward, staying near the live edge.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import shutil
import tempfile
from typing import Optional

from transcoder import (
    AUDIO_READ_BYTES,
    SAMPLES_PER_BYTE,
    download_source,
    iter_audio,
    iter_video,
    needs_seekable_source,
    parse_timestamp,
    probe_source_info,
)

log = logging.getLogger("livecc")

OP_VIDEO = b"\x01"
OP_AUDIO = b"\x02"

# Buffering parameters (seconds of media).
VOD_PREBUFFER = 2.0
VOD_MAX_BUFFER = 15.0
LIVE_PREBUFFER = 1.0
LIVE_MAX_BUFFER = 4.0

# Release audio this far ahead of the playback clock so the CC speaker buffer
# stays full (the speaker plays at 48 kHz, so playback stays at realtime — the
# lead is pure jitter slack and does not desync from video, which is sent at
# exactly the clock).
AUDIO_LEAD = 2.0

_AUDIO_CHUNK_SECONDS = AUDIO_READ_BYTES * SAMPLES_PER_BYTE / 48000.0  # ~1.0 s


class TimedBuffer:
    """A FIFO of (pts, data) with a max depth in items.

    drop_oldest=False (VOD): put() blocks when full -> backpressures the producer.
    drop_oldest=True  (live): put() discards the oldest item when full -> skip.
    """

    def __init__(self, max_items: int, drop_oldest: bool) -> None:
        self._dq: collections.deque[tuple[float, bytes]] = collections.deque()
        self._max = max(1, max_items)
        self._drop = drop_oldest
        self._not_full = asyncio.Event()
        self._not_full.set()
        self._closed = False

    async def put(self, pts: float, data: bytes) -> None:
        if self._drop:
            self._dq.append((pts, data))
            while len(self._dq) > self._max:
                self._dq.popleft()
        else:
            while len(self._dq) >= self._max and not self._closed:
                self._not_full.clear()
                await self._not_full.wait()
            self._dq.append((pts, data))

    def pop_due(self, media_now: float) -> list[tuple[float, bytes]]:
        out = []
        while self._dq and self._dq[0][0] <= media_now:
            out.append(self._dq.popleft())
        if len(self._dq) < self._max:
            self._not_full.set()
        return out

    def head_pts(self) -> Optional[float]:
        return self._dq[0][0] if self._dq else None

    def tail_pts(self) -> Optional[float]:
        return self._dq[-1][0] if self._dq else None

    def seconds(self) -> float:
        if len(self._dq) < 2:
            return 0.0
        return self._dq[-1][0] - self._dq[0][0]

    def empty(self) -> bool:
        return not self._dq

    def close(self) -> None:
        self._closed = True
        self._not_full.set()


class StreamSession:
    def __init__(self, url: str, w: int, h: int, fps: int, want_audio: bool,
                 start: str = "", end: str = "", loop: bool = False,
                 rate: int = 48000) -> None:
        self.url = url
        self.w, self.h, self.fps = w, h, fps
        self.want_audio = want_audio
        self.rate = rate

        self.start = parse_timestamp(start)             # seconds, 0 if absent
        _end = parse_timestamp(end)
        self.end: Optional[int] = _end if _end > 0 else None
        self.loop = loop

        self.is_live = False
        self.prebuffer = VOD_PREBUFFER
        self._start_eff = 0                             # effective offset (VOD only)
        self._end_eff: Optional[int] = None
        self._source_path: Optional[str] = None         # temp file (loop / seekable)
        self._tmpdir: Optional[str] = None
        self._need_seekable = False                      # MP4-family VOD -> download

        self.video_buf: TimedBuffer | None = None
        self.audio_buf: TimedBuffer | None = None

        self.video_done = asyncio.Event()
        self.audio_done = asyncio.Event()
        self.video_sent = 0
        self.audio_sent = 0

    # ----- setup -----------------------------------------------------------

    def _setup_buffers(self) -> None:
        if self.is_live:
            self.prebuffer = LIVE_PREBUFFER
            max_s, drop = LIVE_MAX_BUFFER, True
        else:
            self.prebuffer = VOD_PREBUFFER
            max_s, drop = VOD_MAX_BUFFER, False
        self.video_buf = TimedBuffer(int(max_s * self.fps), drop)
        self.audio_buf = TimedBuffer(int(max_s / _AUDIO_CHUNK_SECONDS), drop)

    # ----- producers -------------------------------------------------------

    async def _produce_video(self) -> None:
        # Keep an explicit handle so we can aclose() it: when this task is
        # cancelled mid-buffer.put() the generator is suspended at its yield,
        # and only aclose() runs its finally (which kills yt-dlp/ffmpeg) — GC is
        # not prompt enough and would leave the pipeline running.
        agen = iter_video(self.url, self.w, self.h, self.fps,
                          start=self._start_eff, end=self._end_eff,
                          source_path=self._source_path, loop=self.loop)
        i = 0
        try:
            async for frame in agen:
                await self.video_buf.put(i / self.fps, frame)
                i += 1
        finally:
            await agen.aclose()
            self.video_buf.close()
            self.video_done.set()

    async def _produce_audio(self) -> None:
        agen = iter_audio(self.url, self.rate,
                          start=self._start_eff, end=self._end_eff,
                          source_path=self._source_path, loop=self.loop)
        samples = 0
        try:
            async for chunk in agen:
                await self.audio_buf.put(samples / self.rate, chunk)
                samples += len(chunk) * SAMPLES_PER_BYTE   # DFPWM: 8 samples/byte
        finally:
            await agen.aclose()
            self.audio_buf.close()
            self.audio_done.set()

    # ----- scheduler helpers ----------------------------------------------

    def _producing_video(self) -> bool:
        return not self.video_done.is_set()

    def _producing_audio(self) -> bool:
        return self.want_audio and not self.audio_done.is_set()

    def _oldest_pts(self) -> Optional[float]:
        cands = [p for p in (self.video_buf.head_pts(),
                             self.audio_buf.head_pts() if self.want_audio else None)
                 if p is not None]
        return min(cands) if cands else None

    def _all_done_and_empty(self) -> bool:
        v = self.video_done.is_set() and self.video_buf.empty()
        a = (not self.want_audio) or (self.audio_done.is_set() and self.audio_buf.empty())
        return v and a

    async def _prebuffer(self) -> None:
        while not self.video_done.is_set() and self.video_buf.seconds() < self.prebuffer:
            await asyncio.sleep(0.05)

    async def _release(self, ws, media_now: float) -> bool:
        sent = False
        # Audio leads so the speaker stays buffered; video is sent exactly on time.
        for _pts, data in self.audio_buf.pop_due(media_now + AUDIO_LEAD):
            await ws.send_bytes(OP_AUDIO + data)
            self.audio_sent += 1
            sent = True
        for _pts, data in self.video_buf.pop_due(media_now):
            await ws.send_bytes(OP_VIDEO + data)
            self.video_sent += 1
            sent = True
        return sent

    # ----- main loop -------------------------------------------------------

    async def run(self, ws) -> None:
        loop = asyncio.get_running_loop()
        self.is_live, ext = await probe_source_info(self.url)
        # start/end only make sense for VOD; ignore them for live streams.
        self._start_eff = 0 if self.is_live else self.start
        self._end_eff = None if self.is_live else self.end
        self._setup_buffers()

        # A VOD whose container can hide its index at the end (MP4 family) can't be
        # demuxed from a non-seekable pipe, so decode it from a downloaded file.
        # Live never hits this (segmented), and --loop already downloads anyway.
        self._need_seekable = not self.is_live and needs_seekable_source(ext)

        await ws.send_text(f"META {self.w} {self.h} {self.fps}")
        log.info("session: %s is_live=%s ext=%s audio=%s start=%ss end=%s loop=%s seekable=%s",
                 self.url, self.is_live, ext, self.want_audio,
                 self._start_eff, self._end_eff, self.loop and not self.is_live,
                 self._need_seekable)

        tasks: list[asyncio.Task] = []
        try:
            await ws.send_text("BUFFERING")

            # Download the [start,end] section to a seekable temp file when we
            # either need to replay it (--loop) or can't pipe-stream the container
            # (MP4 moov-at-end).  Then ffmpeg decodes the file; --loop also replays.
            if (self.loop or self._need_seekable) and not self.is_live:
                self._tmpdir = tempfile.mkdtemp(prefix="livecc_")
                self._source_path = await download_source(
                    self.url, self._tmpdir, self._start_eff, self._end_eff,
                    self.want_audio)
                if not self._source_path:
                    await ws.send_text(
                        "ERROR Failed to prepare source. Check the server logs.")
                    return

            tasks.append(asyncio.create_task(self._produce_video()))
            if self.want_audio:
                tasks.append(asyncio.create_task(self._produce_audio()))

            await self._prebuffer()
            await ws.send_text("PLAYING")

            t0 = loop.time()
            origin = self._oldest_pts() or 0.0   # media time of the first item

            while True:
                media_now = (loop.time() - t0) + origin
                sent = await self._release(ws, media_now)

                if self._all_done_and_empty():
                    break

                if self.is_live:
                    head = self._oldest_pts()
                    if head is None:
                        await asyncio.sleep(0.02)          # gap: wait for data
                    elif head - media_now > self.prebuffer:
                        origin, t0 = head, loop.time()      # behind -> skip to head
                    elif not sent:
                        await asyncio.sleep(0.005)
                else:  # VOD
                    if self.video_buf.empty() and self._producing_video():
                        await ws.send_text("BUFFERING")     # underrun -> pause
                        await self._prebuffer()
                        await ws.send_text("PLAYING")
                        origin, t0 = media_now, loop.time()
                    elif not sent:
                        await asyncio.sleep(0.005)

            if self.video_sent == 0:
                await ws.send_text("ERROR Failed to load video. Check the server logs.")
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._tmpdir:
                shutil.rmtree(self._tmpdir, ignore_errors=True)

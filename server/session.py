"""
StreamSession — server-side buffering, A/V sync, and pacing.

Two producers (video, audio) fill bounded TimedBuffers; each item is a finished
CCMF chunk tagged with a media-time PTS derived from its output position (video
GOP -> its first frame's index/fps; audio byte k -> k/rate) on one shared
timeline.  A single scheduler advances a media clock and releases chunks whose
PTS is due, writing them to one sink (the WebSocket, or a sync-group fanout):

    video chunk  -> binary message  CCMF chunk type 0 (one self-contained GOP)
    audio chunk  -> binary message  CCMF chunk type 1 (PCM or DFPWM samples)
    status       -> binary message  CCMF control frame STATUS / ERROR / END

Because both streams are released against the same clock, they stay in sync.
Audio is released AUDIO_LEAD early (keeps the CC speaker buffer full); video
GOPs are released VIDEO_LEAD early (the client paces the frames inside each
chunk by PTS, so the lead is jitter slack, not a sync shift).

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

import ccmf
from transcoder import (
    AUDIO_CHUNK_SECONDS,
    GOP_SECONDS,
    PCM,
    download_source,
    iter_audio,
    iter_video,
    needs_download,
    needs_seekable_source,
    probe_moov_at_end,
    probe_source_info,
)

log = logging.getLogger("livecc")

# Buffering parameters (seconds of media).
VOD_PREBUFFER = 2.0
VOD_MAX_BUFFER = 15.0
LIVE_PREBUFFER = 1.0
LIVE_MAX_BUFFER = 4.0

# Release audio this far ahead of the playback clock so the CC speaker buffer
# stays full (the speaker plays at 48 kHz, so playback stays at realtime — the
# lead is pure jitter slack and does not desync from video).
AUDIO_LEAD = 2.0

# Release video GOP chunks this far ahead of the clock.  The client renders each
# frame at its own PTS, so an early chunk just sits in its queue — this only
# hides network/scheduler jitter at GOP boundaries.
VIDEO_LEAD = 0.5

_AUDIO_CHUNK_SECONDS = AUDIO_CHUNK_SECONDS   # ~0.1 s, same for both codecs


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
    """One production: a source URL transcoded and paced to one sink.

    `start`/`end` are seconds into the source (the ROOM message carries ms; the
    caller converts).  `audio_codec` is the transcoder.AudioCodec negotiated
    from the client's CAPS (PCM preferred, DFPWM fallback).
    """

    def __init__(self, url: str, w: int, h: int, fps: int, want_audio: bool,
                 want_video: bool = True, start: float = 0,
                 end: Optional[float] = None, loop: bool = False,
                 rate: int = 48000, audio_codec=PCM) -> None:
        self.url = url
        self.w, self.h, self.fps = w, h, fps
        self.want_audio = want_audio
        self.want_video = want_video
        self.rate = rate
        self.codec = audio_codec
        self._codec_id = ccmf.CODEC_PCM8 if audio_codec.name == "pcm" \
            else ccmf.CODEC_DFPWM

        self.start = max(0.0, float(start))
        self.end: Optional[float] = float(end) if end else None
        self.loop = loop

        self.is_live = False
        self.prebuffer = VOD_PREBUFFER
        self._start_eff = 0.0                           # effective offset (VOD only)
        self._end_eff: Optional[float] = None
        self._source_path: Optional[str] = None         # temp file (loop / seekable)
        self._tmpdir: Optional[str] = None
        self._need_download = False                      # decode from a downloaded file

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
        # Video items are whole GOPs (~GOP_SECONDS each), audio items ~0.1 s.
        self.video_buf = TimedBuffer(int(max_s / GOP_SECONDS) + 1, drop)
        self.audio_buf = TimedBuffer(int(max_s / _AUDIO_CHUNK_SECONDS), drop)

    # ----- producers -------------------------------------------------------

    async def _produce_video(self) -> None:
        # Keep an explicit handle so we can aclose() it: when this task is
        # cancelled mid-buffer.put() the generator is suspended at its yield,
        # and only aclose() runs its finally (which kills yt-dlp/ffmpeg) — GC is
        # not prompt enough and would leave the pipeline running.
        #
        # Everything is inside try/finally so video_done is ALWAYS set, even if the
        # generator can't be constructed or dies unexpectedly.  Otherwise the
        # scheduler would wait forever on a producer that will never report done.
        # CancelledError (BaseException) still propagates; only real errors are
        # swallowed-and-logged, degrading to "no frames" -> ERROR downstream.
        agen = None
        try:
            agen = iter_video(self.url, self.w, self.h, self.fps,
                              start=self._start_eff, end=self._end_eff,
                              source_path=self._source_path, loop=self.loop)
            # iter_video yields finished CCMF GOP chunks tagged with their first
            # frame's PTS in samples; it may skip source frames to keep up
            # (adaptive pacing) — so trust its pts, don't count.
            async for pts, chunk in agen:
                await self.video_buf.put(pts / ccmf.SAMPLE_RATE, chunk)
        except Exception:
            log.exception("video producer failed")
        finally:
            if agen is not None:
                await agen.aclose()
            self.video_buf.close()
            self.video_done.set()

    async def _produce_audio(self) -> None:
        # Same guarantee as video: audio_done is always set so a failed audio
        # producer degrades to a silent stream rather than hanging the scheduler.
        agen = None
        samples = 0
        try:
            agen = iter_audio(self.url, self.rate, self.codec,
                              start=self._start_eff, end=self._end_eff,
                              source_path=self._source_path, loop=self.loop)
            async for data in agen:
                # Wrap here, where the running sample counter (= the chunk's PTS,
                # spec §4.6) is known; the buffer then holds wire-ready bytes.
                chunk = ccmf.chunk(samples, ccmf.TYPE_AUDIO,
                                   ccmf.audio_payload(self._codec_id, data))
                await self.audio_buf.put(samples / self.rate, chunk)
                samples += len(data) * self.codec.samples_per_byte
        except Exception:
            log.exception("audio producer failed")
        finally:
            if agen is not None:
                await agen.aclose()
            self.audio_buf.close()
            self.audio_done.set()

    # ----- scheduler helpers ----------------------------------------------

    def _producing_video(self) -> bool:
        return self.want_video and not self.video_done.is_set()

    def _producing_audio(self) -> bool:
        return self.want_audio and not self.audio_done.is_set()

    # The "primary" stream drives prebuffer/underrun gating: video when present,
    # else audio (audio-only / --no-video).  Both streams are still released by PTS
    # against the shared clock — this only picks what startup/underrun keys off.
    def _primary_buf(self) -> TimedBuffer:
        return self.video_buf if self.want_video else self.audio_buf

    def _primary_done(self) -> asyncio.Event:
        return self.video_done if self.want_video else self.audio_done

    def _primary_producing(self) -> bool:
        return self._producing_video() if self.want_video else self._producing_audio()

    def _oldest_pts(self) -> Optional[float]:
        cands = [p for p in (self.video_buf.head_pts() if self.want_video else None,
                             self.audio_buf.head_pts() if self.want_audio else None)
                 if p is not None]
        return min(cands) if cands else None

    def _all_done_and_empty(self) -> bool:
        v = (not self.want_video) or (self.video_done.is_set() and self.video_buf.empty())
        a = (not self.want_audio) or (self.audio_done.is_set() and self.audio_buf.empty())
        return v and a

    async def _prebuffer(self) -> None:
        buf, done = self._primary_buf(), self._primary_done()
        while not done.is_set() and buf.seconds() < self.prebuffer:
            await asyncio.sleep(0.05)

    async def _send_status(self, ws, status: int) -> None:
        await ws.send_bytes(ccmf.control(ccmf.OP_STATUS, bytes([status])))

    async def _send_error(self, ws, message: str) -> None:
        # ERROR bodies are client-facing: keep them sanitized (no internal
        # detail; details go to the server log), per spec §7.
        await ws.send_bytes(ccmf.control(ccmf.OP_ERROR, message.encode("utf-8")))

    async def _release(self, ws, media_now: float) -> bool:
        sent = False
        # Both buffers hold wire-ready CCMF chunks; audio leads so the speaker
        # stays buffered, video leads a little so the client-side PTS pacing
        # never starves at a GOP boundary.
        for _pts, data in self.audio_buf.pop_due(media_now + AUDIO_LEAD):
            await ws.send_bytes(data)
            self.audio_sent += 1
            sent = True
        for _pts, data in self.video_buf.pop_due(media_now + VIDEO_LEAD):
            await ws.send_bytes(data)
            self.video_sent += 1
            sent = True
        return sent

    # ----- main loop -------------------------------------------------------

    async def run(self, ws) -> None:
        if not self.want_video and not self.want_audio:
            await self._send_error(ws, "Nothing to play (audio and video both disabled).")
            return
        loop = asyncio.get_running_loop()
        self.is_live, ext = await probe_source_info(self.url)
        # start/end only make sense for VOD; ignore them for live streams.
        self._start_eff = 0.0 if self.is_live else self.start
        self._end_eff = None if self.is_live else self.end
        self._setup_buffers()

        # Decide whether to decode from a downloaded file instead of the pipe (live
        # is segmented; --loop downloads regardless):
        #   * GIF -> always download (the server's ffmpeg can't demux a GIF from a
        #     pipe; it plays once from the file, unless --loop was also given);
        #   * MP4 family -> probe, download only a genuine moov-at-end file.
        self._need_download = False
        if not self.is_live and not self.loop:
            if needs_download(ext):
                self._need_download = True
            elif needs_seekable_source(ext):
                self._need_download = await probe_moov_at_end(self.url)

        log.info("session: %s is_live=%s ext=%s audio=%s start=%ss end=%s loop=%s download=%s",
                 self.url, self.is_live, ext, self.want_audio,
                 self._start_eff, self._end_eff, self.loop and not self.is_live,
                 self._need_download)

        tasks: list[asyncio.Task] = []
        try:
            await self._send_status(ws, ccmf.STATUS_BUFFERING)

            # Download the [start,end] section to a temp file when we need to replay
            # it (--loop) or can't pipe-stream it (GIF / MP4 moov-at-end).  Then
            # ffmpeg decodes the seekable file; --loop also replays it.
            if (self.loop or self._need_download) and not self.is_live:
                self._tmpdir = tempfile.mkdtemp(prefix="livecc_")
                self._source_path = await download_source(
                    self.url, self._tmpdir, self._start_eff, self._end_eff,
                    self.want_audio)
                if not self._source_path:
                    await self._send_error(
                        ws, "Failed to prepare source. Check the server logs.")
                    return

            if self.want_video:
                tasks.append(asyncio.create_task(self._produce_video()))
            if self.want_audio:
                tasks.append(asyncio.create_task(self._produce_audio()))

            await self._prebuffer()
            await self._send_status(ws, ccmf.STATUS_PLAYING)

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
                    elif head - media_now > self.prebuffer + GOP_SECONDS:
                        # Behind the live edge -> skip to head.  The GOP_SECONDS
                        # margin matters: a video chunk's PTS runs up to one GOP
                        # ahead of the moment it is produced (it flushes when
                        # the NEXT GOP's first frame arrives), so head sits
                        # ~GOP_SECONDS ahead of the clock at normal cadence —
                        # without the margin this fired on every jitter, and
                        # the constant origin jumps made release timing (and
                        # the client's pacing) erratic.
                        origin, t0 = head, loop.time()
                    elif not sent:
                        await asyncio.sleep(0.005)
                else:  # VOD
                    if self._primary_buf().empty() and self._primary_producing():
                        await self._send_status(ws, ccmf.STATUS_BUFFERING)
                        await self._prebuffer()             # underrun -> pause
                        await self._send_status(ws, ccmf.STATUS_PLAYING)
                        origin, t0 = media_now, loop.time()
                    elif not sent:
                        await asyncio.sleep(0.005)

            if (self.video_sent if self.want_video else self.audio_sent) == 0:
                what = "video" if self.want_video else "audio"
                await self._send_error(ws, f"Failed to load {what}. Check the server logs.")
            else:
                await ws.send_bytes(ccmf.control(ccmf.OP_END))
        except Exception:
            # Catch-all for anything not handled deeper (producers self-contain
            # their errors): log the traceback and tell the client something broke
            # rather than dropping the socket silently.  CancelledError (client
            # disconnect) is a BaseException, so it isn't caught here and tears
            # down normally.
            log.exception("session: unexpected error")
            try:
                await self._send_error(ws, "Internal server error. Check the server logs.")
            except Exception:
                pass   # socket may already be gone; don't mask the original error
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._tmpdir:
                shutil.rmtree(self._tmpdir, ignore_errors=True)

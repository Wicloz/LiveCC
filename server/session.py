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
    ROLE_SOURCE_CHANNEL,
    download_source,
    iter_audio,
    iter_audio_channels,
    iter_video,
    needs_download,
    needs_seekable_source,
    negotiate_channel_roles,
    probe_audio_channels,
    probe_moov_at_end,
    probe_source_info,
)

# Bound how long reconfigure_channels() waits for a session's initial setup
# (run()) before giving up -- only matters if run() never gets scheduled or
# is stuck; a normal session becomes ready within milliseconds.
_READY_TIMEOUT = 30.0

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

    `caps_channels` seeds `self.requested_channels` -- for a private session
    that's simply the one client's CAPS `channels` bitmask and never changes;
    for a sync room (`dynamic_channels=True`) it's the union of every current
    subscriber's request, which main._SyncGroup updates and re-applies (via
    reconfigure_channels) each time a client joins or leaves. self.channel_roles
    (the roles actually being produced right now) is derived from
    self.requested_channels and the source's real channel count
    (transcoder.negotiate_channel_roles) and can change over the session's
    lifetime for a sync room -- newly-wanted roles start producing without
    disturbing ones already flowing; roles nobody wants anymore stop being
    served (their buffer closes) though their underlying decode pipeline (see
    _produce_audio_positional) keeps running rather than restart later.
    """

    def __init__(self, url: str, w: int, h: int, fps: int, want_audio: bool,
                 want_video: bool = True, start: float = 0,
                 end: Optional[float] = None, loop: bool = False,
                 rate: int = 48000, audio_codec=PCM,
                 caps_channels: int = ccmf.CAP_CHANNEL_MONO,
                 dynamic_channels: bool = False) -> None:
        self.url = url
        self.w, self.h, self.fps = w, h, fps
        self.want_audio = want_audio
        self.want_video = want_video
        self.rate = rate
        self.codec = audio_codec
        self.requested_channels = caps_channels
        self.dynamic_channels = dynamic_channels
        self.source_channels = 1                         # probed in run() if needed
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
        self._buf_max_s = VOD_MAX_BUFFER                 # set for real in _setup_buffers
        self._buf_drop = False

        self.video_buf: TimedBuffer | None = None
        self.channel_roles: list[int] = []               # populated by _apply_channel_target
        self.audio_bufs: dict[int, TimedBuffer] = {}      # role -> buffer

        self.video_done = asyncio.Event()
        self.audio_done: dict[int, asyncio.Event] = {}    # role -> done
        self.video_sent = 0
        self.audio_sent = 0

        # Set once run()'s initial setup (probe + _apply_channel_target) has
        # happened, so a racing reconfigure_channels() call (a sync room's
        # second subscriber joining before the first's session finished
        # starting up) waits instead of touching half-initialised state.
        self._ready = asyncio.Event()
        self._mono_task: Optional[asyncio.Task] = None
        self._positional_task: Optional[asyncio.Task] = None

    # ----- setup -----------------------------------------------------------

    def _setup_buffers(self) -> None:
        if self.is_live:
            self.prebuffer = LIVE_PREBUFFER
            self._buf_max_s, self._buf_drop = LIVE_MAX_BUFFER, True
        else:
            self.prebuffer = VOD_PREBUFFER
            self._buf_max_s, self._buf_drop = VOD_MAX_BUFFER, False
        # Video items are whole GOPs (~GOP_SECONDS each); audio buffers are
        # created on demand in _apply_channel_target (audio items ~0.1 s).
        self.video_buf = TimedBuffer(int(self._buf_max_s / GOP_SECONDS) + 1, self._buf_drop)

    def _apply_channel_target(self) -> None:
        """Diff self.channel_roles against what self.requested_channels +
        self.source_channels now call for, and apply the delta: create a
        buffer for each newly-wanted role (starting the mono and/or positional
        producer the first time either is needed), close the buffer for each
        role nobody wants anymore. Synchronous and await-free by construction
        (buffer/task creation and TimedBuffer.close()/Event.set() never
        suspend), so concurrent callers can't interleave mid-update.

        A no-op when audio is off for this session: a sync room's subscriber
        churn (main._SyncGroup) calls reconfigure_channels() unconditionally,
        without knowing whether this particular room even wants audio.
        """
        if not self.want_audio:
            return
        target = set(negotiate_channel_roles(self.requested_channels, self.source_channels))
        current = set(self.channel_roles)
        added, removed = target - current, current - target
        if not added and not removed:
            return

        for role in added:
            self.audio_bufs[role] = TimedBuffer(int(self._buf_max_s / _AUDIO_CHUNK_SECONDS),
                                                self._buf_drop)
            self.audio_done[role] = asyncio.Event()
        if 0 in added and self._mono_task is None:
            self._mono_task = asyncio.create_task(self._produce_audio_mono())
        if (added - {0}) and self._positional_task is None:
            self._positional_task = asyncio.create_task(self._produce_audio_positional())

        for role in removed:
            self.audio_bufs.pop(role).close()
            self.audio_done.pop(role).set()

        self.channel_roles = sorted(target)
        log.info("session: %s audio roles now %s (added=%s removed=%s)",
                 self.url, self.channel_roles, sorted(added), sorted(removed))

    async def reconfigure_channels(self) -> None:
        """Public entry point for an external caller (main._SyncGroup, when a
        sync room's subscriber set changes) to re-derive and apply the target
        role set from the current self.requested_channels. Waits for run()'s
        initial setup first so it never touches the session before it's ready."""
        try:
            await asyncio.wait_for(self._ready.wait(), _READY_TIMEOUT)
        except asyncio.TimeoutError:
            return
        self._apply_channel_target()

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

    async def _produce_audio_mono(self) -> None:
        # Role 0's independent downmix pipeline -- started the first time mono
        # is wanted (_apply_channel_target), kept running for the rest of the
        # session even if mono later becomes momentarily unwanted (a sync room
        # might ask for it again; restarting a live decode mid-stream is worse
        # than idling). A chunk is simply dropped (not buffered/sent) whenever
        # role 0 currently has no buffer -- see the `buf is not None` check --
        # which is exactly what "obsolete channels no longer served" means here.
        agen = None
        samples = 0
        try:
            agen = iter_audio(self.url, self.rate, self.codec,
                              start=self._start_eff, end=self._end_eff,
                              source_path=self._source_path, loop=self.loop,
                              source_channel=None)
            async for data in agen:
                buf = self.audio_bufs.get(0)
                if buf is not None:
                    chunk = ccmf.chunk(samples, ccmf.TYPE_AUDIO,
                                       ccmf.audio_payload(self._codec_id, data, channel=0))
                    await buf.put(samples / self.rate, chunk)
                samples += len(data) * self.codec.samples_per_byte
        except Exception:
            log.exception("audio producer failed (role=mono)")
        finally:
            if agen is not None:
                await agen.aclose()
            # close()/set() only -- NOT pop(): self.channel_roles still lists
            # role 0 (only _apply_channel_target's "removed" path retires a
            # role from there), so the dict entry must stay for the scheduler's
            # self.audio_done[r]/self.audio_bufs[r] lookups to keep working.
            if 0 in self.audio_bufs:
                self.audio_bufs[0].close()
            if 0 in self.audio_done:
                self.audio_done[0].set()
            self._mono_task = None

    async def _produce_audio_positional(self) -> None:
        # One shared decode covers every positional role the source can
        # possibly supply (spec §4.6), decoded once at full source width
        # regardless of which subset is CURRENTLY wanted -- so a later-joining
        # sync-room client asking for a not-yet-served channel doesn't need a
        # pipeline restart (which would briefly hiccup a live source). Started
        # the first time ANY positional role is wanted; like mono, kept running
        # for the rest of the session once started. iter_audio_channels
        # guarantees every role's sample 0 is the same source sample -- see its
        # docstring for why independent per-role pipelines would let stereo
        # drift apart on a live source.
        agen = None
        samples = 0
        roles = sorted(r for r, idx in ROLE_SOURCE_CHANNEL.items() if idx < self.source_channels)
        try:
            agen = iter_audio_channels(self.url, self.rate, self.codec, roles,
                                       start=self._start_eff, end=self._end_eff,
                                       source_path=self._source_path, loop=self.loop)
            async for chunks in agen:
                pts = samples / self.rate
                for role, data in chunks.items():
                    buf = self.audio_bufs.get(role)
                    if buf is not None:
                        chunk = ccmf.chunk(samples, ccmf.TYPE_AUDIO,
                                           ccmf.audio_payload(self._codec_id, data, channel=role))
                        await buf.put(pts, chunk)
                samples += len(next(iter(chunks.values()))) * self.codec.samples_per_byte
        except Exception:
            log.exception("audio producer failed (positional, roles=%s)", roles)
        finally:
            if agen is not None:
                await agen.aclose()
            # close()/set() only -- see _produce_audio_mono's finally for why
            # these dict entries must NOT be popped here.
            for role in roles:
                if role in self.audio_bufs:
                    self.audio_bufs[role].close()
                if role in self.audio_done:
                    self.audio_done[role].set()
            self._positional_task = None

    # ----- scheduler helpers ----------------------------------------------

    def _producing_video(self) -> bool:
        return self.want_video and not self.video_done.is_set()

    def _producing_audio(self) -> bool:
        return self.want_audio and not all(e.is_set() for e in self.audio_done.values())

    # The "primary" stream drives prebuffer/underrun gating: video when present,
    # else audio (audio-only / --no-video).  Both streams are still released by PTS
    # against the shared clock — this only picks what startup/underrun keys off.
    # With multiple audio roles, the first one stands in for all of them (their
    # buffers fill in lockstep, being fed by at most two shared pipelines --
    # mono and positional). channel_roles[0] is always role 0 (mono) whenever
    # want_audio and at least one subscriber is present: a client can only get
    # want_audio=True with the mandatory mono bit set (main._negotiate), so the
    # union driving negotiate_channel_roles always includes it.
    def _primary_buf(self) -> TimedBuffer:
        return self.video_buf if self.want_video else self.audio_bufs[self.channel_roles[0]]

    def _primary_done(self) -> asyncio.Event:
        return self.video_done if self.want_video else self.audio_done[self.channel_roles[0]]

    def _primary_producing(self) -> bool:
        return self._producing_video() if self.want_video else self._producing_audio()

    def _oldest_pts(self) -> Optional[float]:
        cands = []
        if self.want_video:
            p = self.video_buf.head_pts()
            if p is not None:
                cands.append(p)
        if self.want_audio:
            for buf in self.audio_bufs.values():
                p = buf.head_pts()
                if p is not None:
                    cands.append(p)
        return min(cands) if cands else None

    def _all_done_and_empty(self) -> bool:
        v = (not self.want_video) or (self.video_done.is_set() and self.video_buf.empty())
        a = (not self.want_audio) or all(
            self.audio_done[r].is_set() and self.audio_bufs[r].empty()
            for r in self.channel_roles)
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
        # All buffers hold wire-ready CCMF chunks; audio leads so the speaker
        # stays buffered, video leads a little so the client-side PTS pacing
        # never starves at a GOP boundary.  Each audio role is an independent
        # buffer/pipeline, so they're drained in whatever order dict iteration
        # gives -- release order across roles doesn't matter, only PTS does.
        for buf in self.audio_bufs.values():
            for _pts, data in buf.pop_due(media_now + AUDIO_LEAD):
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
            self._ready.set()   # unblock any racing reconfigure_channels() caller
            return
        loop = asyncio.get_running_loop()
        self.is_live, ext = await probe_source_info(self.url)
        # start/end only make sense for VOD; ignore them for live streams.
        self._start_eff = 0.0 if self.is_live else self.start
        self._end_eff = None if self.is_live else self.end

        # The channel-count probe only runs when it could actually change the
        # outcome: a dynamic (sync-room) session's future subscribers are
        # unpredictable, so it always probes; a private session's request is
        # fixed for its whole lifetime, so the common mono-only client (no
        # CAPS channels beyond the mandatory bit) skips it.
        if self.want_audio and (self.dynamic_channels
                                or self.requested_channels != ccmf.CAP_CHANNEL_MONO):
            self.source_channels = await probe_audio_channels(self.url)

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

        log.info("session: %s is_live=%s ext=%s audio=%s "
                 "start=%ss end=%s loop=%s download=%s",
                 self.url, self.is_live, ext, self.want_audio,
                 self._start_eff, self._end_eff, self.loop and not self.is_live,
                 self._need_download)

        tasks: list[asyncio.Task] = []
        try:
            await self._send_status(ws, ccmf.STATUS_BUFFERING)

            # Download the [start,end] section to a temp file when we need to replay
            # it (--loop) or can't pipe-stream it (GIF / MP4 moov-at-end).  Then
            # ffmpeg decodes the seekable file; --loop also replays it.  This MUST
            # happen before starting anything that reads self._source_path (the
            # video producer, and _apply_channel_target's audio pipelines just
            # below) -- otherwise they'd start against the pipe/None and never
            # pick up the downloaded file.
            if (self.loop or self._need_download) and not self.is_live:
                self._tmpdir = tempfile.mkdtemp(prefix="livecc_")
                self._source_path = await download_source(
                    self.url, self._tmpdir, self._start_eff, self._end_eff,
                    self.want_audio)
                if not self._source_path:
                    await self._send_error(
                        ws, "Failed to prepare source. Check the server logs.")
                    self._ready.set()   # unblock any racing reconfigure_channels() caller
                    return

            if self.want_video:
                tasks.append(asyncio.create_task(self._produce_video()))
            self._apply_channel_target()   # no-op if not self.want_audio
            self._ready.set()

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
            self._ready.set()   # safety net: unblock a racing caller even on an
                                # unexpected exception before the normal set() above
            audio_tasks = [t for t in (self._mono_task, self._positional_task)
                          if t is not None]
            for t in tasks + audio_tasks:
                t.cancel()
            await asyncio.gather(*tasks, *audio_tasks, return_exceptions=True)
            if self._tmpdir:
                shutil.rmtree(self._tmpdir, ignore_errors=True)

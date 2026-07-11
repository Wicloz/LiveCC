"""
StreamSession — server-side buffering, pacing, and the playback clock.

Producers fill bounded TimedBuffers; each item is a finished CCMF chunk whose
PTS lives on ONE shared session timeline: transcoder.SourceTimeline reconciles
the separately-fetched video and audio pipelines onto it, and every audio
channel role is cut from a single decode pass (transcoder.iter_audio_roles),
so nothing the server emits can be mutually skewed at the source.

A single scheduler advances a media clock and releases chunks whose PTS is
due, SEND_LEAD early, to one sink (the WebSocket, or a sync-group fanout):

    video chunk  -> binary message  CCMF chunk type 0 (one self-contained GOP)
    audio chunk  -> binary message  CCMF chunk type 1 (PCM or DFPWM samples)
    status       -> binary message  CCMF control frame STATUS (buffering/
                                     playing/ended) / ERROR

The lead is DELIVERY slack only (network/scheduler jitter): the client
presents every chunk by PTS against the clock this server announces.  A
`STATUS playing` carries the clock origin (spec §5.6) and is re-sent at every
timeline discontinuity, so the client never has to infer the mapping from
arrival times:

  * playback start   -> playing(origin = primary stream's first PTS)
  * VOD underrun     -> buffering; after re-prebuffering, playing(resume PTS)
  * live-edge skip   -> playing(the new head PTS)
  * live source gap  -> buffering once the gap exceeds LIVE_GAP_STATUS
  * late room joiner -> main.py unicasts playing(playback_origin()) on join

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
import dfpwm
from cc_encoder import VideoConfig
from transcoder import (
    ALL_CHANNEL_ROLES,
    AUDIO_CHUNK_SECONDS,
    GOP_SECONDS,
    PCM,
    SourceTimeline,
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

# Bound how long reconfigure_channels() waits for a session's initial setup
# (run()) before giving up -- only matters if run() never gets scheduled or
# is stuck; a normal session becomes ready within milliseconds.
_READY_TIMEOUT = 30.0

log = logging.getLogger("livecc")

# Buffering parameters (seconds of media).
VOD_PREBUFFER = 2.0
VOD_MAX_BUFFER = 15.0
LIVE_PREBUFFER = 1.5
LIVE_MAX_BUFFER = 4.0

# Release every chunk this far ahead of the playback clock.  Pure delivery
# slack: the client buffers early chunks and presents them by PTS, so the lead
# hides network/scheduler jitter without shifting anything.  One value for
# both streams — the asymmetric audio-vs-video leads of the old design existed
# only because the client used to play audio on arrival.
SEND_LEAD = 1.5

# Live: report buffering to the client once the source has been dry this long.
LIVE_GAP_STATUS = 1.0

# VOD: how far past the primary stream's head the playback origin may be
# pulled by a later-starting stream (--start section pre-roll skip).  A head
# beyond this is broken timestamps, not pre-roll.
_START_PREROLL_CAP = 10.0

_AUDIO_CHUNK_SECONDS = AUDIO_CHUNK_SECONDS   # ~2.0 s, same for both codecs


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

    def purge(self, pred) -> int:
        """Drop every buffered item whose data satisfies `pred`; return the count
        dropped.  Used when a wire-format change (ANS turning off) makes some
        buffered chunks undecodable by a newly-joined subscriber."""
        before = len(self._dq)
        self._dq = collections.deque((p, d) for p, d in self._dq if not pred(d))
        if len(self._dq) < self._max:
            self._not_full.set()
        return before - len(self._dq)

    def close(self) -> None:
        self._closed = True
        self._not_full.set()


class StreamSession:
    """One production: a source URL transcoded and paced to one sink.

    `start`/`end` are seconds into the source (the ROOM message carries ms; the
    caller converts).  `audio_codec` is the transcoder.AudioCodec negotiated
    from the client's CAPS (PCM preferred, DFPWM fallback).

    Channel roles: ONE audio producer decodes the source once and cuts every
    role this session may ever serve (self.producible_roles) from that same
    pass — a role the source has no discrete channel for carries its fallback
    mix (nearer channels, ultimately the mono downmix), so ANY speaker layout
    plays ANY source layout.  A role is "served" iff it has an open buffer.
    `caps_channels` seeds `self.requested_channels`: for a private
    session that's the one client's CAPS `channels` bitmask, fixed for life;
    for a sync room (`dynamic_channels=True`) it's the union of every current
    subscriber's request, which main._SyncGroup re-applies (via
    reconfigure_channels) on every join/leave.  Adding or removing a served
    role only opens/closes its buffer — the decode pipeline never restarts, so
    a role that joins mid-stream continues the session timeline (its first
    chunk's PTS is "now", not zero) and roles can never skew against each
    other or against video.
    """

    def __init__(self, url: str, w: int, h: int, fps: int, want_audio: bool,
                 want_video: bool = True, start: float = 0,
                 end: Optional[float] = None, loop: bool = False,
                 rate: int = 48000, audio_codec=PCM,
                 caps_channels: int = ccmf.CAP_CHANNEL_MONO,
                 dynamic_channels: bool = False,
                 compression: int = ccmf.COMPRESSION_NONE,
                 use_ans: bool = False) -> None:
        self.url = url
        self.w, self.h, self.fps = w, h, fps
        self.want_audio = want_audio
        self.want_video = want_video
        self.rate = rate
        self.codec = audio_codec
        self.requested_channels = caps_channels
        self.dynamic_channels = dynamic_channels
        self.compression = compression                   # payload compression (spec §4.1.2)
        # Live video encoding knobs (read by the GopEncoder inside iter_video at
        # each GOP boundary).  use_ans may flip during a sync room's lifetime as
        # ANS-incapable subscribers come and go (set_use_ans); packed GOPs then
        # use `compression`, ANS GOPs never do (spec §4.5.3).
        self.video_config = VideoConfig(compression=compression, use_ans=use_ans)
        self.source_channels = 1                         # probed in run() if needed
        self._codec_id = ccmf.CODEC_PCM8 if audio_codec.name == "pcm" \
            else ccmf.CODEC_DFPWM
        # DFPWM is itself an adaptive 1-bit predictive codec, so its output is
        # already near-maximum-entropy: LZ4 gains ~nothing on real audio (and
        # slightly EXPANDS it), while still costing the CC client a Lua inflate
        # per chunk.  So compress audio only when it's PCM8 (which does benefit,
        # e.g. on quiet/sparse passages).  Video keeps `compression` regardless.
        self._audio_compression = (self.compression
                                   if self._codec_id == ccmf.CODEC_PCM8
                                   else ccmf.COMPRESSION_NONE)

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
        self.producible_roles: list[int] = []            # set in run() after probing
        self.channel_roles: list[int] = []               # roles currently SERVED
        self.audio_bufs: dict[int, TimedBuffer] = {}      # role -> buffer
        self._timeline: Optional[SourceTimeline] = None

        self.video_done = asyncio.Event()
        self.audio_done: dict[int, asyncio.Event] = {}    # role -> done
        self.video_sent = 0
        self.audio_sent = 0

        # Playback clock (media seconds); valid while _playing.  Exposed via
        # playback_origin() so a sync room's late joiner can anchor (spec §5.6).
        self._playing = False
        self._clock_origin = 0.0
        self._clock_t0: Optional[float] = None

        # Set once run()'s initial setup (probe + _apply_channel_target) has
        # happened, so a racing reconfigure_channels() call (a sync room's
        # second subscriber joining before the first's session finished
        # starting up) waits instead of touching half-initialised state.
        self._ready = asyncio.Event()
        self._audio_task: Optional[asyncio.Task] = None
        self._audio_dead = False                         # producer finished/failed

    # ----- setup -----------------------------------------------------------

    def _setup_buffers(self) -> None:
        if self.is_live:
            self.prebuffer = LIVE_PREBUFFER
            self._buf_max_s, self._buf_drop = LIVE_MAX_BUFFER, True
        else:
            self.prebuffer = VOD_PREBUFFER
            self._buf_max_s, self._buf_drop = VOD_MAX_BUFFER, False
        # Video items are whole GOPs (~GOP_SECONDS each); audio buffers are
        # created on demand in _apply_channel_target (audio items
        # ~AUDIO_CHUNK_SECONDS each).
        self.video_buf = TimedBuffer(int(self._buf_max_s / GOP_SECONDS) + 1, self._buf_drop)

    def _apply_channel_target(self) -> None:
        """Diff self.channel_roles against what self.requested_channels now
        calls for, and apply the delta: open a buffer for each newly-wanted
        role, close the buffer for each role nobody wants anymore.  Every
        requested role is served unconditionally — a role the source lacks
        carries its fallback mix (transcoder._ROLE_FALLBACKS), never silence.
        Buffers are the ONLY thing that changes — the single audio producer
        keeps cutting every producible role regardless, and simply drops
        chunks for roles without a buffer.  Synchronous and await-free by
        construction, so concurrent callers can't interleave mid-update.

        A no-op when audio is off for this session: a sync room's subscriber
        churn (main._SyncGroup) calls reconfigure_channels() unconditionally,
        without knowing whether this particular room even wants audio.
        """
        if not self.want_audio:
            return
        target = set(negotiate_channel_roles(self.requested_channels))
        current = set(self.channel_roles)
        added, removed = target - current, current - target
        if self._audio_dead:
            added = set()      # a new buffer would never fill or report done
        if not added and not removed:
            return

        for role in added:
            self.audio_bufs[role] = TimedBuffer(int(self._buf_max_s / _AUDIO_CHUNK_SECONDS),
                                                self._buf_drop)
            self.audio_done[role] = asyncio.Event()

        for role in removed:
            self.audio_bufs.pop(role).close()
            self.audio_done.pop(role).set()

        self.channel_roles = sorted((current | added) - removed)
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

    def set_use_ans(self, use_ans: bool) -> None:
        """Turn the ANS video encoding on or off for this (sync-room) session as
        its membership changes -- ANS stays on only while EVERY subscriber can
        decode it (main._SyncGroup computes the AND).  The GopEncoder picks up
        the new setting at the next GOP boundary; when turning OFF we also purge
        any ANS GOPs already buffered so the ANS-incapable joiner that triggered
        the change never receives one (packed GOPs are decodable by all, so
        turning ON needs no purge).  Synchronous and await-free, so it can't
        interleave with the scheduler releasing chunks."""
        if self.video_config.use_ans == use_ans:
            return
        self.video_config.use_ans = use_ans
        if not use_ans and self.video_buf is not None:
            self.video_buf.purge(ccmf.video_chunk_is_ans)

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
                              source_path=self._source_path, loop=self.loop,
                              timeline=self._timeline, config=self.video_config)
            # iter_video yields finished CCMF GOP chunks tagged with their first
            # frame's PTS in samples (already on the shared timeline); it may
            # skip source frames to keep up (adaptive pacing) — so trust its
            # pts, don't count.
            async for pts, chunk in agen:
                # A GOP that was opened as ANS just before use_ans flipped off
                # (an ANS-incapable subscriber joining) flushes AFTER the flip;
                # drop it so that subscriber never receives an undecodable chunk.
                # The already-buffered ANS GOPs are cleared by set_use_ans().
                if not self.video_config.use_ans and ccmf.video_chunk_is_ans(chunk):
                    continue
                await self.video_buf.put(pts / ccmf.SAMPLE_RATE, chunk)
        except Exception:
            log.exception("video producer failed")
        finally:
            if agen is not None:
                await agen.aclose()
            self.video_buf.close()
            self.video_done.set()

    async def _produce_audio(self) -> None:
        # THE audio producer: one fetch, one decode, every producible role cut
        # sample-aligned from it (iter_audio_roles).  Runs for the whole
        # session; which roles actually reach the wire is decided purely by
        # which buffers are open (see the `buf is not None` check), so a sync
        # room's role churn never touches the pipeline.
        agen = None
        ev_loop = asyncio.get_running_loop()
        try:
            agen = iter_audio_roles(self.url, self.rate,
                                    roles=self.producible_roles,
                                    decode_channels=self.source_channels,
                                    start=self._start_eff, end=self._end_eff,
                                    source_path=self._source_path, loop=self.loop,
                                    timeline=self._timeline)
            async for pts, chunks in agen:
                encoded: dict[bytes, bytes] = {}
                for role, data in chunks.items():
                    buf = self.audio_bufs.get(role)
                    if buf is None:
                        continue
                    if self._codec_id == ccmf.CODEC_DFPWM:
                        # Per-chunk encode with fresh state (spec §4.6); the
                        # sequential bit-loop runs off the event loop.  Roles
                        # aliasing the same mix (fallback duplicates) encode
                        # once per batch.
                        wire = encoded.get(data)
                        if wire is None:
                            wire = await ev_loop.run_in_executor(
                                None, dfpwm.encode, data)
                            encoded[data] = wire
                        data = wire
                    chunk = ccmf.chunk(pts, ccmf.TYPE_AUDIO,
                                       ccmf.audio_payload(self._codec_id, data,
                                                          channel=role),
                                       compression=self._audio_compression)
                    await buf.put(pts / self.rate, chunk)
        except Exception:
            log.exception("audio producer failed (roles=%s)", self.producible_roles)
        finally:
            if agen is not None:
                await agen.aclose()
            self._audio_dead = True
            for buf in self.audio_bufs.values():
                buf.close()
            for done in self.audio_done.values():
                done.set()

    # ----- scheduler helpers ----------------------------------------------

    def _producing_video(self) -> bool:
        return self.want_video and not self.video_done.is_set()

    def _producing_audio(self) -> bool:
        return self.want_audio and not all(e.is_set() for e in self.audio_done.values())

    # The "primary" stream drives prebuffer/underrun gating: video when present,
    # else audio (audio-only / --no-video).  Both streams are still released by PTS
    # against the shared clock — this only picks what startup/underrun keys off.
    # With multiple audio roles, the first one stands in for all of them (they
    # fill in lockstep from the single producer).  channel_roles[0] is always
    # role 0 (mono) whenever want_audio and at least one subscriber is present:
    # a client can only get want_audio=True with the mandatory mono bit set
    # (main._negotiate), so the union driving negotiate_channel_roles always
    # includes it.
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

    def playback_origin(self) -> Optional[int]:
        """The clock's current position in samples, or None when not playing.
        What a mid-stream sync-room joiner should anchor to (spec §5.6)."""
        if not self._playing or self._clock_t0 is None:
            return None
        now = asyncio.get_running_loop().time()
        media_now = (now - self._clock_t0) + self._clock_origin
        return max(0, round(media_now * ccmf.SAMPLE_RATE))

    async def _send_buffering(self, ws) -> None:
        self._playing = False
        await ws.send_bytes(ccmf.control(ccmf.OP_STATUS,
                                         ccmf.status_body(ccmf.STATUS_BUFFERING)))

    async def _mark_playing(self, ws, origin: float, t0: float) -> None:
        """Announce (or re-announce) the playback clock: media time `origin`
        is due for presentation now (spec §5.6).  Called at start and at every
        discontinuity, so the client re-anchors instead of guessing."""
        self._clock_origin, self._clock_t0 = origin, t0
        self._playing = True
        await ws.send_bytes(ccmf.control(ccmf.OP_STATUS, ccmf.status_body(
            ccmf.STATUS_PLAYING, max(0, round(origin * ccmf.SAMPLE_RATE)))))

    async def _send_error(self, ws, message: str) -> None:
        # ERROR bodies are client-facing: keep them sanitized (no internal
        # detail; details go to the server log), per spec §7.
        # Latin-1, printable range, per spec §3 — the CC client paints these bytes
        # straight to the terminal font (no UTF-8, which would render as mojibake).
        await ws.send_bytes(ccmf.control(ccmf.OP_ERROR, message.encode("latin-1", "replace")))

    async def _release(self, ws, media_now: float) -> bool:
        sent = False
        # All buffers hold wire-ready CCMF chunks released SEND_LEAD early; the
        # client schedules them by PTS.  Chunks that are already entirely in
        # the past (a live-edge skip, or backlog behind a late-set origin) are
        # dropped here rather than shipped for the client to discard.
        for buf in self.audio_bufs.values():
            for pts, data in buf.pop_due(media_now + SEND_LEAD):
                if pts + _AUDIO_CHUNK_SECONDS < media_now:
                    continue
                await ws.send_bytes(data)
                self.audio_sent += 1
                sent = True
        for pts, data in self.video_buf.pop_due(media_now + SEND_LEAD):
            if pts + 2 * GOP_SECONDS < media_now:
                continue
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

        # Every role this session may ever have to serve — ANY role can be
        # produced from ANY source (fallback mixes), so this is bounded by who
        # might ask, not by the source: a sync room's future subscribers are
        # unpredictable (cut everything), a private client's request is fixed
        # for life.  The single producer cuts all of these for the whole
        # session, so serving a new role later is just opening a buffer,
        # never a pipeline restart.
        self.producible_roles = (
            list(ALL_CHANNEL_ROLES) if self.dynamic_channels
            else negotiate_channel_roles(self.requested_channels))

        # One timeline for however many pipelines this session runs; each
        # producer reports its first source timestamp and gets the offset that
        # places its counted PTS on the shared timeline (A/V alignment for
        # separately-fetched live streams).  live matters: it selects the
        # fallback shape when timestamps can't align (wall times for live —
        # raw counters there would starve the earlier pipeline entirely).
        self._timeline = SourceTimeline(
            [n for n, wanted in (("video", self.want_video),
                                 ("audio", self.want_audio)) if wanted],
            live=self.is_live)

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
                 "start=%ss end=%s loop=%s download=%s channels=%d",
                 self.url, self.is_live, ext, self.want_audio,
                 self._start_eff, self._end_eff, self.loop and not self.is_live,
                 self._need_download, self.source_channels)

        tasks: list[asyncio.Task] = []
        try:
            await self._send_buffering(ws)

            # Download the [start,end] section to a temp file when we need to replay
            # it (--loop) or can't pipe-stream it (GIF / MP4 moov-at-end).  Then
            # ffmpeg decodes the seekable file; --loop also replays it.  This MUST
            # happen before starting anything that reads self._source_path (the
            # producers below) -- otherwise they'd start against the pipe/None
            # and never pick up the downloaded file.
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

            self._apply_channel_target()   # open buffers; no-op if not want_audio
            if self.want_video:
                tasks.append(asyncio.create_task(self._produce_video()))
            if self.want_audio:
                self._audio_task = asyncio.create_task(self._produce_audio())
            self._ready.set()

            await self._prebuffer()

            # Playback begins at the NEWEST stream head (VOD): a --start
            # section's video legitimately reaches back to the keyframe before
            # the requested start while audio cuts near-exactly, so starting
            # at the primary (video) head would play seconds of silent
            # pre-roll.  Starting at the newest head instead skips straight to
            # where every stream has content; whatever sits before it is
            # stale by definition.  Capped: a stream claiming to start
            # implausibly far ahead (broken timestamps) must not drag the
            # session past the primary content.  Live keeps the primary head
            # — its streams are already wall-aligned and the skip logic owns
            # any catch-up.
            origin = self._primary_buf().head_pts() or 0.0
            if not self.is_live:
                heads = [b.head_pts() for b in self.audio_bufs.values()]
                if self.want_video:
                    heads.append(self.video_buf.head_pts())
                newest = max((h for h in heads if h is not None), default=origin)
                if origin < newest <= origin + _START_PREROLL_CAP:
                    origin = newest
            t0 = loop.time()
            await self._mark_playing(ws, origin, t0)

            gap_since: Optional[float] = None      # live: source dry since
            gap_reported = False

            while True:
                media_now = (loop.time() - t0) + origin
                sent = await self._release(ws, media_now)

                if self._all_done_and_empty():
                    break

                if self.is_live:
                    head = self._oldest_pts()
                    if head is None:
                        if gap_since is None:
                            gap_since = loop.time()
                        elif (not gap_reported
                              and loop.time() - gap_since > LIVE_GAP_STATUS):
                            await self._send_buffering(ws)
                            gap_reported = True
                        await asyncio.sleep(0.02)          # gap: wait for data
                        continue
                    if head - media_now > self.prebuffer + GOP_SECONDS:
                        # Behind the live edge -> skip to head.  The GOP_SECONDS
                        # margin matters: a video chunk's PTS runs up to one GOP
                        # ahead of the moment it is produced (it flushes when
                        # the NEXT GOP's first frame arrives), so head sits
                        # ~GOP_SECONDS ahead of the clock at normal cadence —
                        # without the margin this fired on every jitter, and
                        # the constant origin jumps made release timing (and
                        # the client's pacing) erratic.
                        origin, t0 = head, loop.time()
                        await self._mark_playing(ws, origin, t0)
                    elif gap_reported:
                        # Source recovered without a skip: re-announce the
                        # clock so the client leaves "Buffering" cleanly.
                        origin, t0 = media_now, loop.time()
                        await self._mark_playing(ws, origin, t0)
                    gap_since, gap_reported = None, False
                    if not sent:
                        await asyncio.sleep(0.005)
                else:  # VOD
                    if self._primary_buf().empty() and self._primary_producing():
                        await self._send_buffering(ws)
                        await self._prebuffer()             # underrun -> pause
                        origin, t0 = media_now, loop.time()
                        await self._mark_playing(ws, origin, t0)
                    elif not sent:
                        await asyncio.sleep(0.005)

            if (self.video_sent if self.want_video else self.audio_sent) == 0:
                what = "video" if self.want_video else "audio"
                await self._send_error(ws, f"Failed to load {what}. Check the server logs.")
            else:
                await ws.send_bytes(ccmf.control(
                    ccmf.OP_STATUS, ccmf.status_body(ccmf.STATUS_ENDED)))
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
            self._playing = False
            self._ready.set()   # safety net: unblock a racing caller even on an
                                # unexpected exception before the normal set() above
            if self._audio_task is not None:
                tasks.append(self._audio_task)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._tmpdir:
                shutil.rmtree(self._tmpdir, ignore_errors=True)

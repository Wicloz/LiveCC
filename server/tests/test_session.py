import asyncio

import pytest

import ccmf
import session
from session import StreamSession, TimedBuffer
from transcoder import DFPWM, PCM


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# TimedBuffer
# --------------------------------------------------------------------------- #

def test_timedbuffer_purge_drops_matching_items():
    async def go():
        b = TimedBuffer(10, drop_oldest=False)
        await b.put(0.0, b"keep0")
        await b.put(0.1, b"dropme")
        await b.put(0.2, b"keep1")
        dropped = b.purge(lambda d: d == b"dropme")
        assert dropped == 1
        assert [d for _, d in b.pop_due(9.9)] == [b"keep0", b"keep1"]
    run(go())


def test_set_use_ans_purges_buffered_ans_chunks():
    # Turning ANS off (an ANS-incapable joiner) must evict already-buffered ANS
    # GOPs so that joiner never receives one; packed chunks stay.
    import numpy as np
    from cc_encoder import GopEncoder, VideoConfig

    async def go():
        s = StreamSession("u", w=6, h=4, fps=24, want_audio=False, use_ans=True)
        s._setup_buffers()
        frame = np.random.default_rng(0).integers(
            0, 256, (4 * 3, 6 * 2, 3), dtype=np.uint8)
        ans_gop = GopEncoder(config=VideoConfig(use_ans=True))
        ans_gop.add(0, frame)
        _, ans_chunk, is_ans = ans_gop.flush(next_pts=2000)
        assert is_ans
        packed_gop = GopEncoder(config=VideoConfig(use_ans=False))
        packed_gop.add(0, frame)
        _, packed_chunk, _ = packed_gop.flush(next_pts=2000)

        await s.video_buf.put(0.0, ans_chunk)
        await s.video_buf.put(0.1, packed_chunk)
        s.set_use_ans(False)
        remaining = [d for _, d in s.video_buf.pop_due(9.9)]
        assert remaining == [packed_chunk]           # the ANS GOP was purged
        assert s.video_config.use_ans is False
    run(go())


def test_pop_due_releases_in_pts_order():
    async def go():
        b = TimedBuffer(10, drop_oldest=False)
        await b.put(0.0, b"a")
        await b.put(0.1, b"b")
        await b.put(0.2, b"c")
        assert [d for _, d in b.pop_due(0.1)] == [b"a", b"b"]
        assert [d for _, d in b.pop_due(9.9)] == [b"c"]
        assert b.empty()
    run(go())


def test_live_buffer_drops_oldest_when_full():
    async def go():
        b = TimedBuffer(2, drop_oldest=True)
        for i in range(5):
            await b.put(float(i), bytes([i]))
        remaining = [d for _, d in b.pop_due(100.0)]
        assert remaining == [bytes([3]), bytes([4])]   # only newest 2 survive
    run(go())


def test_vod_buffer_backpressures_until_drained():
    async def go():
        b = TimedBuffer(1, drop_oldest=False)
        await b.put(0.0, b"x")                     # buffer full
        blocked = asyncio.create_task(b.put(1.0, b"y"))
        await asyncio.sleep(0.05)
        assert not blocked.done()                  # producer is blocked
        b.pop_due(0.0)                             # consumer frees a slot
        await asyncio.wait_for(blocked, 1.0)       # producer now proceeds
    run(go())


def test_seconds_spans_head_to_tail():
    async def go():
        b = TimedBuffer(10, drop_oldest=False)
        await b.put(1.0, b"a")
        await b.put(3.5, b"b")
        assert b.seconds() == pytest.approx(2.5)
    run(go())


# --------------------------------------------------------------------------- #
# StreamSession scheduling (fake producers + fake socket)
# --------------------------------------------------------------------------- #

class FakeWS:
    def __init__(self):
        self.bins = []

    async def send_bytes(self, b):
        self.bins.append(b)


# Every message is binary CCMF: marker-led media chunks, or control frames.
def _media(bins, ctype):
    return [b for b in bins if b[0] == ccmf.MARKER and b[10] == ctype]


def _vids(bins):
    return _media(bins, ccmf.TYPE_VIDEO)


def _auds(bins):
    return _media(bins, ccmf.TYPE_AUDIO)


def _controls(bins):
    return [ccmf.parse_message(b) for b in bins if b[0] != ccmf.MARKER]


def _statuses(bins):
    return [body[0] for op, body in _controls(bins) if op == ccmf.OP_STATUS]


def _errors(bins):
    return [body.decode() for op, body in _controls(bins) if op == ccmf.OP_ERROR]


def _ended(bins):
    return any(op == ccmf.OP_STATUS and body and body[0] == ccmf.STATUS_ENDED
               for op, body in _controls(bins))


def _patch(monkeypatch, n_video, n_audio, is_live, ext="webm", moov_at_end=True):
    async def fake_video(url, w, h, fps, start=0, end=None, source_path=None,
                         loop=False, timeline=None, compression=0, config=None):
        # iter_video yields (pts_samples, GOP chunk); one fake chunk per "GOP"
        for i in range(n_video):
            pts = round(i * ccmf.SAMPLE_RATE / fps)
            yield pts, ccmf.chunk(pts, ccmf.TYPE_VIDEO, b"\x00" * 16)

    # The ONE audio producer: yields (pts, {role: u8 PCM chunk}) for every
    # producible role; the session serves whichever roles have open buffers.
    async def fake_audio_roles(url, rate, roles, decode_channels=1, start=0,
                               end=None, source_path=None, loop=False,
                               timeline=None):
        samples = 0
        for _ in range(n_audio):
            yield samples, {role: b"\x80" * 4096 for role in roles}
            samples += 4096

    async def fake_probe(url):
        return is_live, ext

    async def fake_moov(url):
        return moov_at_end

    monkeypatch.setattr(session, "iter_video", fake_video)
    monkeypatch.setattr(session, "iter_audio_roles", fake_audio_roles)
    monkeypatch.setattr(session, "probe_source_info", fake_probe)
    monkeypatch.setattr(session, "probe_moov_at_end", fake_moov)


def test_playing_status_carries_clock_origin(monkeypatch):
    # STATUS playing must carry the clock origin (spec §5.6): the PTS due for
    # presentation at receipt.  A VOD starts at its first frame -> origin 0.
    _patch(monkeypatch, n_video=8, n_audio=0, is_live=False)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False)
    run(s.run(ws))
    statuses = [ccmf.parse_status(body) for op, body in _controls(ws.bins)
                if op == ccmf.OP_STATUS]
    assert (ccmf.STATUS_BUFFERING, None) == statuses[0]
    playing = [origin for state, origin in statuses
               if state == ccmf.STATUS_PLAYING]
    assert playing and playing[0] == 0


def test_vod_delivers_every_chunk_in_order(monkeypatch):
    _patch(monkeypatch, n_video=8, n_audio=4, is_live=False)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=True)
    run(s.run(ws))

    vids = _vids(ws.bins)
    auds = _auds(ws.bins)
    assert len(vids) == 8          # VOD never drops
    assert len(auds) == 4
    assert _statuses(ws.bins)[0] == ccmf.STATUS_BUFFERING   # opens buffering
    assert ccmf.STATUS_PLAYING in _statuses(ws.bins)
    assert _ended(ws.bins)                                   # STATUS ended closes the stream


def test_audio_chunks_carry_sample_pts_and_codec(monkeypatch):
    _patch(monkeypatch, n_video=0, n_audio=3, is_live=False)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=True, want_video=False,
                      audio_codec=PCM)
    run(s.run(ws))
    auds = _auds(ws.bins)
    assert len(auds) == 3
    ptss = []
    for chunk in auds:
        pts, ctype, payload, _ = ccmf.parse_chunk(chunk)
        codec, channel, data = ccmf.parse_audio_payload(payload)
        assert (codec, channel) == (ccmf.CODEC_PCM8, ccmf.CHANNEL_MONO)
        assert len(data) == 4096
        ptss.append(pts)
    assert ptss == [0, 4096, 8192]      # PCM: 1 byte/sample -> running sample index


def test_audio_chunks_compressed_when_lz4_negotiated(monkeypatch):
    # A session created with LZ4 emits audio chunks whose compression byte is
    # set; parse_chunk still inflates them to the same PCM samples.
    _patch(monkeypatch, n_video=0, n_audio=3, is_live=False)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=True, want_video=False,
                      compression=ccmf.COMPRESSION_LZ4)
    run(s.run(ws))
    auds = _auds(ws.bins)
    assert auds and all(a[11] == ccmf.COMPRESSION_LZ4 for a in auds)
    _pts, _t, payload, _ = ccmf.parse_chunk(auds[0])
    codec, _channel, data = ccmf.parse_audio_payload(payload)
    assert codec == ccmf.CODEC_PCM8 and len(data) == 4096


def test_stereo_map_produces_both_channel_roles(monkeypatch):
    # A client whose CAPS ask for just a front_left+front_right pair (no mono),
    # against a (fake) 2-channel source, gets one independent audio stream per
    # role, each tagged with its own channel id -- not collapsed into mono.
    _patch(monkeypatch, n_video=0, n_audio=3, is_live=False)

    async def fake_channels(url):
        return 2

    monkeypatch.setattr(session, "probe_audio_channels", fake_channels)
    caps_channels = ccmf.CAP_CHANNEL_FRONT_LEFT | ccmf.CAP_CHANNEL_FRONT_RIGHT
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=True, want_video=False,
                      caps_channels=caps_channels)
    run(s.run(ws))

    assert s.channel_roles == [ccmf.CHANNEL_FRONT_LEFT, ccmf.CHANNEL_FRONT_RIGHT]
    auds = _auds(ws.bins)
    assert len(auds) == 6                    # 3 chunks per role x 2 roles
    channels_seen = set()
    for chunk in auds:
        _pts, _ctype, payload, _ = ccmf.parse_chunk(chunk)
        _codec, channel, _data = ccmf.parse_audio_payload(payload)
        channels_seen.add(channel)
    assert channels_seen == {ccmf.CHANNEL_FRONT_LEFT, ccmf.CHANNEL_FRONT_RIGHT}


def _patch_offset_audio(monkeypatch, video_secs, audio_start_s, audio_secs):
    # Video from pts 0 (a --start section's keyframe pre-roll); audio content
    # beginning `audio_start_s` later, chunks of 0.1 s.
    async def fake_video(url, w, h, fps, start=0, end=None, source_path=None,
                         loop=False, timeline=None, compression=0, config=None):
        for i in range(video_secs):
            pts = i * ccmf.SAMPLE_RATE
            yield pts, ccmf.chunk(pts, ccmf.TYPE_VIDEO, b"\x00" * 16)

    async def fake_audio_roles(url, rate, roles, decode_channels=1, start=0,
                               end=None, source_path=None, loop=False,
                               timeline=None):
        samples = audio_start_s * ccmf.SAMPLE_RATE
        for _ in range(audio_secs * 10):
            yield samples, {role: b"\x80" * 4800 for role in roles}
            samples += 4800

    async def fake_probe(url):
        return False, "webm"

    monkeypatch.setattr(session, "iter_video", fake_video)
    monkeypatch.setattr(session, "iter_audio_roles", fake_audio_roles)
    monkeypatch.setattr(session, "probe_source_info", fake_probe)


def _playing_origins(bins):
    return [origin for op, body in _controls(bins) if op == ccmf.OP_STATUS
            for state, origin in [ccmf.parse_status(body)]
            if state == ccmf.STATUS_PLAYING]


def test_vod_origin_skips_to_the_newest_stream_head(monkeypatch):
    # A --start section's video legitimately reaches back to the keyframe
    # before the requested start while audio cuts near-exactly: playback must
    # open where EVERY stream has content — not spend the pre-roll seconds
    # playing silent video.
    # The video drop-gate margin (_release) is 2*GOP_SECONDS; give the pre-roll
    # gap enough room past that (and within _START_PREROLL_CAP) to still
    # exercise "some dropped, some kept" regardless of how GOP_SECONDS is tuned.
    margin = 2 * session.GOP_SECONDS
    audio_start_s = int(margin) + 2
    video_secs = audio_start_s + 3
    _patch_offset_audio(monkeypatch, video_secs=video_secs,
                        audio_start_s=audio_start_s, audio_secs=1)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=True)
    run(asyncio.wait_for(s.run(ws), 10))

    assert _playing_origins(ws.bins)[0] == audio_start_s * ccmf.SAMPLE_RATE
    assert len(_auds(ws.bins)) > 0                  # the audio actually played
    # Deep pre-roll video was dropped at the gate (pts + margin < origin kept
    # nothing before origin - margin), not shipped.
    vid_pts = [ccmf.parse_chunk(c)[0] for c in _vids(ws.bins)]
    assert vid_pts and min(vid_pts) >= (audio_start_s - margin) * ccmf.SAMPLE_RATE


def test_vod_origin_preroll_cap_ignores_broken_heads(monkeypatch):
    # A stream whose first pts is implausibly far ahead (broken timestamps,
    # not pre-roll) must not drag playback away from the primary content.
    _patch_offset_audio(monkeypatch, video_secs=6, audio_start_s=60, audio_secs=1)

    async def go():
        ws = FakeWS()
        s = StreamSession("u", w=4, h=2, fps=50, want_audio=True)
        task = asyncio.create_task(s.run(ws))
        for _ in range(100):
            await asyncio.sleep(0.05)
            if _playing_origins(ws.bins):
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert _playing_origins(ws.bins) == [0]     # anchored to the video

    run(go())


def test_mono_source_still_serves_positional_roles(monkeypatch):
    # Channel mismatch is resolved with fallback mixes, not by dropping roles:
    # a stereo-mapped client on an honestly-mono source must get role 1/2
    # chunks (carrying the mono content), not six... er, two dead speakers.
    _patch(monkeypatch, n_video=0, n_audio=3, is_live=False)

    async def fake_channels(url):
        return 1                              # the source really is mono

    monkeypatch.setattr(session, "probe_audio_channels", fake_channels)
    caps = (ccmf.CAP_CHANNEL_MONO | ccmf.CAP_CHANNEL_FRONT_LEFT
            | ccmf.CAP_CHANNEL_FRONT_RIGHT)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=True, want_video=False,
                      caps_channels=caps)
    run(s.run(ws))
    assert s.channel_roles == [0, 1, 2]
    channels_seen = set()
    for chunk in _auds(ws.bins):
        _pts, _t, payload, _ = ccmf.parse_chunk(chunk)
        channels_seen.add(ccmf.parse_audio_payload(payload)[1])
    assert channels_seen == {0, 1, 2}


def test_reconfigure_channels_adds_and_removes_roles_live(monkeypatch):
    # The sync-room scenario: a subscriber union that grows (another client
    # joins wanting front_left+front_right) and later shrinks (that client
    # leaves) while the session keeps running -- newly-wanted roles are just
    # buffers opening on the one running producer (so their chunks continue
    # the session timeline), and roles nobody wants anymore stop being served
    # (buffer closed) without tearing anything else down.
    async def fake_video(url, w, h, fps, start=0, end=None, source_path=None,
                         loop=False, timeline=None, compression=0, config=None):
        if False:
            yield 0, b""

    async def fake_audio_roles(url, rate, roles, decode_channels=1, start=0,
                               end=None, source_path=None, loop=False,
                               timeline=None):
        samples = 0
        while True:
            yield samples, {role: b"\x80" * 4096 for role in roles}
            samples += 4096
            await asyncio.sleep(0.01)

    async def fake_probe(url):
        return True, "webm"   # live: never finishes on its own

    async def fake_channels(url):
        return 2

    monkeypatch.setattr(session, "iter_video", fake_video)
    monkeypatch.setattr(session, "iter_audio_roles", fake_audio_roles)
    monkeypatch.setattr(session, "probe_source_info", fake_probe)
    monkeypatch.setattr(session, "probe_audio_channels", fake_channels)
    # Buffer depth (in items) scales with 1/_AUDIO_CHUNK_SECONDS; match it to
    # this fake's actual per-chunk span (4096 samples) rather than the real
    # producer's 1.0 s chunks, so the live buffer can still hold enough TIME
    # to fill the prebuffer.
    monkeypatch.setattr(session, "_AUDIO_CHUNK_SECONDS", 4096 / ccmf.SAMPLE_RATE)

    async def go():
        ws = FakeWS()
        s = StreamSession("u", w=2, h=2, fps=50, want_audio=True, want_video=False,
                          caps_channels=ccmf.CAP_CHANNEL_MONO, dynamic_channels=True)
        task = asyncio.create_task(s.run(ws))
        await asyncio.sleep(0.05)
        assert s.channel_roles == [0]              # only mono served so far
        # …but the producer cuts every role a future subscriber could ask
        # for (any role is producible from any source via fallback mixes).
        assert s.producible_roles == list(range(9))
        audio_task_before = s._audio_task

        # A second subscriber joins wanting front_left+front_right too.
        s.requested_channels = (ccmf.CAP_CHANNEL_MONO | ccmf.CAP_CHANNEL_FRONT_LEFT
                                | ccmf.CAP_CHANNEL_FRONT_RIGHT)
        await s.reconfigure_channels()
        # Long enough for the fake (~8x realtime) to fill the live prebuffer
        # and for the scheduler to start releasing chunks.
        await asyncio.sleep(0.4)
        assert s.channel_roles == [0, 1, 2]
        assert 1 in s.audio_bufs and 2 in s.audio_bufs
        # The newly-served roles continue the running timeline -- their chunks
        # start at "now", not back at PTS 0 (the old per-role pipelines used
        # to restart their sample counters).
        role1_pts = []
        for chunk in _auds(ws.bins):
            pts, _t, payload, _ = ccmf.parse_chunk(chunk)
            if ccmf.parse_audio_payload(payload)[1] == ccmf.CHANNEL_FRONT_LEFT:
                role1_pts.append(pts)
        assert role1_pts and min(role1_pts) > 0

        # That subscriber leaves: front_left/front_right are no longer served,
        # but mono (still wanted) keeps flowing undisturbed.
        s.requested_channels = ccmf.CAP_CHANNEL_MONO
        await s.reconfigure_channels()
        assert s.channel_roles == [0]
        assert 1 not in s.audio_bufs and 2 not in s.audio_bufs
        assert s._audio_task is audio_task_before   # the pipeline never restarted

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    run(go())


def test_live_skip_reannounces_clock(monkeypatch):
    # When a live session skips to the buffered head (fell behind the edge),
    # the timeline jumped — the client can only follow if the server re-sends
    # STATUS playing with the new origin (spec §5.6).
    jump = 30 * ccmf.SAMPLE_RATE

    async def fake_video(url, w, h, fps, start=0, end=None, source_path=None,
                         loop=False, timeline=None, compression=0, config=None):
        if False:
            yield 0, b""

    async def fake_audio_roles(url, rate, roles, decode_channels=1, start=0,
                               end=None, source_path=None, loop=False,
                               timeline=None):
        samples = 0
        for _ in range(20):                    # ~1.7 s: fills the live prebuffer
            yield samples, {role: b"\x80" * 4096 for role in roles}
            samples += 4096
        samples += jump                        # source discontinuity: way ahead
        for _ in range(5):
            yield samples, {role: b"\x80" * 4096 for role in roles}
            samples += 4096
            await asyncio.sleep(0.01)
        await asyncio.sleep(3600)              # live: never ends on its own

    async def fake_probe(url):
        return True, "webm"

    monkeypatch.setattr(session, "iter_video", fake_video)
    monkeypatch.setattr(session, "iter_audio_roles", fake_audio_roles)
    monkeypatch.setattr(session, "probe_source_info", fake_probe)
    # See test_reconfigure_channels_adds_and_removes_roles_live: match the
    # live buffer's item depth to this fake's 4096-sample chunks.
    monkeypatch.setattr(session, "_AUDIO_CHUNK_SECONDS", 4096 / ccmf.SAMPLE_RATE)

    async def go():
        ws = FakeWS()
        s = StreamSession("u", w=2, h=2, fps=50, want_audio=True, want_video=False)
        task = asyncio.create_task(s.run(ws))
        await asyncio.sleep(0.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        playing = [origin for state, origin in
                   (ccmf.parse_status(body) for op, body in _controls(ws.bins)
                    if op == ccmf.OP_STATUS)
                   if state == ccmf.STATUS_PLAYING]
        assert playing[0] == 0                     # started at the first chunk
        assert any(o >= jump for o in playing)     # the skip was re-announced

    run(go())


def test_live_audio_flows_despite_unaligned_skewed_pipelines(monkeypatch):
    # THE live-silence regression, reproduced mechanically.  On a live source
    # the audio pipeline produces output almost immediately while video's
    # first frame lands seconds later (decode + encoder warm-up).  With raw
    # 0-based counters the clock anchors to video's first chunk, and the
    # audio buffer — drop-oldest, fed at realtime — keeps its head a CONSTANT
    # distance beyond the release window: every chunk is evicted before it
    # ever comes due.  Not bad lipsync; zero audio, forever.
    #
    # Both halves run REAL wall-clock pacing through the real scheduler:
    #   * control (raw counters, the old fallback): audio must starve — if
    #     this half ever starts flowing, the simulation has gone dull and the
    #     fixed half proves nothing, so revisit it;
    #   * live wall-time fallback (the fix): audio must flow.
    # The control run's steady-state audio-head gap works out to
    #   gap ≈ video_delay + prebuffer_fill(≈0.4) − audio_window
    # and it must sit BETWEEN the send lead (or audio just flows) and the
    # live-skip margin, prebuffer + GOP_SECONDS (or _oldest_pts — the AUDIO
    # head once the video buffer drains to due — trips the skip, which drags
    # origin to the audio head and "rescues" the control).  Chosen numbers
    # put gap ≈ 0.7 with ≥0.45 s of slack to each cliff, so a loaded CI
    # machine can't tip either half:
    #   lead 0.25  «  gap 0.7  «  skip margin 1.5
    # The fixed half's gap is video_backlog(≤0.6) − window(0.8) < 0 — audio
    # is due immediately once the wall-time offsets align the counters.
    monkeypatch.setattr(session, "LIVE_PREBUFFER", 0.3)
    monkeypatch.setattr(session, "SEND_LEAD", 0.25)
    monkeypatch.setattr(session, "GOP_SECONDS", 1.2)
    monkeypatch.setattr(session, "LIVE_MAX_BUFFER", 4.0)   # video: 4 GOPs buffered
    monkeypatch.setattr(session, "_AUDIO_CHUNK_SECONDS", 0.25)  # audio: 16 items
    video_delay = 1.1          # audio's head start
    chunk = 2400               # 0.05 s of samples per fake audio chunk

    async def fake_probe(url):
        return True, "webm"

    def make_fakes(use_timeline):
        async def fake_video(url, w, h, fps, start=0, end=None, source_path=None,
                             loop=False, timeline=None, compression=0, config=None):
            ev = asyncio.get_running_loop()
            await asyncio.sleep(video_delay)
            off = (await timeline.offset_samples("video", None)) if use_timeline else 0
            start_t, n = ev.time(), 0
            while True:
                pts = n * 9600 + off                       # 0.2 s per "GOP"
                yield pts, ccmf.chunk(pts, ccmf.TYPE_VIDEO, b"\x00" * 16)
                n += 1
                delay = start_t + n * 0.2 - ev.time()
                if delay > 0:
                    await asyncio.sleep(delay)

        async def fake_audio_roles(url, rate, roles, decode_channels=1, start=0,
                                   end=None, source_path=None, loop=False,
                                   timeline=None):
            # Realtime pacing with catch-up, like a real pipe: while the
            # rendezvous blocks, output backs up, then bursts through.
            ev = asyncio.get_running_loop()
            start_t = ev.time()
            off = (await timeline.offset_samples("audio", None)) if use_timeline else 0
            n = 0
            while True:
                yield n * chunk + off, {role: b"\x80" * chunk for role in roles}
                n += 1
                delay = start_t + n * 0.05 - ev.time()
                if delay > 0:
                    await asyncio.sleep(delay)

        return fake_video, fake_audio_roles

    async def run_once(use_timeline):
        fake_video, fake_audio_roles = make_fakes(use_timeline)
        monkeypatch.setattr(session, "iter_video", fake_video)
        monkeypatch.setattr(session, "iter_audio_roles", fake_audio_roles)
        monkeypatch.setattr(session, "probe_source_info", fake_probe)
        ws = FakeWS()
        s = StreamSession("u", w=2, h=2, fps=50, want_audio=True)
        task = asyncio.create_task(s.run(ws))
        await asyncio.sleep(video_delay + 1.6)     # prebuffer + a steady stretch
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return len(_vids(ws.bins)), len(_auds(ws.bins))

    async def go():
        vids, auds = await run_once(use_timeline=False)
        assert vids > 0
        assert auds == 0, "control run flowed — the starvation simulation went dull"
        vids, auds = await run_once(use_timeline=True)
        assert vids > 0
        assert auds > 0, "live wall-time fallback failed to unstarve audio"

    run(go())


def test_mono_only_caps_never_probes_channel_count(monkeypatch):
    # The common case (no speaker map -> CAPS channels is just the mandatory
    # mono bit) must not pay for the extra channel-count probe at all.
    _patch(monkeypatch, n_video=0, n_audio=2, is_live=False)

    async def fail_probe(url):
        raise AssertionError("must not probe channel count for a mono-only client")

    monkeypatch.setattr(session, "probe_audio_channels", fail_probe)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=True, want_video=False)
    run(s.run(ws))
    assert s.channel_roles == [0]
    assert len(_auds(ws.bins)) == 2


def test_session_reports_error_when_no_video(monkeypatch):
    _patch(monkeypatch, n_video=0, n_audio=0, is_live=False)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False)
    run(s.run(ws))
    assert _errors(ws.bins)
    assert not _ended(ws.bins)          # a failed stream is not a clean ended


def test_failing_video_producer_reports_error_without_hanging(monkeypatch):
    # A producer that dies (here: blows up on construction) must still set its
    # done event so the scheduler finishes — degrading to "no frames" -> ERROR
    # rather than waiting forever.  The wait_for turns a regression into a failure.
    _patch(monkeypatch, n_video=0, n_audio=0, is_live=False)

    def boom(*a, **k):
        raise RuntimeError("video kaboom")

    monkeypatch.setattr(session, "iter_video", boom)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False)
    run(asyncio.wait_for(s.run(ws), 5))
    assert _errors(ws.bins)


def test_failing_audio_producer_degrades_to_silent_video(monkeypatch):
    # If only audio fails, video must keep flowing (audio_done still gets set) —
    # graceful degradation, not a hang.
    _patch(monkeypatch, n_video=4, n_audio=0, is_live=False)

    def boom(*a, **k):
        raise RuntimeError("audio kaboom")

    monkeypatch.setattr(session, "iter_audio_roles", boom)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=True)
    run(asyncio.wait_for(s.run(ws), 5))
    assert len(_vids(ws.bins)) == 4


def test_unexpected_error_reports_generic_error(monkeypatch):
    # An error in the run path that nothing handles deeper (here: a socket send
    # that explodes on media) must be caught and reported, not dropped silently.
    _patch(monkeypatch, n_video=4, n_audio=0, is_live=False)

    class BoomWS(FakeWS):
        async def send_bytes(self, b):
            if b and b[0] == ccmf.MARKER:
                raise RuntimeError("send exploded")
            await super().send_bytes(b)

    ws = BoomWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False)
    run(asyncio.wait_for(s.run(ws), 5))
    assert any(e.startswith("Internal server error") for e in _errors(ws.bins))


def test_audio_disabled_sends_no_audio(monkeypatch):
    _patch(monkeypatch, n_video=5, n_audio=99, is_live=False)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False)
    run(s.run(ws))
    assert not _auds(ws.bins)


def test_no_video_streams_audio_only(monkeypatch):
    # --no-video: audio-only.  The scheduler keys off the audio buffer instead of
    # video; no video chunks are produced and it must not hang.  n_video=99 is a
    # canary — if a video producer were wrongly created we'd see chunks.
    _patch(monkeypatch, n_video=99, n_audio=4, is_live=False)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=True, want_video=False)
    run(asyncio.wait_for(s.run(ws), 5))
    assert len(_auds(ws.bins)) == 4
    assert not _vids(ws.bins)
    assert ccmf.STATUS_PLAYING in _statuses(ws.bins)


def test_both_streams_disabled_reports_error(monkeypatch):
    _patch(monkeypatch, n_video=0, n_audio=0, is_live=False)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False, want_video=False)
    run(asyncio.wait_for(s.run(ws), 5))
    assert any(e.startswith("Nothing to play") for e in _errors(ws.bins))


def test_audio_codec_is_negotiated_not_hardcoded():
    # The codec comes from the client's CAPS via the caller; both spec codecs work.
    assert StreamSession("u", w=4, h=2, fps=24, want_audio=True).codec.name == "pcm"
    assert StreamSession("u", w=4, h=2, fps=24, want_audio=True,
                         audio_codec=DFPWM).codec.name == "dfpwm"


def test_dfpwm_chunks_are_tagged_and_packed_dfpwm(monkeypatch):
    _patch(monkeypatch, n_video=0, n_audio=2, is_live=False)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=True, want_video=False,
                      audio_codec=DFPWM)
    run(s.run(ws))
    auds = _auds(ws.bins)
    assert auds
    _pts, _t, payload, _ = ccmf.parse_chunk(auds[0])
    codec, _chan, data = ccmf.parse_audio_payload(payload)
    assert codec == ccmf.CODEC_DFPWM
    # The producer yields PCM (4096 samples); the session packs DFPWM at
    # 8 samples/byte, and PTS advances by SAMPLES either way.
    assert len(data) == 4096 // 8
    pts2, _t, _p, _ = ccmf.parse_chunk(auds[1])
    assert pts2 == 4096


def test_loop_downloads_section_once_then_streams(monkeypatch):
    _patch(monkeypatch, n_video=6, n_audio=0, is_live=False)
    calls = {"n": 0, "args": None}

    async def fake_download(url, out_dir, start, end, want_audio):
        calls["n"] += 1
        calls["args"] = (start, end)
        return "/tmp/fake_source.mkv"

    monkeypatch.setattr(session, "download_source", fake_download)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False,
                      start=30, end=60, loop=True)
    run(s.run(ws))

    assert calls["n"] == 1                  # cached exactly once
    assert calls["args"] == (30, 60)        # seconds passed through
    assert s._source_path == "/tmp/fake_source.mkv"
    assert len(_vids(ws.bins)) == 6


def test_moov_at_end_mp4_vod_downloads_for_seekable_decode(monkeypatch):
    # A moov-at-end MP4 VOD can't be pipe-streamed, so even without --loop the
    # session downloads it once and decodes the seekable file.
    _patch(monkeypatch, n_video=4, n_audio=0, is_live=False, ext="mp4", moov_at_end=True)
    calls = {"n": 0}

    async def fake_download(url, out_dir, start, end, want_audio):
        calls["n"] += 1
        return "/tmp/fake_source.mkv"

    monkeypatch.setattr(session, "download_source", fake_download)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False)   # no --loop
    run(s.run(ws))

    assert calls["n"] == 1                          # downloaded despite no --loop
    assert s._source_path == "/tmp/fake_source.mkv"
    assert len(_vids(ws.bins)) == 4


def test_gif_downloads_and_plays_once_without_loop(monkeypatch):
    # A GIF can't be pipe-streamed, so the session downloads it and decodes the
    # file — but it does NOT loop unless --loop was given.  It skips the moov probe.
    _patch(monkeypatch, n_video=4, n_audio=0, is_live=False, ext="gif")
    calls = {"download": 0, "moov": 0}

    async def fake_download(url, out_dir, start, end, want_audio):
        calls["download"] += 1
        return "/tmp/fake_source.gif"

    async def counting_moov(url):
        calls["moov"] += 1
        return False

    monkeypatch.setattr(session, "download_source", fake_download)
    monkeypatch.setattr(session, "probe_moov_at_end", counting_moov)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False)   # no --loop flag
    run(s.run(ws))

    assert s.loop is False                           # GIF is not looped by default
    assert calls["download"] == 1                    # downloaded (can't pipe-stream)
    assert calls["moov"] == 0                        # GIF isn't an MP4 -> no probe
    assert s._source_path == "/tmp/fake_source.gif"
    assert len(_vids(ws.bins)) == 4


def test_faststart_mp4_vod_streams_without_download(monkeypatch):
    # A faststart MP4 (moov up front) pipe-streams fine — the moov probe says so,
    # so it must NOT be downloaded.
    _patch(monkeypatch, n_video=3, n_audio=0, is_live=False, ext="mp4", moov_at_end=False)

    async def fail_download(*a, **k):
        raise AssertionError("faststart MP4 must stream, not download")

    monkeypatch.setattr(session, "download_source", fail_download)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False)
    run(s.run(ws))
    assert s._source_path is None


def test_webm_vod_streams_without_download(monkeypatch):
    # WebM streams fine from a pipe — must NOT even probe moov or download.
    _patch(monkeypatch, n_video=3, n_audio=0, is_live=False, ext="webm")

    async def fail_download(*a, **k):
        raise AssertionError("webm VOD must stream, not download")

    monkeypatch.setattr(session, "download_source", fail_download)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False)
    run(s.run(ws))
    assert s._source_path is None


def test_cancel_finalizes_producer_generator(monkeypatch):
    # The real leak: on cancel, the producer generator must be aclose()d so its
    # finally (which kills yt-dlp/ffmpeg) runs promptly — not left to GC.
    closed = {"video": False}

    async def fake_video(url, w, h, fps, start=0, end=None, source_path=None,
                         loop=False, timeline=None, compression=0, config=None):
        i = 0
        try:
            while True:
                pts = round(i * ccmf.SAMPLE_RATE / fps)
                yield pts, ccmf.chunk(pts, ccmf.TYPE_VIDEO, b"\x00" * 16)
                i += 1
                await asyncio.sleep(0.01)
        finally:
            closed["video"] = True

    async def fake_audio(url, rate, roles, decode_channels=1, start=0, end=None,
                         source_path=None, loop=False, timeline=None, compression=0, config=None):
        if False:
            yield 0, {}

    async def fake_probe(url):
        return False, "webm"

    monkeypatch.setattr(session, "iter_video", fake_video)
    monkeypatch.setattr(session, "iter_audio_roles", fake_audio)
    monkeypatch.setattr(session, "probe_source_info", fake_probe)

    async def go():
        s = StreamSession("u", w=2, h=2, fps=50, want_audio=False)
        task = asyncio.create_task(s.run(FakeWS()))
        await asyncio.sleep(0.2)            # let it start streaming
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    run(go())
    assert closed["video"] is True         # generator finalized -> subprocess killed


def test_loop_ignored_for_livestream(monkeypatch):
    _patch(monkeypatch, n_video=3, n_audio=0, is_live=True)

    async def fail_download(*a, **k):
        raise AssertionError("must not download for a livestream")

    monkeypatch.setattr(session, "download_source", fail_download)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False, loop=True)
    # Live runs forever; cancel shortly after it starts streaming.
    async def go():
        task = asyncio.create_task(s.run(ws))
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    run(go())
    assert s._source_path is None

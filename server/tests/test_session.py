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
    return any(op == ccmf.OP_END for op, _ in _controls(bins))


def _patch(monkeypatch, n_video, n_audio, is_live, ext="webm", moov_at_end=True):
    async def fake_video(url, w, h, fps, start=0, end=None, source_path=None,
                         loop=False):
        # iter_video yields (pts_samples, GOP chunk); one fake chunk per "GOP"
        for i in range(n_video):
            pts = round(i * ccmf.SAMPLE_RATE / fps)
            yield pts, ccmf.chunk(pts, ccmf.TYPE_VIDEO, b"\x00" * 16)

    async def fake_audio(url, rate, codec=None, start=0, end=None,
                         source_path=None, loop=False, source_channel=None):
        for _ in range(n_audio):
            yield b"\x00" * 4096

    async def fake_probe(url):
        return is_live, ext

    async def fake_moov(url):
        return moov_at_end

    monkeypatch.setattr(session, "iter_video", fake_video)
    monkeypatch.setattr(session, "iter_audio", fake_audio)
    monkeypatch.setattr(session, "probe_source_info", fake_probe)
    monkeypatch.setattr(session, "probe_moov_at_end", fake_moov)


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
    assert _ended(ws.bins)                                   # END closes the stream


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


def test_stereo_map_produces_both_channel_roles(monkeypatch):
    # A client whose CAPS ask for a full front_left+front_right pair, against a
    # (fake) 2-channel source, gets one independent audio stream per role,
    # each tagged with its own channel id -- not both collapsed into mono.
    _patch(monkeypatch, n_video=0, n_audio=3, is_live=False)

    async def fake_channels(url):
        return 2

    monkeypatch.setattr(session, "probe_audio_channels", fake_channels)
    caps_channels = ccmf.CAP_CHANNEL_MONO | ccmf.CAP_CHANNEL_FRONT_LEFT | ccmf.CAP_CHANNEL_FRONT_RIGHT
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
    assert not _ended(ws.bins)          # a failed stream is not a clean END


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

    monkeypatch.setattr(session, "iter_audio", boom)
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


def test_dfpwm_chunks_are_tagged_dfpwm(monkeypatch):
    _patch(monkeypatch, n_video=0, n_audio=2, is_live=False)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=True, want_video=False,
                      audio_codec=DFPWM)
    run(s.run(ws))
    auds = _auds(ws.bins)
    assert auds
    _pts, _t, payload, _ = ccmf.parse_chunk(auds[0])
    codec, _chan, _data = ccmf.parse_audio_payload(payload)
    assert codec == ccmf.CODEC_DFPWM
    # DFPWM: 8 samples/byte -> the second chunk's PTS reflects that
    pts2, _t, _p, _ = ccmf.parse_chunk(auds[1])
    assert pts2 == 4096 * 8


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
                         loop=False):
        i = 0
        try:
            while True:
                pts = round(i * ccmf.SAMPLE_RATE / fps)
                yield pts, ccmf.chunk(pts, ccmf.TYPE_VIDEO, b"\x00" * 16)
                i += 1
                await asyncio.sleep(0.01)
        finally:
            closed["video"] = True

    async def fake_audio(url, rate, codec=None, start=0, end=None,
                         source_path=None, loop=False, source_channel=None):
        if False:
            yield b""

    async def fake_probe(url):
        return False, "webm"

    monkeypatch.setattr(session, "iter_video", fake_video)
    monkeypatch.setattr(session, "iter_audio", fake_audio)
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

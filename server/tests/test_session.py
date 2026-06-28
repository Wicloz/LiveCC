import asyncio

import pytest

import session
from session import StreamSession, TimedBuffer


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
        self.texts = []
        self.bins = []

    async def send_text(self, s):
        self.texts.append(s)

    async def send_bytes(self, b):
        self.bins.append(b)


def _patch(monkeypatch, n_video, n_audio, is_live, ext="webm", moov_at_end=True):
    async def fake_video(url, w, h, fps, start=0, end=None, source_path=None, loop=False):
        for i in range(n_video):   # iter_video yields (pts, frame); pts = i/fps
            yield i / fps, bytes((0, w, 0, h)) + b"\x00" * (w * h * 3)

    async def fake_audio(url, rate, codec=None, start=0, end=None,
                         source_path=None, loop=False):
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


def test_vod_delivers_every_frame_in_order(monkeypatch):
    _patch(monkeypatch, n_video=8, n_audio=4, is_live=False)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=True)
    run(s.run(ws))

    vids = [b for b in ws.bins if b[:1] == session.OP_VIDEO]
    auds = [b for b in ws.bins if b[:1] == session.OP_AUDIO]
    assert len(vids) == 8          # VOD never drops
    assert len(auds) == 4
    assert ws.texts[0].startswith("META")
    assert "PLAYING" in ws.texts


def test_session_reports_error_when_no_video(monkeypatch):
    _patch(monkeypatch, n_video=0, n_audio=0, is_live=False)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False)
    run(s.run(ws))
    assert any(t.startswith("ERROR") for t in ws.texts)


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
    assert any(t.startswith("ERROR") for t in ws.texts)


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
    assert len([b for b in ws.bins if b[:1] == session.OP_VIDEO]) == 4


def test_unexpected_error_reports_generic_error(monkeypatch):
    # An error in the run path that nothing handles deeper (here: a socket send
    # that explodes) must be caught and reported to the client, not dropped.
    _patch(monkeypatch, n_video=4, n_audio=0, is_live=False)

    class BoomWS(FakeWS):
        async def send_bytes(self, b):
            raise RuntimeError("send exploded")

    ws = BoomWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False)
    run(asyncio.wait_for(s.run(ws), 5))
    assert any(t.startswith("ERROR Internal server error") for t in ws.texts)


def test_audio_disabled_sends_no_audio(monkeypatch):
    _patch(monkeypatch, n_video=5, n_audio=99, is_live=False)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False)
    run(s.run(ws))
    assert all(b[:1] != session.OP_AUDIO for b in ws.bins)


def test_no_video_streams_audio_only(monkeypatch):
    # --no-video: audio-only.  The scheduler keys off the audio buffer instead of
    # video; no video frames are produced and it must not hang.  n_video=99 is a
    # canary — if a video producer were wrongly created we'd see frames.
    _patch(monkeypatch, n_video=99, n_audio=4, is_live=False)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=True, want_video=False)
    run(asyncio.wait_for(s.run(ws), 5))
    assert len([b for b in ws.bins if b[:1] == session.OP_AUDIO]) == 4
    assert all(b[:1] != session.OP_VIDEO for b in ws.bins)
    assert "PLAYING" in ws.texts


def test_both_streams_disabled_reports_error(monkeypatch):
    _patch(monkeypatch, n_video=0, n_audio=0, is_live=False)
    ws = FakeWS()
    s = StreamSession("u", w=4, h=2, fps=50, want_audio=False, want_video=False)
    run(asyncio.wait_for(s.run(ws), 5))
    assert any(t.startswith("ERROR Nothing to play") for t in ws.texts)


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
                      start="30", end="1m", loop=True)
    run(s.run(ws))

    assert calls["n"] == 1                  # cached exactly once
    assert calls["args"] == (30, 60)        # parsed start=30s, end=60s
    assert s._source_path == "/tmp/fake_source.mkv"
    assert len([b for b in ws.bins if b[:1] == session.OP_VIDEO]) == 6


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
    assert len([b for b in ws.bins if b[:1] == session.OP_VIDEO]) == 4


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
    assert len([b for b in ws.bins if b[:1] == session.OP_VIDEO]) == 4


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


def test_crunchy_selects_dfpwm_codec():
    # --crunchy trades fidelity for bandwidth: 1-bit DFPWM instead of raw PCM.
    assert StreamSession("u", w=4, h=2, fps=24, want_audio=True).codec.name == "pcm"
    assert StreamSession("u", w=4, h=2, fps=24, want_audio=True,
                         crunchy=True).codec.name == "dfpwm"


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

    async def fake_video(url, w, h, fps, start=0, end=None, source_path=None, loop=False):
        i = 0
        try:
            while True:
                yield i / fps, b"\x00" * 16   # (pts, frame)
                i += 1
                await asyncio.sleep(0.01)
        finally:
            closed["video"] = True

    async def fake_audio(url, rate, codec=None, start=0, end=None,
                         source_path=None, loop=False):
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

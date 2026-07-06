"""
Full-stack tests: the real FastAPI app over a real (test) websocket.

The unit suites pin individual pieces; these pin the SEAMS — handshake
framing, negotiation, session wiring, and (with real tools present) the whole
yt-dlp -> ffmpeg -> encoder -> websocket pipeline against a served media file,
exactly the path a CC client exercises in production.  The stereo test exists
because a real regression ("stereo went mono / live went silent") lived
precisely in a seam no unit test crossed: the channel-count probe against a
source yt-dlp has no metadata for.
"""

import http.server
import shutil
import socketserver
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import ccmf
import main
import session
import transcoder

_HAVE_TOOLS = shutil.which("ffmpeg") is not None and shutil.which("yt-dlp") is not None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _handshake(ws, url, channels=ccmf.CAP_CHANNEL_MONO, w=8, h=4, fps=5):
    ws.send_bytes(ccmf.control(ccmf.OP_ROOM, ccmf.build_room(url)))
    op, _ = ccmf.parse_message(ws.receive_bytes())
    assert op == ccmf.OP_ACK
    ws.send_bytes(ccmf.control(ccmf.OP_CAPS, ccmf.build_caps(
        channels=channels, width=w, height=h, fps=fps)))
    op, _ = ccmf.parse_message(ws.receive_bytes())
    assert op == ccmf.OP_ACK
    ws.send_bytes(ccmf.control(ccmf.OP_START))


def _collect_stream(ws, max_messages=20000):
    """Read until STATUS ended (or ERROR/exhaustion) -> (roles, statuses,
    n_video, errors); roles maps channel -> [(pts, payload bytes), ...]."""
    roles: dict[int, list] = {}
    statuses, errors = [], []
    n_video = 0
    for _ in range(max_messages):
        op, body = ccmf.parse_message(ws.receive_bytes())
        if op == ccmf.MARKER:
            pts, ctype, payload, _ = ccmf.parse_chunk(body)
            if ctype == ccmf.TYPE_AUDIO:
                _codec, channel, data = ccmf.parse_audio_payload(payload)
                roles.setdefault(channel, []).append((pts, data))
            else:
                n_video += 1
        elif op == ccmf.OP_STATUS:
            st = ccmf.parse_status(body)
            statuses.append(st)
            if st[0] == ccmf.STATUS_ENDED:
                break
        elif op == ccmf.OP_ERROR:
            errors.append(body.decode("latin-1"))
            break
    return roles, statuses, n_video, errors


@pytest.fixture
def media_server(tmp_path):
    """Serve tmp_path over local HTTP; yields a base URL."""
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(tmp_path), **k)

        def log_message(self, *a):
            pass

    httpd = socketserver.TCPServer(("127.0.0.1", 0), Quiet)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# real pipeline: served stereo VOD through the whole stack
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not _HAVE_TOOLS, reason="needs ffmpeg and yt-dlp on PATH")
def test_vod_stereo_survives_unknown_channel_probe(tmp_path, media_server):
    # A direct-URL VOD goes through yt-dlp's generic extractor, which reports
    # audio_channels as "NA".  That MUST NOT collapse the production to mono:
    # a stereo client must still get distinct front_left/front_right (plus the
    # mono downmix), cut from a real fetch + decode.  This is the end-to-end
    # net for the "stereo went mono" regression — it fails if any layer
    # (probe, negotiation, decode, de-interleave, wire) drops the channels.
    clip = tmp_path / "stereo.mp4"
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=3",
         "-f", "lavfi", "-i", "aevalsrc=0.25|0.75:c=stereo:s=48000:d=3",
         "-c:v", "mpeg4", "-c:a", "aac", "-movflags", "+faststart",
         "-shortest", str(clip)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0:
        pytest.skip("ffmpeg can't build the stereo clip")
    url = f"{media_server}/stereo.mp4"

    # Precondition for the regression: the probe really can't tell here.  If
    # yt-dlp ever starts reporting a count for generic URLs, this test loses
    # its point and should move to a source that still probes unknown.
    assert transcoder._probe_audio_channels_blocking(url) == transcoder._ASSUMED_CHANNELS

    client = TestClient(main.app)
    stereo_caps = (ccmf.CAP_CHANNEL_MONO | ccmf.CAP_CHANNEL_FRONT_LEFT
                   | ccmf.CAP_CHANNEL_FRONT_RIGHT)
    with client.websocket_connect("/ws/play") as ws:
        _handshake(ws, url, channels=stereo_caps)
        roles, statuses, n_video, errors = _collect_stream(ws)

    assert not errors
    assert n_video > 0
    assert set(roles) == {ccmf.CHANNEL_MONO, ccmf.CHANNEL_FRONT_LEFT,
                          ccmf.CHANNEL_FRONT_RIGHT}

    def level(role):
        data = b"".join(d for _p, d in roles[role])
        assert len(data) > 48000        # at least a second of each role
        return np.frombuffer(data, np.uint8).mean()

    left = level(ccmf.CHANNEL_FRONT_LEFT)      # 0.25 -> ~160
    right = level(ccmf.CHANNEL_FRONT_RIGHT)    # 0.75 -> ~224
    mono = level(ccmf.CHANNEL_MONO)
    assert left < 180 < right                  # distinct, not swapped, not downmixed
    assert left < mono < right                 # mono = downmix of the same frames

    # Every role is PTS-contiguous on the wire (chunk N+1 = chunk N + samples).
    for role, chunks in roles.items():
        for (p1, d1), (p2, _d2) in zip(chunks, chunks[1:]):
            assert p2 == p1 + len(d1), f"role {role} has a PTS hole"

    # The clock was announced before any media.
    assert statuses[0][0] == ccmf.STATUS_BUFFERING
    assert any(st == ccmf.STATUS_PLAYING and origin is not None
               for st, origin in statuses)


# --------------------------------------------------------------------------- #
# fast full-stack: real websocket framing, faked producers
# --------------------------------------------------------------------------- #

def _patch_fast_session(monkeypatch, n_video=4, n_audio=6, is_live=False):
    async def fake_video(url, w, h, fps, start=0, end=None, source_path=None,
                         loop=False, timeline=None):
        for i in range(n_video):
            pts = round(i * ccmf.SAMPLE_RATE / fps)
            yield pts, ccmf.chunk(pts, ccmf.TYPE_VIDEO, b"\x00" * 16)

    async def fake_audio_roles(url, rate, roles, decode_channels=1, start=0,
                               end=None, source_path=None, loop=False,
                               timeline=None):
        samples = 0
        for _ in range(n_audio):
            yield samples, {role: bytes([128 + role]) * 4800 for role in roles}
            samples += 4800

    async def fake_probe(url):
        return is_live, "webm"

    async def fake_channels(url):
        return 2

    monkeypatch.setattr(session, "iter_video", fake_video)
    monkeypatch.setattr(session, "iter_audio_roles", fake_audio_roles)
    monkeypatch.setattr(session, "probe_source_info", fake_probe)
    monkeypatch.setattr(session, "probe_audio_channels", fake_channels)


def test_full_websocket_session_end_to_end(monkeypatch):
    # Handshake framing, negotiation, streaming, and shutdown over a real
    # websocket — the layer test_main's unit tests fake away.
    _patch_fast_session(monkeypatch)
    client = TestClient(main.app)
    stereo_caps = (ccmf.CAP_CHANNEL_MONO | ccmf.CAP_CHANNEL_FRONT_LEFT
                   | ccmf.CAP_CHANNEL_FRONT_RIGHT)
    with client.websocket_connect("/ws/play") as ws:
        _handshake(ws, "https://example.com/v", channels=stereo_caps)
        roles, statuses, n_video, errors = _collect_stream(ws)

    assert not errors
    assert n_video == 4
    assert set(roles) == {0, 1, 2}             # the union the caps asked for
    for role, chunks in roles.items():
        assert [p for p, _ in chunks] == [i * 4800 for i in range(len(chunks))]
    assert statuses[0][0] == ccmf.STATUS_BUFFERING
    assert (ccmf.STATUS_PLAYING, 0) in statuses
    assert statuses[-1][0] == ccmf.STATUS_ENDED


def test_websocket_rejects_malformed_room():
    client = TestClient(main.app)
    with client.websocket_connect("/ws/play") as ws:
        ws.send_bytes(ccmf.control(ccmf.OP_ROOM, b""))     # empty body
        op, body = ccmf.parse_message(ws.receive_bytes())
        assert op == ccmf.OP_ERROR
        assert b"ROOM" in body

import asyncio

import ccmf
import main
from transcoder import DFPWM, PCM


class _FakeURL:
    def __init__(self, scheme):
        self.scheme = scheme


class _FakeRequest:
    """Minimal stand-in for a Starlette Request for the URL helpers."""

    def __init__(self, headers=None, scheme="http", base_url="http://fallback:8080/"):
        self.headers = headers or {}
        self.url = _FakeURL(scheme)
        self.base_url = base_url


class FakeWS:
    """Minimal stand-in for a Starlette WebSocket: scripted incoming messages,
    recorded outgoing ones."""

    def __init__(self, incoming=()):
        self._incoming = list(incoming)
        self.sent = []
        self.closed = None

    async def receive(self):
        if not self._incoming:
            return {"type": "websocket.disconnect"}
        return {"type": "websocket.receive", "bytes": self._incoming.pop(0)}

    async def send_bytes(self, b):
        self.sent.append(b)

    async def close(self, code=1000):
        self.closed = code


def _controls(sent):
    return [ccmf.parse_message(b) for b in sent]


def _room(url="https://ex/v", **kw):
    return ccmf.control(ccmf.OP_ROOM, ccmf.build_room(url, **kw))


def _caps(**kw):
    return ccmf.control(ccmf.OP_CAPS, ccmf.build_caps(**kw))


# --------------------------------------------------------------------------- #
# _server_base_url — keep https intact behind a TLS-terminating proxy
# --------------------------------------------------------------------------- #

def test_base_url_honours_forwarded_proto():
    req = _FakeRequest(headers={"x-forwarded-proto": "https", "host": "cc.example:8080"},
                       scheme="http")
    assert main._server_base_url(req) == "https://cc.example:8080"


def test_base_url_honours_forwarded_host():
    req = _FakeRequest(headers={"x-forwarded-proto": "https",
                                "x-forwarded-host": "public.example",
                                "host": "internal:8080"})
    assert main._server_base_url(req) == "https://public.example"


def test_base_url_takes_first_proto_in_list():
    req = _FakeRequest(headers={"x-forwarded-proto": "https, http", "host": "h"})
    assert main._server_base_url(req) == "https://h"


def test_base_url_falls_back_to_request_scheme_and_host():
    req = _FakeRequest(headers={"host": "h:8080"}, scheme="http")
    assert main._server_base_url(req) == "http://h:8080"


def test_base_url_falls_back_to_base_url_without_host():
    req = _FakeRequest(headers={}, base_url="http://fb:8080/")
    assert main._server_base_url(req) == "http://fb:8080"


# --------------------------------------------------------------------------- #
# Server URL baked into the Lua scripts
# --------------------------------------------------------------------------- #

def test_serve_lua_injects_server_url():
    req = _FakeRequest(headers={"x-forwarded-proto": "https", "host": "srv"})
    body = main._serve_lua("player.lua", req).body.decode()
    assert 'local SERVER = "https://srv"' in body
    assert "{{SERVER}}" not in body


def test_lua_scripts_contain_server_placeholder():
    # Both scripts must keep the placeholder so the server can inject its URL.
    for name in ("player.lua", "install.lua"):
        assert '"{{SERVER}}"' in (main._LUA_DIR / name).read_text()


# --------------------------------------------------------------------------- #
# Handshake (spec §5.3): ROOM -> ACK, CAPS -> ACK, START
# --------------------------------------------------------------------------- #

def test_handshake_acks_room_and_returns_both_hellos():
    ws = FakeWS([_room(start_ms=90_000, loop=True), _caps(width=80, height=20)])

    got = asyncio.run(main._handshake(ws))
    assert got is not None
    room, caps = got
    assert room.url == "https://ex/v" and room.start_ms == 90_000 and room.loop
    assert (caps.width, caps.height) == (80, 20)
    # exactly one ACK so far (ROOM); the CAPS ACK is the caller's accept decision
    assert [op for op, _ in _controls(ws.sent)] == [ccmf.OP_ACK]


def test_handshake_rejects_wrong_first_opcode():
    ws = FakeWS([ccmf.control(ccmf.OP_START)])
    assert asyncio.run(main._handshake(ws)) is None
    assert any(op == ccmf.OP_ERROR for op, _ in _controls(ws.sent))
    assert ws.closed == 1008


def test_handshake_rejects_malformed_room():
    ws = FakeWS([ccmf.control(ccmf.OP_ROOM, b"")])
    assert asyncio.run(main._handshake(ws)) is None
    assert any(op == ccmf.OP_ERROR for op, _ in _controls(ws.sent))


def test_handshake_rejects_disconnect_before_caps():
    ws = FakeWS([_room()])
    assert asyncio.run(main._handshake(ws)) is None


def test_await_start_accepts_start_only():
    assert asyncio.run(main._await_start(FakeWS([ccmf.control(ccmf.OP_START)])))
    ws = FakeWS([ccmf.control(ccmf.OP_QUIT)])
    assert not asyncio.run(main._await_start(ws))
    assert ws.closed == 1008


def test_until_quit_returns_on_quit_and_on_disconnect():
    # QUIT resolves the watcher; so does the socket closing.
    asyncio.run(asyncio.wait_for(
        main._until_quit_or_disconnect(FakeWS([ccmf.control(ccmf.OP_QUIT)])), 1))
    asyncio.run(asyncio.wait_for(
        main._until_quit_or_disconnect(FakeWS([])), 1))


# --------------------------------------------------------------------------- #
# CAPS negotiation -> session kwargs
# --------------------------------------------------------------------------- #

def test_negotiate_defaults_prefer_pcm():
    room = ccmf.parse_room(ccmf.build_room("u", start_ms=10_000, end_ms=20_000,
                                           loop=True))
    caps = ccmf.parse_caps(ccmf.build_caps(width=80, height=20, fps=30))
    kwargs = main._negotiate(room, caps)
    assert kwargs == {
        "url": "u",
        "w": 80,
        "h": 20,
        "fps": 30,
        "want_audio": True,
        "want_video": True,
        "start": 10.0,          # ms -> seconds
        "end": 20.0,
        "loop": True,
        "audio_codec": PCM,
        "caps_channels": ccmf.CAP_CHANNEL_MONO,
        "compression": ccmf.COMPRESSION_NONE,   # default caps advertise none only
        "use_ans": False,                        # default caps advertise no ANS
    }


def test_negotiate_picks_lz4_when_advertised():
    room = ccmf.parse_room(ccmf.build_room("u"))
    caps = ccmf.parse_caps(ccmf.build_caps(
        compress_mask=ccmf.CAP_COMPRESS_NONE | ccmf.CAP_COMPRESS_LZ4))
    assert main._negotiate(room, caps)["compression"] == ccmf.COMPRESSION_LZ4


def test_negotiate_falls_back_to_dfpwm():
    room = ccmf.parse_room(ccmf.build_room("u"))
    caps = ccmf.parse_caps(ccmf.build_caps(audio_mask=ccmf.CAP_AUDIO_DFPWM))
    assert main._negotiate(room, caps)["audio_codec"] is DFPWM


def test_negotiate_no_codec_or_no_mono_means_no_audio():
    room = ccmf.parse_room(ccmf.build_room("u"))
    no_codec = ccmf.parse_caps(ccmf.build_caps(audio_mask=0))
    assert main._negotiate(room, no_codec)["want_audio"] is False
    no_mono = ccmf.parse_caps(ccmf.build_caps(channels=0))
    assert main._negotiate(room, no_mono)["want_audio"] is False


def test_negotiate_nothing_to_play_is_rejected():
    room = ccmf.parse_room(ccmf.build_room("u"))
    caps = ccmf.parse_caps(ccmf.build_caps(want_video=False, want_audio=False))
    assert main._negotiate(room, caps) is None
    # audio-only request with no decodable codec is equally unplayable
    caps = ccmf.parse_caps(ccmf.build_caps(want_video=False, audio_mask=0))
    assert main._negotiate(room, caps) is None


def test_negotiate_clamps_fps_and_defaults_height():
    room = ccmf.parse_room(ccmf.build_room("u"))
    caps = ccmf.parse_caps(ccmf.build_caps(width=80, height=0, fps=200))
    kwargs = main._negotiate(room, caps)
    assert kwargs["w"] == 80
    assert kwargs["h"] == 19           # 0 -> default
    assert kwargs["fps"] == 30         # clamped high


def test_negotiate_accepts_ccs_largest_real_grid():
    # 335x124 (mon16x9, cc_media.py's GRID_PRESETS) is the biggest grid a real
    # CC monitor can be -- it must fit under the cell-count budget untouched.
    room = ccmf.parse_room(ccmf.build_room("u"))
    caps = ccmf.parse_caps(ccmf.build_caps(width=335, height=124))
    kwargs = main._negotiate(room, caps)
    assert (kwargs["w"], kwargs["h"]) == (335, 124)


def test_negotiate_drops_grid_over_cell_budget():
    # Dropped rather than downscaled: the client asked for a specific size
    # that CCMF's u16 delta-span `start` field (spec Section 4.5.2) can't
    # address (256x256 = 65536 cells, one over the 65535 cap).
    room = ccmf.parse_room(ccmf.build_room("u"))
    caps = ccmf.parse_caps(ccmf.build_caps(width=256, height=256))
    assert main._negotiate(room, caps) is None


# --------------------------------------------------------------------------- #
# Sync rooms: keyed by (url, start, end, loop); CAPS divergence is dropped
# --------------------------------------------------------------------------- #

def _room_and_kwargs(url="u", fps=24, **room_kw):
    room = ccmf.parse_room(ccmf.build_room(url, sync=True, **room_kw))
    caps = ccmf.parse_caps(ccmf.build_caps(fps=fps))
    return room, main._negotiate(room, caps)


def test_sync_group_reuse_and_release():
    class WS:
        pass

    async def go():
        main._sync_groups.clear()
        room, kwargs = _room_and_kwargs()
        g1 = await main._acquire_sync_group(room, kwargs)
        g2 = await main._acquire_sync_group(room, kwargs)
        assert g1 is g2

        ws = WS()
        await g1.subscribe(ws, ccmf.CAP_CHANNEL_MONO, False)
        await main._release_sync_group(room.key(), ws)
        assert room.key() not in main._sync_groups

    asyncio.run(go())


def test_sync_group_rejects_mismatched_caps():
    async def go():
        main._sync_groups.clear()
        room, kwargs = _room_and_kwargs()
        await main._acquire_sync_group(room, kwargs)

        _room2, changed = _room_and_kwargs(fps=12)
        got = await main._acquire_sync_group(room, changed)
        assert got is None

    asyncio.run(go())


def test_sync_group_accepts_mismatched_channels():
    # Unlike fps/codec/etc, a differing `channels` request is NOT a mismatch --
    # it's merged into the room's union (spec: serve the union, not reject).
    async def go():
        main._sync_groups.clear()
        room1, kwargs1 = _room_and_kwargs()
        g1 = await main._acquire_sync_group(room1, kwargs1)

        room2 = ccmf.parse_room(ccmf.build_room("u", sync=True))
        caps2 = ccmf.parse_caps(ccmf.build_caps(
            fps=24, channels=ccmf.CAP_CHANNEL_MONO | ccmf.CAP_CHANNEL_FRONT_LEFT
            | ccmf.CAP_CHANNEL_FRONT_RIGHT))
        kwargs2 = main._negotiate(room2, caps2)
        g2 = await main._acquire_sync_group(room2, kwargs2)
        assert g1 is g2

    asyncio.run(go())


class _FakeSession:
    """Stand-in for StreamSession: records what _SyncGroup tells it to do,
    without needing a real run() loop (source probing, ffmpeg, etc)."""

    def __init__(self):
        self.requested_channels = 0
        self.reconcile_calls = 0
        self.use_ans = False

    async def reconfigure_channels(self):
        self.reconcile_calls += 1

    def set_use_ans(self, use_ans):
        self.use_ans = use_ans

    def playback_origin(self):
        return None                      # not playing yet: no anchor to send


def test_sync_group_reconciles_channel_union_on_join_and_leave():
    # The example from the feature request: one subscriber wants mono+lfe,
    # another wants front_left+front_right -- the group must union both, and
    # re-derive the union (dropping what's now unwanted) when one leaves.
    async def go():
        fake_session = _FakeSession()
        group = main._SyncGroup(("u", None, None, False), (), fake_session)
        ws1, ws2 = object(), object()

        await group.subscribe(ws1, ccmf.CAP_CHANNEL_MONO | ccmf.CAP_CHANNEL_LFE, False)
        await asyncio.sleep(0.01)   # let the fire-and-forget reconcile task run
        assert fake_session.requested_channels == ccmf.CAP_CHANNEL_MONO | ccmf.CAP_CHANNEL_LFE
        assert fake_session.reconcile_calls == 1

        await group.subscribe(ws2, ccmf.CAP_CHANNEL_FRONT_LEFT | ccmf.CAP_CHANNEL_FRONT_RIGHT, False)
        await asyncio.sleep(0.01)
        assert fake_session.requested_channels == (
            ccmf.CAP_CHANNEL_MONO | ccmf.CAP_CHANNEL_LFE
            | ccmf.CAP_CHANNEL_FRONT_LEFT | ccmf.CAP_CHANNEL_FRONT_RIGHT)
        assert fake_session.reconcile_calls == 2

        # ws1 leaves: mono+lfe drop out of the union, front_left/front_right remain.
        empty = await group.unsubscribe(ws1)
        assert not empty
        await asyncio.sleep(0.01)
        assert fake_session.requested_channels == (
            ccmf.CAP_CHANNEL_FRONT_LEFT | ccmf.CAP_CHANNEL_FRONT_RIGHT)
        assert fake_session.reconcile_calls == 3

    asyncio.run(go())


def test_sync_group_turns_ans_off_for_incompatible_member_and_back_on():
    # ANS stays on only while every subscriber can decode it: an ANS-incapable
    # joiner forces the room back to packed keyframes; when it leaves, ANS
    # resumes for the remaining (ANS-capable) members.
    async def go():
        fake_session = _FakeSession()
        fake_session.use_ans = True                 # room opened ANS
        group = main._SyncGroup(("u", None, None, False), (), fake_session)
        ws_ans, ws_plain = object(), object()

        await group.subscribe(ws_ans, ccmf.CAP_CHANNEL_MONO, True)
        assert fake_session.use_ans is True          # all members ANS-capable

        await group.subscribe(ws_plain, ccmf.CAP_CHANNEL_MONO, False)
        assert fake_session.use_ans is False         # one can't -> packed for all

        empty = await group.unsubscribe(ws_plain)
        assert not empty
        assert fake_session.use_ans is True          # incompatible member gone -> resume

    asyncio.run(go())


def test_sync_group_anchors_late_joiner_to_running_clock():
    # A mid-stream joiner missed the room's STATUS playing, so subscribe()
    # must unicast one carrying the session's CURRENT clock position — the
    # joiner anchors to the room's clock instead of guessing from arrivals.
    class _PlayingSession(_FakeSession):
        def playback_origin(self):
            return 96_000                # 2 s in, in 48 kHz samples

    class _WS:
        def __init__(self):
            self.sent = []

        async def send_bytes(self, b):
            self.sent.append(b)

    async def go():
        group = main._SyncGroup(("u", None, None, False), (), _PlayingSession())
        ws = _WS()
        await group.subscribe(ws, ccmf.CAP_CHANNEL_MONO, False)
        statuses = [ccmf.parse_status(body) for op, body in
                    (ccmf.parse_message(b) for b in ws.sent)
                    if op == ccmf.OP_STATUS]
        assert (ccmf.STATUS_PLAYING, 96_000) in statuses

        # Before playback starts (origin None) nothing is sent.
        group2 = main._SyncGroup(("u", None, None, False), (), _FakeSession())
        ws2 = _WS()
        await group2.subscribe(ws2, ccmf.CAP_CHANNEL_MONO, False)
        assert ws2.sent == []

    asyncio.run(go())


def test_sync_rooms_differ_by_section_not_by_caps_only():
    # (url, start, end, loop) is the room key: a different section is a
    # DIFFERENT room (its own production), not a mismatch in the same room.
    async def go():
        main._sync_groups.clear()
        room1, kwargs1 = _room_and_kwargs()
        room2, kwargs2 = _room_and_kwargs(start_ms=60_000)
        g1 = await main._acquire_sync_group(room1, kwargs1)
        g2 = await main._acquire_sync_group(room2, kwargs2)
        assert g1 is not g2
        assert len(main._sync_groups) == 2

    asyncio.run(go())

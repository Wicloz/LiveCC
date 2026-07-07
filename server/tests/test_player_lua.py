"""
The player.lua test suite — the client half of the sync contract.

Runs the REAL client script under embedded Lua 5.1 (lupa; CC:Tweaked's Cobalt
is 5.1) with every CC API stubbed (cc_stubs.lua), then drives the internals
the script exports under _LIVECC_TEST.  This half of the system had zero test
coverage while the server side had two hundred tests — and client-side
regressions (speaker routing, PTS gating) shipped invisibly.  Chunks and
control frames are built with the server's own ccmf module, so both ends of
the wire are pinned to the same bytes.

Conventions: the harness speaks raw bytes to Lua (LuaRuntime(encoding=None)),
audio levels are chosen so a mean identifies which role a speaker played
(u8 160 = front_left content, 224 = front_right, 192 = the mono downmix), and
the clock helper positions os.epoch so a given PTS is exactly "now".
"""

from pathlib import Path

import numpy as np
import pytest

lupa = pytest.importorskip("lupa.lua51", reason="lupa (Lua 5.1) not installed")

import ccmf

_REPO = Path(__file__).resolve().parent.parent.parent
_PLAYER_SRC = (_REPO / "lua" / "player.lua").read_bytes().replace(
    b'"{{SERVER}}"', b'"http://srv.test"')
_STUBS = (Path(__file__).parent / "cc_stubs.lua").read_bytes()

_ACK = ccmf.control(ccmf.OP_ACK)
_STEREO_MAP = b'{ left = "front_left", right = "front_right" }'

CHUNK = 4800                       # samples per test audio chunk (0.1 s)
CLIENT_DELAY_MS = 300              # player.lua's fixed anchor slack
SPK_AHEAD = 38400                  # player.lua's speaker feed budget (0.8 s)


class Player:
    """One loaded player.lua instance plus its stubbed world."""

    def __init__(self, lua, stub, export):
        self._lua, self._stub, self._ex = lua, stub, export
        self._anchor_wall = None   # now_ms when the anchor was set

    def __getattr__(self, name):
        val = self._ex[name.encode()]
        if val is None:
            raise AttributeError(name)
        return val

    # ----- stubbed world ---------------------------------------------------

    @property
    def stub(self):
        return self._stub

    def sent(self):
        return [v for v in self._stub[b"ws_sent"].values()]

    def console_lines(self):
        return [v.decode() for v in self._stub[b"console"].values()]

    def stats(self, speaker, i):
        """(sample count, mean) of `speaker`'s i-th played buffer."""
        return self._stub[b"stats"](speaker, i)

    def play_count(self, speaker):
        return self._stub[b"play_count"](speaker)

    def stop_count(self, speaker):
        return self._stub[b"stop_count"](speaker)

    def set_accept(self, speaker, accept):
        self._stub[b"speakers"][speaker][b"accept"] = accept

    # ----- clock -----------------------------------------------------------

    def anchor(self, origin):
        """Deliver STATUS playing(origin); remembers the wall time so
        at_clock() can position the clock exactly."""
        self._anchor_wall = self._stub[b"now_ms"]
        self.handle_message(
            ccmf.control(ccmf.OP_STATUS,
                         ccmf.status_body(ccmf.STATUS_PLAYING, origin)))
        self._origin = origin

    def at_clock(self, samples):
        """Advance os.epoch so the media clock reads exactly `samples`."""
        assert self._anchor_wall is not None, "anchor() first"
        assert (samples - self._origin) % 48 == 0, "pick a multiple of 48"
        self._stub[b"now_ms"] = (self._anchor_wall + CLIENT_DELAY_MS
                                 + (samples - self._origin) // 48)


def load_player(speakers=(b"left", b"right"), speaker_map=_STEREO_MAP,
                args=(b"http://media.test/v",), monitor_present=True):
    lua = lupa.LuaRuntime(encoding=None)
    lua.execute(_STUBS)
    g = lua.globals()
    stub = g[b"stub"]
    stub[b"monitor"][b"present"] = monitor_present
    for name in speakers:
        stub[b"add_speaker"](name)
    if speaker_map is not None:
        stub[b"files"][b"livecc.speakers"] = speaker_map
    stub[b"ws_inbox"][1] = _ACK    # ROOM ack
    stub[b"ws_inbox"][2] = _ACK    # CAPS ack
    g[b"_LIVECC_TEST"] = True
    g[b"PLAYER_SRC"] = _PLAYER_SRC
    chunk = lua.eval(b"loadstring(PLAYER_SRC, 'player.lua')")
    assert chunk is not None, "player.lua failed to parse"
    export = chunk(*args)
    assert export is not None, "test gate missing: player.lua returned nothing"
    stub[b"reset_speakers"]()      # drop the dedupe probe's silence bursts
    return Player(lua, stub, export)


def audio_chunk(pts, role, level, n=CHUNK, codec=ccmf.CODEC_PCM8):
    data = bytes([level]) * (n if codec == ccmf.CODEC_PCM8 else n // 8)
    return ccmf.chunk(pts, ccmf.TYPE_AUDIO,
                      ccmf.audio_payload(codec, data, channel=role))


def video_gop(pts, cell=(0x81, 1, 0), w=2, h=1, extra_delta=True):
    glyph = np.full((h, w), cell[0], np.uint8)
    fg = np.full((h, w), cell[1], np.uint8)
    bg = np.full((h, w), cell[2], np.uint8)
    units = ccmf.palette_unit(bytes(48)) + ccmf.raw_frame_unit(2000, glyph, fg, bg)
    if extra_delta:
        changed = glyph.copy()
        changed[0, 0] = 0x9F
        units += ccmf.delta_frame_unit(
            2000, ccmf.delta_spans((glyph, fg, bg), (changed, fg, bg)))
    return ccmf.chunk(pts, ccmf.TYPE_VIDEO, ccmf.video_payload(w, h, units))


# --------------------------------------------------------------------------- #
# Handshake + capability negotiation
# --------------------------------------------------------------------------- #

def test_speaker_map_advertises_positional_roles_in_caps():
    # The client half of the stereo contract: a mapped stereo pair must ask
    # the server for mono|front_left|front_right — if this mask collapses,
    # the server (correctly) sends mono only and "stereo goes mono".
    p = load_player()
    sent = p.sent()
    ops = [ccmf.parse_message(m)[0] for m in sent]
    assert ops == [ccmf.OP_ROOM, ccmf.OP_CAPS, ccmf.OP_START]
    caps = ccmf.parse_caps(ccmf.parse_message(sent[1])[1])
    assert caps.channels == (ccmf.CAP_CHANNEL_MONO | ccmf.CAP_CHANNEL_FRONT_LEFT
                             | ccmf.CAP_CHANNEL_FRONT_RIGHT)
    assert caps.want_audio and caps.want_video
    assert (caps.width, caps.height) == (20, 10)      # the stub monitor's size
    room = ccmf.parse_room(ccmf.parse_message(sent[0])[1])
    assert room.url == "http://media.test/v"


def test_no_map_advertises_mono_only():
    p = load_player(speaker_map=None)
    caps = ccmf.parse_caps(ccmf.parse_message(p.sent()[1])[1])
    assert caps.channels == ccmf.CAP_CHANNEL_MONO
    assert int(p.channels_mask()) == 1


# --------------------------------------------------------------------------- #
# Audio routing: the multichannel core
# --------------------------------------------------------------------------- #

def test_stereo_roles_route_to_their_mapped_speakers():
    # Distinct wire roles must land on distinct speakers — and the mono
    # downmix (role 0) must NOT play anywhere when a map claims no speaker
    # for it.  This is the client half of the "both speakers play the same
    # mono" regression.
    p = load_player()
    p.anchor(0)
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_FRONT_LEFT, 160))
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_FRONT_RIGHT, 224))
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_MONO, 192))
    p.at_clock(0)
    p.feed_roles()

    assert p.play_count(b"left") == 1
    assert p.play_count(b"right") == 1
    n, mean = p.stats(b"left", 1)
    assert (n, mean) == (CHUNK, 160 - 128)     # front_left content, u8 -> signed
    n, mean = p.stats(b"right", 1)
    assert (n, mean) == (CHUNK, 224 - 128)     # front_right content — not mono!


def test_mono_wire_with_map_stays_silent():
    # Documented behaviour: a map with no mono speaker plays nothing when the
    # server only produces role 0 (it must not silently fall back).
    p = load_player()
    p.anchor(0)
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_MONO, 192))
    p.at_clock(0)
    p.feed_roles()
    assert p.play_count(b"left") == 0
    assert p.play_count(b"right") == 0
    assert not p.audio_pending()               # dropped, not queued forever


def test_no_map_plays_mono_on_every_speaker():
    p = load_player(speaker_map=None)
    p.anchor(0)
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_MONO, 192))
    p.at_clock(0)
    p.feed_roles()
    for spk in (b"left", b"right"):
        n, mean = p.stats(spk, 1)
        assert (n, mean) == (CHUNK, 192 - 128)


def test_full_surround_map_routes_every_role_to_its_speaker():
    # A 7.1 + mono map (nine speakers): CAPS must advertise all nine role
    # bits, and each wire role must land on exactly its own speaker — no
    # bleed, no substitution.  The server may well send byte-identical
    # content on some roles (fallback mixes for a narrow source); routing
    # must still be strictly by role id.
    roles = {b"s_mono": (b"mono", 0), b"s_fl": (b"front_left", 1),
             b"s_fr": (b"front_right", 2), b"s_c": (b"center", 3),
             b"s_lfe": (b"lfe", 4), b"s_sl": (b"surround_left", 5),
             b"s_sr": (b"surround_right", 6), b"s_rl": (b"rear_left", 7),
             b"s_rr": (b"rear_right", 8)}
    map_src = b"{ " + b", ".join(
        name + b' = "' + role + b'"' for name, (role, _) in roles.items()) + b" }"
    p = load_player(speakers=tuple(roles), speaker_map=map_src)

    caps = ccmf.parse_caps(ccmf.parse_message(p.sent()[1])[1])
    assert caps.channels == 0b111111111          # mono + all eight positional

    p.anchor(0)
    for _name, (_role, rid) in roles.items():
        p.handle_message(audio_chunk(0, rid, 130 + rid * 10))
    p.at_clock(0)
    p.feed_roles()
    for name, (_role, rid) in roles.items():
        assert p.play_count(name) == 1, name
        n, mean = p.stats(name, 1)
        assert (n, mean) == (CHUNK, 130 + rid * 10 - 128), name


def test_dfpwm_chunks_are_decoded_per_chunk():
    # Codec nibble 1 must route through a fresh DFPWM decode (the stub
    # decoder yields the recognisable constant 17) and derive the sample
    # count from the 8-samples-per-byte packing.
    p = load_player()
    p.anchor(0)
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_FRONT_LEFT, 0,
                                 codec=ccmf.CODEC_DFPWM))
    p.at_clock(0)
    p.feed_roles()
    n, mean = p.stats(b"left", 1)
    assert (n, mean) == (CHUNK, 17)


# --------------------------------------------------------------------------- #
# Audio pacing: due / budget / stale / holes / refusal
# --------------------------------------------------------------------------- #

def test_audio_waits_for_its_pts_then_plays():
    # Delivery is not presentation (spec §5.6): an early chunk must sit in
    # the queue until the clock reaches its PTS.
    p = load_player()
    p.anchor(0)
    p.handle_message(audio_chunk(48000, ccmf.CHANNEL_FRONT_LEFT, 160))
    p.at_clock(0)
    wait = p.feed_roles()
    assert p.play_count(b"left") == 0
    assert wait == pytest.approx(1000)         # due in exactly one second
    p.at_clock(48000)
    p.feed_roles()
    assert p.play_count(b"left") == 1


def test_stale_audio_is_dropped_and_straddler_trimmed():
    p = load_player()
    p.anchor(0)
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_FRONT_LEFT, 160))       # fully past
    p.handle_message(audio_chunk(CHUNK, ccmf.CHANNEL_FRONT_LEFT, 160))   # straddles
    p.at_clock(CHUNK + 2400)                   # halfway through the second chunk
    p.feed_roles()
    assert p.play_count(b"left") == 1          # the stale chunk never played
    n, _mean = p.stats(b"left", 1)
    assert n == 2400                           # only the remaining half played


def test_budget_gate_stops_feeding_past_spk_ahead():
    # Once rolling, the speaker may hold at most SPK_AHEAD of future audio —
    # more would turn delivery lead back into audible lead.
    # feed_roles() admits/decodes at most one due chunk's worth of work per
    # call now (each CHUNK-sized message here fits in one decode slice), so
    # draining a backlog of already-admissible chunks takes one call each.
    p = load_player()
    p.anchor(0)
    for i in range(12):
        p.handle_message(audio_chunk(i * CHUNK, ccmf.CHANNEL_FRONT_LEFT, 160))
    p.at_clock(0)
    wait = None
    for _ in range(12):
        wait = p.feed_roles()
    # pts 0..SPK_AHEAD inclusive = 9 chunks; the 10th (43200) is over budget
    assert p.play_count(b"left") == 9
    assert wait == pytest.approx(100)          # budget frees up in 0.1 s
    p.at_clock(4800)
    p.feed_roles()
    assert p.play_count(b"left") == 10


def test_big_chunk_is_sliced_across_multiple_feed_calls():
    # A chunk bigger than one decode slice (AUDIO_SLICE_SAMPLES, == CHUNK here)
    # must not be decoded/fed in one feed_roles() call -- that's the whole
    # point of slicing: bound the per-call decode/render-stall to one slice
    # regardless of how big the wire chunk is.
    p = load_player()
    p.anchor(0)
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_FRONT_LEFT, 160, n=CHUNK * 3))
    p.at_clock(0)

    wait = p.feed_roles()
    assert p.play_count(b"left") == 1          # only the first slice fed so far
    assert wait == pytest.approx(0)             # more of this chunk: come straight back
    n, mean = p.stats(b"left", 1)
    assert (n, mean) == (CHUNK, 160 - 128)

    p.feed_roles()
    assert p.play_count(b"left") == 2
    wait = p.feed_roles()
    assert p.play_count(b"left") == 3           # fully drained
    assert wait is None or wait != pytest.approx(0)
    total = sum(p.stats(b"left", i)[0] for i in range(1, 4))
    assert total == CHUNK * 3


def test_dfpwm_big_chunk_reuses_one_decoder_across_slices():
    # spec §4.6: DFPWM state resets per CHUNK, not per slice -- a chunk that
    # takes several feed_roles() calls to drain must still share ONE decoder
    # instance across all of them (cc.audio.dfpwm's decoder is built to be
    # called repeatedly on successive byte ranges of the same stream).
    p = load_player()
    p.anchor(0)
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_FRONT_LEFT, 0,
                                 n=CHUNK * 3, codec=ccmf.CODEC_DFPWM))
    p.at_clock(0)
    for _ in range(3):
        p.feed_roles()

    assert p.play_count(b"left") == 3
    assert p.stub[b"dfpwm_decoders"] == 1        # one decoder for the whole chunk
    calls = list(p.stub[b"dfpwm_decode_calls"].values())
    assert calls == [1, 1, 1]                    # every slice decoded by it


def test_hole_in_role_stream_waits_for_drain_then_due():
    # A gap must NOT be fed early (the speaker would play it back-to-back,
    # early by the hole's width).  It plays exactly at its own PTS after the
    # device drains — the gap is heard as silence, never as a shift.
    p = load_player()
    p.anchor(0)
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_FRONT_LEFT, 160))
    p.at_clock(0)
    p.feed_roles()
    assert p.play_count(b"left") == 1
    p.handle_message(audio_chunk(43200, ccmf.CHANNEL_FRONT_LEFT, 160))   # hole
    p.feed_roles()
    assert p.play_count(b"left") == 1          # not fed while device still full
    p.at_clock(9600)                           # drained (fed_until=4800 < clock)…
    p.feed_roles()
    assert p.play_count(b"left") == 1          # …but the chunk is not due yet
    p.at_clock(43200)
    p.feed_roles()
    assert p.play_count(b"left") == 2          # restarts exactly on the clock


def test_refused_speaker_retries_in_order():
    # playAudio() returning false must queue a retry and block newer chunks
    # for that role — feeding out of order would permanently shift the role.
    p = load_player()
    p.anchor(0)
    p.set_accept(b"left", False)
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_FRONT_LEFT, 140))
    p.handle_message(audio_chunk(CHUNK, ccmf.CHANNEL_FRONT_LEFT, 200))
    p.at_clock(0)
    p.feed_roles()
    assert p.play_count(b"left") == 0
    assert p.audio_pending()
    p.feed_roles()                             # still refused: still pending
    assert p.play_count(b"left") == 0
    p.set_accept(b"left", True)
    p.feed_roles()
    assert p.play_count(b"left") == 2          # retry, then the follow-up
    assert p.stats(b"left", 1)[1] == 140 - 128 # in PTS order
    assert p.stats(b"left", 2)[1] == 200 - 128


def test_refusal_on_one_role_does_not_block_the_other():
    p = load_player()
    p.anchor(0)
    p.set_accept(b"left", False)
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_FRONT_LEFT, 160))
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_FRONT_RIGHT, 224))
    p.at_clock(0)
    p.feed_roles()
    assert p.play_count(b"left") == 0
    assert p.play_count(b"right") == 1


# --------------------------------------------------------------------------- #
# The clock: anchoring, jumps, fallback
# --------------------------------------------------------------------------- #

def test_status_playing_origin_anchors_the_clock():
    p = load_player()
    p.anchor(96000)
    p.at_clock(96000)
    assert p.clock_now() == pytest.approx(96000)
    pts, _ms = p.get_anchor()
    assert pts == 96000


def test_first_chunk_anchors_when_server_sends_no_origin():
    # draft-00 fallback: no STATUS origin ever — the first media chunk's PTS
    # becomes the anchor instead of the client sitting clockless forever.
    p = load_player()
    p.handle_message(audio_chunk(5000, ccmf.CHANNEL_FRONT_LEFT, 160))
    pts, _ms = p.get_anchor()
    assert pts == 5000


def test_forward_clock_jump_flushes_speakers_and_prunes_video():
    # A live-edge skip re-anchors ahead: buffered device audio is stale (it
    # was timed against the old clock) and queued video before the new origin
    # is stale except the keyframe the next frames chain from.
    p = load_player()
    p.anchor(0)
    p.handle_message(video_gop(0))
    p.handle_message(video_gop(96000))
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_FRONT_LEFT, 160))
    p.at_clock(0)
    p.feed_roles()
    assert p.play_count(b"left") == 1

    p.anchor(96000)                            # 2 s jump forward
    assert p.stop_count(b"left") == 1          # device flushed
    assert p.stop_count(b"right") == 1
    vq = p.get_vqueue()
    assert vq[1][b"palette"] is not None       # resumes at GOP2's palette…
    assert vq[1][b"pts"] == 96000              # …not GOP1's stale units
    p.at_clock(96000)
    p.feed_roles()
    assert p.play_count(b"left") == 1          # stale audio dropped, not played


def test_status_ended_marks_the_stream_over():
    p = load_player()
    assert not p.is_ended()
    p.handle_message(ccmf.control(ccmf.OP_STATUS,
                                  ccmf.status_body(ccmf.STATUS_ENDED)))
    assert p.is_ended()


# --------------------------------------------------------------------------- #
# Video: parsing + due rendering
# --------------------------------------------------------------------------- #

def test_video_units_render_only_when_due():
    p = load_player()
    p.anchor(0)
    p.handle_message(video_gop(0))             # raw at 0, delta at 2000
    p.at_clock(0)
    p.drain_due()
    blits = p.stub[b"blits"]
    assert len(blits) == 1                     # the keyframe row (2x1 grid)
    assert len(blits[1][b"text"]) == 2         # both cells in one blit
    assert int(p.stub[b"palettes"]) == 16      # GOP palette applied
    assert int(p.stub[b"flushes"]) == 1        # one atomic redraw

    p.drain_due()
    assert len(blits) == 1                     # delta (pts 2000) not due yet

    p.at_clock(2400)                           # 50 ms later: delta due
    p.drain_due()
    assert len(blits) == 2
    assert len(blits[2][b"text"]) == 1         # the single changed cell


# --------------------------------------------------------------------------- #
# --verbose: chunk header logging
# --------------------------------------------------------------------------- #

def test_verbose_prints_video_and_audio_chunk_headers():
    p = load_player(args=(b"http://media.test/v", b"--verbose"))
    p.handle_message(video_gop(1234))
    p.handle_message(audio_chunk(1234, ccmf.CHANNEL_FRONT_LEFT, 160))
    lines = p.console_lines()

    video_lines = [l for l in lines if l.startswith("LiveCC: [video]")]
    assert len(video_lines) == 1
    assert "pts=1234" in video_lines[0]
    assert "2x1" in video_lines[0]
    assert "compression=none" in video_lines[0]
    # video_gop() emits palette + raw(2000) + delta(2000, 1 changed cell)
    assert "units=1(palette)+1(raw)+1(delta,1 spans)+0(repeat)" in video_lines[0]
    assert "gop_dur=0.08s" in video_lines[0]        # 4000 samples / 48000

    audio_lines = [l for l in lines if l.startswith("LiveCC: [audio]")]
    assert len(audio_lines) == 1
    assert "pts=1234" in audio_lines[0]
    assert "codec=pcm8" in audio_lines[0]
    assert "role=front_left" in audio_lines[0]
    assert f"samples={CHUNK}" in audio_lines[0]


def test_verbose_reports_dfpwm_codec_and_role_by_name():
    p = load_player(args=(b"http://media.test/v", b"--verbose"))
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_SURROUND_LEFT, 0,
                                 codec=ccmf.CODEC_DFPWM))
    audio_lines = [l for l in p.console_lines() if l.startswith("LiveCC: [audio]")]
    assert len(audio_lines) == 1
    assert "codec=dfpwm" in audio_lines[0]
    assert "role=surround_left" in audio_lines[0]
    assert f"samples={CHUNK}" in audio_lines[0]     # 8 samples/byte, CHUNK//8 bytes


def test_without_verbose_no_chunk_headers_are_logged():
    p = load_player()                          # no --verbose
    p.handle_message(video_gop(0))
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_FRONT_LEFT, 160))
    lines = p.console_lines()
    assert not any(l.startswith("LiveCC: [video]") or l.startswith("LiveCC: [audio]")
                   for l in lines)


def test_verbose_is_silenced_when_terminal_is_the_display():
    # No monitor: the terminal itself is the video surface, so verbose (like
    # every other status line) is routed through console() and silenced —
    # printing there would just be overwritten by the next frame render.
    p = load_player(monitor_present=False, args=(b"http://media.test/v", b"--verbose"))
    p.handle_message(video_gop(0))
    p.handle_message(audio_chunk(0, ccmf.CHANNEL_FRONT_LEFT, 160))
    lines = p.console_lines()
    assert not any(l.startswith("LiveCC: [video]") or l.startswith("LiveCC: [audio]")
                   for l in lines)

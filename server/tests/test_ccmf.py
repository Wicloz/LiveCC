"""
CCMF wire format (ccmf.py) + GOP encoder (cc_encoder.GopEncoder).

Byte-level packing/parsing roundtrips against the spec (docs/cc-media-format.md),
plus the encoder-side GOP policy: every chunk opens palette + raw keyframe,
unchanged frames become repeats, small changes become deltas, scene cuts re-key
mid-GOP, and durations are the true PTS gaps.
"""

import numpy as np
import pytest

import ccmf
from cc_encoder import GopEncoder


def _grids(w, h, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0x80, 0xA0, (h, w), dtype=np.uint8),   # glyph
            rng.integers(0, 16, (h, w), dtype=np.uint8),         # fg
            rng.integers(0, 16, (h, w), dtype=np.uint8))         # bg


def _solid_frame(rgb, w, h):
    return np.tile(np.array(rgb, dtype=np.uint8), (h * 3, w * 2, 1))


# --------------------------------------------------------------------------- #
# Chunk framing
# --------------------------------------------------------------------------- #

def test_chunk_header_layout():
    payload = bytes(range(20))
    c = ccmf.chunk(pts=0x0123456789AB, ctype=ccmf.TYPE_AUDIO, payload=payload)
    assert c[0] == 0x43                                  # marker "C"
    assert c[1:7] == (0x0123456789AB).to_bytes(6, "little")
    assert c[7:10] == (20).to_bytes(3, "little")         # u24 payload length
    assert c[10] == ccmf.TYPE_AUDIO
    assert c[11] == ccmf.COMPRESSION_NONE
    assert c[12:] == payload                             # header is exactly 12 B

    pts, ctype, body, end = ccmf.parse_chunk(c)
    assert (pts, ctype, body, end) == (0x0123456789AB, ccmf.TYPE_AUDIO, payload, len(c))


def test_chunks_chain_back_to_back():
    buf = (ccmf.chunk(0, ccmf.TYPE_VIDEO, b"aa")
           + ccmf.chunk(2000, ccmf.TYPE_AUDIO, b"bbb")
           + ccmf.chunk(4000, ccmf.TYPE_VIDEO, b""))
    out = list(ccmf.iter_chunks(buf))
    assert [(p, t, len(d)) for p, t, d in out] == \
        [(0, 0, 2), (2000, 1, 3), (4000, 0, 0)]


def test_chunk_rejects_out_of_range_fields():
    with pytest.raises(ValueError):
        ccmf.chunk(1 << 48, ccmf.TYPE_VIDEO, b"")        # PTS is u48
    with pytest.raises(ValueError):
        ccmf.parse_chunk(b"X" + bytes(11))               # bad marker (12 B, marker != C)


# --------------------------------------------------------------------------- #
# Raw plane packing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [1, 7, 8, 9, 16, 41])
def test_pack_chars_roundtrip(n):
    rng = np.random.default_rng(n)
    glyph = rng.integers(0x80, 0xA0, n, dtype=np.uint8)
    packed = ccmf.pack_chars(glyph)
    assert len(packed) == ((n + 7) // 8) * 5             # 8 cells -> 5 bytes
    assert np.array_equal(ccmf.unpack_chars(packed, n), glyph)


@pytest.mark.parametrize("n", [1, 2, 3, 8, 41])
def test_pack_nibbles_roundtrip_high_nibble_first(n):
    rng = np.random.default_rng(n)
    idx = rng.integers(0, 16, n, dtype=np.uint8)
    packed = ccmf.pack_nibbles(idx)
    assert len(packed) == (n + 1) // 2
    assert packed[0] >> 4 == idx[0]                      # cell 0 in the high nibble
    assert np.array_equal(ccmf.unpack_nibbles(packed, n), idx)


# --------------------------------------------------------------------------- #
# Units + video payload
# --------------------------------------------------------------------------- #

def test_video_payload_roundtrip_raw():
    w, h = 6, 4
    glyph, fg, bg = _grids(w, h)
    pal = bytes(range(48))
    units = ccmf.palette_unit(pal) + ccmf.raw_frame_unit(2000, glyph, fg, bg)
    dw, dh, frames = ccmf.parse_video_payload(ccmf.video_payload(w, h, units))
    assert (dw, dh) == (w, h)
    assert len(frames) == 1
    f = frames[0]
    assert f.encoding == ccmf.ENC_RAW and f.duration == 2000
    assert np.array_equal(f.glyph, glyph)
    assert np.array_equal(f.fg, fg)
    assert np.array_equal(f.bg, bg)
    assert f.palette.tobytes() == pal


def test_delta_spans_roundtrip():
    w, h = 8, 5
    a = _grids(w, h, seed=1)
    b = tuple(g.copy() for g in a)
    b[0][1, 2] = 0x9F                       # one glyph change
    b[1][1, 3] = 5                          # fg change in the adjacent cell
    b[2][4, 7] = 9                          # bg change at the last cell
    body = ccmf.delta_spans(a, b)
    assert body is not None

    units = (ccmf.palette_unit(bytes(48)) + ccmf.raw_frame_unit(100, *a)
             + ccmf.delta_frame_unit(100, body))
    _w, _h, frames = ccmf.parse_video_payload(ccmf.video_payload(w, h, units))
    assert frames[1].encoding == ccmf.ENC_DELTA
    assert np.array_equal(frames[1].glyph, b[0])
    assert np.array_equal(frames[1].fg, b[1])
    assert np.array_equal(frames[1].bg, b[2])


def test_delta_spans_none_when_identical():
    a = _grids(4, 3)
    assert ccmf.delta_spans(a, tuple(g.copy() for g in a)) is None


def test_delta_spans_never_cross_rows():
    w, h = 6, 3
    a = _grids(w, h, seed=2)
    b = tuple(g.copy() for g in a)
    b[0][0, :] = 0x81                       # change all of row 0
    b[0][1, :] = 0x82                       # ...and all of row 1
    body = ccmf.delta_spans(a, b)
    (count,) = np.frombuffer(body[:2], np.uint16)
    pos = 2
    for _ in range(count):
        start = int.from_bytes(body[pos:pos + 2], "little")
        length = body[pos + 2]
        assert start % w + length <= w      # spec §4.5.2 MUST
        pos += 3 + length * 2


def test_repeat_unit_and_palette_only_change():
    # A pure recolour (e.g. a fade) needs no NEW cell content -- but it still
    # needs a frame unit right after the palette to give it an effective time
    # (spec §4.4/§4.5): here that's a `repeat`, so the screen recolours with
    # no redraw.
    w, h = 4, 2
    g = _grids(w, h)
    pal1, pal2 = bytes(48), bytes(range(48))
    units = (ccmf.palette_unit(pal1) + ccmf.raw_frame_unit(100, *g)
             + ccmf.palette_unit(pal2) + ccmf.repeat_frame_unit(200))
    _w, _h, frames = ccmf.parse_video_payload(ccmf.video_payload(w, h, units))
    assert [f.encoding for f in frames] == [ccmf.ENC_RAW, ccmf.ENC_REPEAT]
    assert frames[0].palette.tobytes() == pal1
    assert frames[1].palette.tobytes() == pal2   # palette applied without a redraw
    assert np.array_equal(frames[1].glyph, frames[0].glyph)


def test_video_payload_must_not_end_with_palette():
    units = ccmf.palette_unit(bytes(48))
    with pytest.raises(ValueError):
        ccmf.parse_video_payload(ccmf.video_payload(2, 2, units))


def test_video_payload_must_not_end_with_palette_after_a_valid_frame():
    # The bare "only a palette" case above isn't the only way to end on one --
    # a dangling palette AFTER legitimate frames is just as undefined (spec
    # §4.4): it has no frame to borrow an effective time from.
    w, h = 4, 2
    g = _grids(w, h)
    units = (ccmf.palette_unit(bytes(48)) + ccmf.raw_frame_unit(100, *g)
             + ccmf.palette_unit(bytes(range(48))))
    with pytest.raises(ValueError):
        ccmf.parse_video_payload(ccmf.video_payload(w, h, units))


def test_video_payload_palette_must_not_be_followed_by_another_palette():
    w, h = 4, 2
    g = _grids(w, h)
    units = (ccmf.palette_unit(bytes(48)) + ccmf.palette_unit(bytes(range(48)))
             + ccmf.raw_frame_unit(100, *g))
    with pytest.raises(ValueError):
        ccmf.parse_video_payload(ccmf.video_payload(w, h, units))


def test_audio_payload_roundtrip():
    p = ccmf.audio_payload(ccmf.CODEC_DFPWM, b"\x55" * 600)
    codec, channel, data = ccmf.parse_audio_payload(p)
    assert (codec, channel) == (ccmf.CODEC_DFPWM, ccmf.CHANNEL_MONO)
    assert data == b"\x55" * 600


def test_audio_payload_roundtrip_positional_channel():
    p = ccmf.audio_payload(ccmf.CODEC_PCM8, b"\x80" * 100, channel=ccmf.CHANNEL_FRONT_RIGHT)
    codec, channel, data = ccmf.parse_audio_payload(p)
    assert (codec, channel) == (ccmf.CODEC_PCM8, ccmf.CHANNEL_FRONT_RIGHT)
    assert data == b"\x80" * 100


def test_channel_roles_are_distinct_one_hot_bits():
    # Spec §4.6/§5.4: bit N of CAPS `channels` accepts channel role N.
    roles = (ccmf.CHANNEL_MONO, ccmf.CHANNEL_FRONT_LEFT, ccmf.CHANNEL_FRONT_RIGHT,
             ccmf.CHANNEL_CENTER, ccmf.CHANNEL_LFE, ccmf.CHANNEL_SURROUND_LEFT,
             ccmf.CHANNEL_SURROUND_RIGHT, ccmf.CHANNEL_REAR_LEFT, ccmf.CHANNEL_REAR_RIGHT)
    assert roles == tuple(range(9))
    caps = (ccmf.CAP_CHANNEL_MONO, ccmf.CAP_CHANNEL_FRONT_LEFT, ccmf.CAP_CHANNEL_FRONT_RIGHT,
            ccmf.CAP_CHANNEL_CENTER, ccmf.CAP_CHANNEL_LFE, ccmf.CAP_CHANNEL_SURROUND_LEFT,
            ccmf.CAP_CHANNEL_SURROUND_RIGHT, ccmf.CAP_CHANNEL_REAR_LEFT, ccmf.CAP_CHANNEL_REAR_RIGHT)
    for role, cap in zip(roles, caps):
        assert cap == 1 << role


# --------------------------------------------------------------------------- #
# Stream format: control frames + HELLO bodies
# --------------------------------------------------------------------------- #

def test_status_body_playing_carries_origin():
    body = ccmf.status_body(ccmf.STATUS_PLAYING, 123_456_789)
    assert len(body) == 7                       # state u8 + origin u48
    assert ccmf.parse_status(body) == (ccmf.STATUS_PLAYING, 123_456_789)


def test_status_body_buffering_is_bare():
    body = ccmf.status_body(ccmf.STATUS_BUFFERING)
    assert body == b"\x00"
    assert ccmf.parse_status(body) == (ccmf.STATUS_BUFFERING, None)


def test_status_body_ended_is_bare():
    # END was merged into STATUS: ended is the bare state byte, no origin.
    body = ccmf.status_body(ccmf.STATUS_ENDED)
    assert body == b"\x02"
    assert ccmf.parse_status(body) == (ccmf.STATUS_ENDED, None)


def test_status_playing_requires_origin():
    with pytest.raises(ValueError):
        ccmf.status_body(ccmf.STATUS_PLAYING)


def test_status_draft00_playing_parses_without_origin():
    # A draft-00 sender's bare playing byte still parses; origin is just None
    # (the receiver falls back to heuristic anchoring).
    assert ccmf.parse_status(b"\x01") == (ccmf.STATUS_PLAYING, None)


def test_control_frame_layout_and_parse():
    f = ccmf.control(ccmf.OP_STATUS, bytes([ccmf.STATUS_PLAYING]))
    assert f == bytes([ccmf.OP_STATUS, 1, 0, ccmf.STATUS_PLAYING])
    assert ccmf.parse_message(f) == (ccmf.OP_STATUS, bytes([ccmf.STATUS_PLAYING]))


def test_media_message_is_the_chunk_itself():
    c = ccmf.chunk(0, ccmf.TYPE_VIDEO, b"x")
    opcode, body = ccmf.parse_message(c)
    assert opcode == ccmf.MARKER
    assert body == c                        # forwarded verbatim, no unwrapping


def test_control_opcode_may_not_shadow_marker():
    with pytest.raises(ValueError):
        ccmf.control(ccmf.MARKER, b"")


def test_room_roundtrip_all_fields():
    body = ccmf.build_room("https://ex/v", start_ms=90_000, end_ms=120_500,
                           loop=True, sync=True)
    room = ccmf.parse_room(body)
    assert room == ccmf.Room("https://ex/v", 90_000, 120_500, True, True)
    assert room.key() == ("https://ex/v", 90_000, 120_500, True)  # sync not in key


def test_room_optional_fields_absent():
    room = ccmf.parse_room(ccmf.build_room("u"))
    assert room == ccmf.Room("u", None, None, False, False)


def test_room_requires_url():
    with pytest.raises(ValueError):
        ccmf.parse_room(ccmf.build_room(""))


def test_caps_roundtrip():
    body = ccmf.build_caps(want_video=True, want_audio=False,
                           audio_mask=ccmf.CAP_AUDIO_DFPWM,
                           channels=ccmf.CAP_CHANNEL_MONO,
                           width=335, height=124, fps=30)
    caps = ccmf.parse_caps(body)
    assert caps.want_video and not caps.want_audio
    assert caps.audio_mask == ccmf.CAP_AUDIO_DFPWM
    assert caps.channels & ccmf.CAP_CHANNEL_MONO
    assert (caps.width, caps.height, caps.fps) == (335, 124, 30)
    assert len(body) == 11                  # spec §5.4: fixed 11-byte body


# --------------------------------------------------------------------------- #
# GopEncoder
# --------------------------------------------------------------------------- #

def _decode_chunk(chunk_bytes):
    pts, ctype, payload, _ = ccmf.parse_chunk(chunk_bytes)
    assert ctype == ccmf.TYPE_VIDEO
    return pts, ccmf.parse_video_payload(payload)


def test_gop_opens_with_palette_and_raw_keyframe():
    enc = GopEncoder(gop_samples=48000, nominal_duration=2000)
    assert enc.add(0, _solid_frame((200, 40, 40), 6, 4)) is None
    pts, (w, h, frames) = _decode_chunk(enc.flush()[1])
    assert (pts, w, h) == (0, 6, 4)
    assert frames[0].encoding == ccmf.ENC_RAW           # palette parsed before it
    assert frames[0].duration == 2000                   # nominal at end of stream


def test_gop_repeats_identical_frames_and_deltas_small_changes():
    # The base carries both colours, so the GOP palette can represent the change
    # (a solid base would quantise to 16 copies of one colour, and the changed
    # cell would collapse back onto it — correctly becoming a repeat).
    w, h = 8, 6
    base = _solid_frame((60, 60, 200), w, h)
    base[:, w:] = (200, 60, 60)                         # right half red
    changed = base.copy()
    changed[:3, :2] = (200, 60, 60)                     # one blue cell -> red
    enc = GopEncoder(gop_samples=10 * 48000)
    enc.add(0, base)
    enc.add(2000, base.copy())                          # identical -> repeat
    enc.add(4000, changed)                              # small change -> delta
    _pts, (_w, _h, frames) = _decode_chunk(enc.flush(next_pts=6000)[1])
    assert [f.encoding for f in frames] == \
        [ccmf.ENC_RAW, ccmf.ENC_REPEAT, ccmf.ENC_DELTA]
    assert [f.duration for f in frames] == [2000, 2000, 2000]
    # the delta must actually land the changed cell
    assert not np.array_equal(frames[2].glyph, frames[1].glyph) or \
        not np.array_equal(frames[2].fg, frames[1].fg) or \
        not np.array_equal(frames[2].bg, frames[1].bg)


def test_gop_boundary_flushes_previous_chunk():
    enc = GopEncoder(gop_samples=48000)
    frame = _solid_frame((10, 120, 90), 4, 3)
    assert enc.add(0, frame) is None
    assert enc.add(24000, frame.copy()) is None         # still inside the GOP
    done = enc.add(48000, frame.copy())                 # crosses the boundary
    assert done is not None
    pts, (_w, _h, frames) = _decode_chunk(done[1])
    assert pts == 0
    assert [f.encoding for f in frames] == [ccmf.ENC_RAW, ccmf.ENC_REPEAT]
    assert [f.duration for f in frames] == [24000, 24000]   # true PTS gaps
    # the boundary frame opened a fresh, self-contained GOP
    pts2, (_w, _h, frames2) = _decode_chunk(enc.flush()[1])
    assert pts2 == 48000
    assert frames2[0].encoding == ccmf.ENC_RAW


def test_gop_scene_cut_rekeys_mid_gop():
    w, h = 12, 8
    rng = np.random.default_rng(5)
    enc = GopEncoder(gop_samples=10 * 48000)
    enc.add(0, _solid_frame((250, 250, 250), w, h))
    # a full-frame content change: delta would exceed the raw planes
    enc.add(2000, rng.integers(0, 256, (h * 3, w * 2, 3), dtype=np.uint8))
    _pts, (_w, _h, frames) = _decode_chunk(enc.flush(next_pts=4000)[1])
    assert [f.encoding for f in frames] == [ccmf.ENC_RAW, ccmf.ENC_RAW]
    # the mid-GOP keyframe brought its own palette
    assert frames[0].palette.tobytes() != frames[1].palette.tobytes()


def test_gop_flush_empty_returns_none():
    assert GopEncoder().flush() is None


def test_gop_respects_byte_budget():
    # A chunk travels as ONE WebSocket message and CC:Tweaked drops the
    # connection when a message exceeds http.max_websocket_message (128 KiB
    # default) — so a GOP must flush early on SIZE, not just duration.  Random
    # frames force worst-case units (every frame re-keys to palette + raw).
    w, h = 20, 10
    rng = np.random.default_rng(6)
    frame_bytes = ccmf.raw_planes_size(w, h) + 49 + 3          # raw + palette + head
    budget = 16 + 3 * frame_bytes + 10                          # fits 3 keyframes
    enc = GopEncoder(gop_samples=100 * 48000, max_chunk_bytes=budget)

    chunks = []
    for i in range(8):
        done = enc.add(i * 2000,
                       rng.integers(0, 256, (h * 3, w * 2, 3), dtype=np.uint8))
        if done is not None:
            chunks.append(done[1])
    done = enc.flush()
    if done is not None:
        chunks.append(done[1])

    assert len(chunks) >= 2                       # the budget forced early flushes
    for chunk in chunks:
        assert len(chunk) <= budget
        # every size-bounded chunk is still a spec-legal RAP
        _pts, _t, payload, _ = ccmf.parse_chunk(chunk)
        _w, _h, frames = ccmf.parse_video_payload(payload)
        assert frames[0].encoding == ccmf.ENC_RAW

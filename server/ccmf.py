"""
CCMF — the ComputerCraft Media Format (docs/cc-media-format.md, draft 00).

Format-only module: byte-level packing/parsing for the two wire formats the spec
defines, shared by the producer (transcoder), the streamer (session/main) and the
Python-side reference decoder (tests / preview tooling).  The Lua client mirrors
the decode side.

1. Container Format — self-describing chunks:
       [marker 1B][PTS u48][length u24][type u8][payload]
   PTS is absolute in 48 kHz samples.  A video payload is a self-contained GOP
   ([w u16][h u16][compression u8] + a unit stream); an audio payload is
   [a-hdr u8] + samples.  Compression 0 (none) only — the rest is deferred.

2. Stream Format — every transport message starts with one byte: the container
   marker means "this message IS a chunk" (MEDIA); anything else is a control
   frame [opcode u8][length u16][body].

All multi-byte integers are unsigned little-endian; all header fields are
byte-aligned (spec §3).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Container constants
# --------------------------------------------------------------------------- #

MARKER = 0x43                    # ASCII "C"; resync byte and the MEDIA "opcode"
SAMPLE_RATE = 48000              # PTS/duration unit: 48 kHz samples

TYPE_VIDEO = 0
TYPE_AUDIO = 1

COMPRESSION_NONE = 0             # deflate/lz4/zstd are deferred

# Frame-unit encodings (unit flags bits 6-4).
ENC_RAW = 0
ENC_DELTA = 1
ENC_REPEAT = 2

# Audio codecs (a-hdr high nibble) and channel roles (a-hdr low nibble, spec §4.6).
CODEC_PCM8 = 0
CODEC_DFPWM = 1
CHANNEL_MONO = 0
CHANNEL_FRONT_LEFT = 1
CHANNEL_FRONT_RIGHT = 2
CHANNEL_CENTER = 3
CHANNEL_LFE = 4
CHANNEL_SURROUND_LEFT = 5
CHANNEL_SURROUND_RIGHT = 6
CHANNEL_REAR_LEFT = 7
CHANNEL_REAR_RIGHT = 8

MAX_DURATION = 0xFFFF            # u16 samples (~1.36 s); longer holds use repeats
_CHUNK_HEADER = 11               # marker + PTS + length + type


# --------------------------------------------------------------------------- #
# Chunk framing
# --------------------------------------------------------------------------- #

def chunk(pts: int, ctype: int, payload: bytes) -> bytes:
    """One container chunk.  `pts` is absolute 48 kHz samples (u48)."""
    if not 0 <= pts < 1 << 48:
        raise ValueError(f"PTS out of u48 range: {pts}")
    if len(payload) >= 1 << 24:
        raise ValueError(f"payload exceeds u24 length: {len(payload)}")
    # struct has no u48/u24, so build the header explicitly.
    return (bytes([MARKER])
            + pts.to_bytes(6, "little")
            + len(payload).to_bytes(3, "little")
            + bytes([ctype])
            + payload)


def parse_chunk(buf: bytes, offset: int = 0) -> tuple[int, int, bytes, int]:
    """Reference parser: -> (pts, type, payload, next_offset).  Raises ValueError
    on a bad marker or a truncated chunk (a decoder must bound-check, spec §7)."""
    if offset + _CHUNK_HEADER > len(buf):
        raise ValueError("truncated chunk header")
    if buf[offset] != MARKER:
        raise ValueError(f"bad marker byte {buf[offset]:#x} at offset {offset}")
    pts = int.from_bytes(buf[offset + 1:offset + 7], "little")
    length = int.from_bytes(buf[offset + 7:offset + 10], "little")
    ctype = buf[offset + 10]
    end = offset + _CHUNK_HEADER + length
    if end > len(buf):
        raise ValueError("truncated chunk payload")
    return pts, ctype, buf[offset + _CHUNK_HEADER:end], end


def iter_chunks(buf: bytes) -> Iterator[tuple[int, int, bytes]]:
    """Walk a chunk sequence (a stored file, or concatenated MEDIA messages)."""
    offset = 0
    while offset < len(buf):
        pts, ctype, payload, offset = parse_chunk(buf, offset)
        yield pts, ctype, payload


# --------------------------------------------------------------------------- #
# Video payload: raw planes
# --------------------------------------------------------------------------- #

def pack_chars(glyph: np.ndarray) -> bytes:
    """chars plane: 5-bit glyph indices, 8 cells packed MSB-first into 5 bytes
    (spec §4.5.1; the packing inherited from sanjuuni's 32vid).  `glyph` holds
    blit chars (0x80 + index), row-major."""
    codes = (np.asarray(glyph, np.uint8).ravel() - np.uint8(0x80))    # (n,) 0..31
    pad = (-codes.size) % 8
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, np.uint8)])
    grp = codes.reshape(-1, 8).astype(np.uint64)
    val = (grp[:, 0] << 35 | grp[:, 1] << 30 | grp[:, 2] << 25 | grp[:, 3] << 20
           | grp[:, 4] << 15 | grp[:, 5] << 10 | grp[:, 6] << 5 | grp[:, 7])
    out = np.empty((val.shape[0], 5), np.uint8)
    out[:, 0] = (val >> 32) & 0xFF
    out[:, 1] = (val >> 24) & 0xFF
    out[:, 2] = (val >> 16) & 0xFF
    out[:, 3] = (val >> 8) & 0xFF
    out[:, 4] = val & 0xFF
    return out.tobytes()


def unpack_chars(data: bytes, n: int) -> np.ndarray:
    """Inverse of pack_chars -> (n,) uint8 blit chars (0x80 + index)."""
    ng = (n + 7) // 8
    scr = np.frombuffer(data, np.uint8, count=ng * 5).reshape(ng, 5).astype(np.uint64)
    val = (scr[:, 0] << 32 | scr[:, 1] << 24 | scr[:, 2] << 16
           | scr[:, 3] << 8 | scr[:, 4])
    codes = np.empty((ng, 8), np.uint8)
    for j in range(8):
        codes[:, j] = (val >> np.uint64(5 * (7 - j))) & np.uint64(0x1F)
    return codes.ravel()[:n] + np.uint8(0x80)


def pack_nibbles(idx: np.ndarray) -> bytes:
    """fg/bg plane: 4-bit palette indices, 2 cells/byte, high nibble first
    (spec §4.5.1)."""
    flat = np.asarray(idx, np.uint8).ravel()
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, np.uint8)])
    return ((flat[0::2] << 4) | flat[1::2]).tobytes()


def unpack_nibbles(data: bytes, n: int) -> np.ndarray:
    """Inverse of pack_nibbles -> (n,) uint8 palette indices."""
    b = np.frombuffer(data, np.uint8, count=(n + 1) // 2)
    out = np.empty(((n + 1) // 2) * 2, np.uint8)
    out[0::2] = b >> 4
    out[1::2] = b & 0x0F
    return out[:n]


def raw_planes_size(w: int, h: int) -> int:
    """Byte size of the packed raw planes (chars + fg + bg) for a W x H grid."""
    n = w * h
    return ((n + 7) // 8) * 5 + ((n + 1) // 2) * 2


# --------------------------------------------------------------------------- #
# Video payload: units
# --------------------------------------------------------------------------- #

def palette_unit(palette: bytes) -> bytes:
    """Palette unit (flags bit7=0): exactly 48 bytes of 16 x RGB."""
    if len(palette) != 48:
        raise ValueError(f"palette must be 48 bytes, got {len(palette)}")
    return b"\x00" + palette


def raw_frame_unit(duration: int, glyph: np.ndarray, fg: np.ndarray,
                   bg: np.ndarray) -> bytes:
    """raw frame unit: keyframe/RAP.  duration u16 then chars/fg/bg planes."""
    return (bytes([0x80 | (ENC_RAW << 4)]) + struct.pack("<H", duration)
            + pack_chars(glyph) + pack_nibbles(fg) + pack_nibbles(bg))


def repeat_frame_unit(duration: int) -> bytes:
    """repeat frame unit: hold the current frame for `duration` samples."""
    return bytes([0x80 | (ENC_REPEAT << 4)]) + struct.pack("<H", duration)


def delta_spans(prev: tuple[np.ndarray, np.ndarray, np.ndarray],
                cur: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bytes | None:
    """Changed-cell spans between two (glyph, fg, bg) grids -> the delta unit BODY
    after `duration` ([span count u16] + spans), or None if nothing changed.

    Spans never cross a row boundary and cap at 255 cells (spec §4.5.2).  Each
    cell is [blit char][colour] with colour = (bg << 4) | fg.
    """
    pg, pf, pb = (np.asarray(a, np.uint8) for a in prev)
    cg, cf, cb = (np.asarray(a, np.uint8) for a in cur)
    h, w = cg.shape
    changed = (pg != cg) | (pf != cf) | (pb != cb)
    hits = np.flatnonzero(changed.ravel())
    if hits.size == 0:
        return None

    # Split consecutive runs of changed cells at row boundaries and at 255 cells.
    breaks = np.flatnonzero((np.diff(hits) != 1) | (hits[1:] % w == 0)) + 1
    runs = np.split(hits, breaks)

    glyph_flat = cg.ravel()
    colour_flat = ((cb << 4) | cf).ravel()
    parts = []
    count = 0
    for run in runs:
        for s in range(0, run.size, 255):
            seg = run[s:s + 255]
            start, length = int(seg[0]), int(seg.size)
            cells = np.empty(length * 2, np.uint8)
            cells[0::2] = glyph_flat[seg]
            cells[1::2] = colour_flat[seg]
            parts.append(struct.pack("<HB", start, length) + cells.tobytes())
            count += 1
    return struct.pack("<H", count) + b"".join(parts)


def delta_frame_unit(duration: int, body: bytes) -> bytes:
    """delta frame unit around a body produced by delta_spans."""
    return bytes([0x80 | (ENC_DELTA << 4)]) + struct.pack("<H", duration) + body


def video_payload(w: int, h: int, units: bytes) -> bytes:
    """A video chunk payload: [w u16][h u16][compression u8] + unit stream.
    Only COMPRESSION_NONE is implemented (the rest is deferred)."""
    return struct.pack("<HHB", w, h, COMPRESSION_NONE) + units


def audio_payload(codec: int, data: bytes, channel: int = CHANNEL_MONO) -> bytes:
    """An audio chunk payload: [a-hdr u8] + samples.  a-hdr high nibble = codec,
    low nibble = channel role."""
    return bytes([((codec & 0x0F) << 4) | (channel & 0x0F)]) + data


# --------------------------------------------------------------------------- #
# Reference video decoder (tests / preview tooling; the Lua client mirrors it)
# --------------------------------------------------------------------------- #

@dataclass
class DecodedFrame:
    """One presented frame: the full grids after applying the unit, plus the
    palette in effect (a (16,3) uint8 copy), its hold duration in samples, and
    the encoding the unit used (ENC_RAW/ENC_DELTA/ENC_REPEAT)."""
    duration: int
    encoding: int
    glyph: np.ndarray
    fg: np.ndarray
    bg: np.ndarray
    palette: np.ndarray


def parse_video_payload(payload: bytes) -> tuple[int, int, list[DecodedFrame]]:
    """Decode a video chunk payload -> (w, h, frames).  Applies palette units and
    materialises delta/repeat frames against the running state, exactly as a
    client would; every chunk is a RAP so no prior state is needed."""
    if len(payload) < 5:
        raise ValueError("truncated video payload header")
    w, h, compression = struct.unpack_from("<HHB", payload)
    if compression != COMPRESSION_NONE:
        raise ValueError(f"unsupported compression {compression}")
    pos = 5
    n = w * h
    palette: Optional[np.ndarray] = None
    glyph = fg = bg = None
    frames: list[DecodedFrame] = []
    while pos < len(payload):
        flags = payload[pos]
        pos += 1
        if not flags & 0x80:                                   # palette unit
            palette = np.frombuffer(payload, np.uint8, count=48,
                                    offset=pos).reshape(16, 3).copy()
            pos += 48
            continue
        enc = (flags >> 4) & 0x07
        (duration,) = struct.unpack_from("<H", payload, pos)
        pos += 2
        if enc == ENC_RAW:
            nc = ((n + 7) // 8) * 5
            nn = (n + 1) // 2
            glyph = unpack_chars(payload[pos:pos + nc], n).reshape(h, w)
            pos += nc
            fg = unpack_nibbles(payload[pos:pos + nn], n).reshape(h, w)
            pos += nn
            bg = unpack_nibbles(payload[pos:pos + nn], n).reshape(h, w)
            pos += nn
        elif enc == ENC_DELTA:
            if glyph is None:
                raise ValueError("delta frame before any raw keyframe")
            glyph, fg, bg = glyph.copy(), fg.copy(), bg.copy()
            (count,) = struct.unpack_from("<H", payload, pos)
            pos += 2
            gf, ff, bf = glyph.ravel(), fg.ravel(), bg.ravel()
            for _ in range(count):
                start, length = struct.unpack_from("<HB", payload, pos)
                pos += 3
                if start % w + length > w or start + length > n:
                    raise ValueError("delta span crosses a row/grid boundary")
                cells = np.frombuffer(payload, np.uint8, count=length * 2,
                                      offset=pos).reshape(length, 2)
                pos += length * 2
                gf[start:start + length] = cells[:, 0]
                ff[start:start + length] = cells[:, 1] & 0x0F
                bf[start:start + length] = cells[:, 1] >> 4
        elif enc == ENC_REPEAT:
            if glyph is None:
                raise ValueError("repeat frame before any raw keyframe")
        else:
            raise ValueError(f"unknown frame encoding {enc}")
        if palette is None:
            raise ValueError("frame before any palette unit")
        frames.append(DecodedFrame(duration, enc, glyph, fg, bg, palette))
    if not frames:
        raise ValueError("video chunk carries no frame (ends with a palette?)")
    return w, h, frames


def parse_audio_payload(payload: bytes) -> tuple[int, int, bytes]:
    """Decode an audio chunk payload -> (codec, channel, samples)."""
    if not payload:
        raise ValueError("empty audio payload")
    return payload[0] >> 4, payload[0] & 0x0F, payload[1:]


# --------------------------------------------------------------------------- #
# Stream Format: control frames
# --------------------------------------------------------------------------- #

OP_ROOM = 1
OP_CAPS = 2
OP_START = 3
OP_QUIT = 4
OP_ACK = 5
OP_ERROR = 6
OP_STATUS = 7
OP_END = 8
# MEDIA has no distinct opcode: its value IS the container marker (67).

STATUS_BUFFERING = 0
STATUS_PLAYING = 1

# CAPS bitmask values.
CAP_AUDIO_PCM8 = 0x01
CAP_AUDIO_DFPWM = 0x02
CAP_COMPRESS_NONE = 0x01
# channels: bit N = accepts channel role N (spec §5.4, roles per §4.6).
CAP_CHANNEL_MONO = 1 << CHANNEL_MONO
CAP_CHANNEL_FRONT_LEFT = 1 << CHANNEL_FRONT_LEFT
CAP_CHANNEL_FRONT_RIGHT = 1 << CHANNEL_FRONT_RIGHT
CAP_CHANNEL_CENTER = 1 << CHANNEL_CENTER
CAP_CHANNEL_LFE = 1 << CHANNEL_LFE
CAP_CHANNEL_SURROUND_LEFT = 1 << CHANNEL_SURROUND_LEFT
CAP_CHANNEL_SURROUND_RIGHT = 1 << CHANNEL_SURROUND_RIGHT
CAP_CHANNEL_REAR_LEFT = 1 << CHANNEL_REAR_LEFT
CAP_CHANNEL_REAR_RIGHT = 1 << CHANNEL_REAR_RIGHT


def control(opcode: int, body: bytes = b"") -> bytes:
    """One control frame: [opcode u8][length u16][body]."""
    if opcode == MARKER:
        raise ValueError("control opcodes must differ from the chunk marker")
    if len(body) > 0xFFFF:
        raise ValueError("control body exceeds u16 length")
    return struct.pack("<BH", opcode, len(body)) + body


def parse_message(msg: bytes) -> tuple[int, bytes]:
    """Classify one transport message -> (opcode, body).  A MEDIA message returns
    (MARKER, whole-chunk-bytes); a control frame returns (opcode, body)."""
    if not msg:
        raise ValueError("empty message")
    if msg[0] == MARKER:
        return MARKER, msg
    if len(msg) < 3:
        raise ValueError("truncated control frame")
    opcode, length = struct.unpack_from("<BH", msg)
    if len(msg) < 3 + length:
        raise ValueError("truncated control body")
    return opcode, msg[3:3 + length]


# --------------------------------------------------------------------------- #
# HELLO bodies (ROOM / CAPS)
# --------------------------------------------------------------------------- #

_ROOM_LOOP = 0x01
_ROOM_SYNC = 0x02
_ROOM_HAS_START = 0x04
_ROOM_HAS_END = 0x08


@dataclass(frozen=True)
class Room:
    url: str
    start_ms: Optional[int]      # ms into the source, None if absent
    end_ms: Optional[int]
    loop: bool
    sync: bool

    def key(self) -> tuple:
        """Shared-room identity (spec §5.5): (url, start, end, loop)."""
        return (self.url, self.start_ms, self.end_ms, self.loop)


def build_room(url: str, start_ms: Optional[int] = None,
               end_ms: Optional[int] = None, loop: bool = False,
               sync: bool = False) -> bytes:
    flags = ((_ROOM_LOOP if loop else 0) | (_ROOM_SYNC if sync else 0)
             | (_ROOM_HAS_START if start_ms is not None else 0)
             | (_ROOM_HAS_END if end_ms is not None else 0))
    body = bytes([flags])
    if start_ms is not None:
        body += struct.pack("<I", start_ms)
    if end_ms is not None:
        body += struct.pack("<I", end_ms)
    return body + url.encode("utf-8")


def parse_room(body: bytes) -> Room:
    if not body:
        raise ValueError("empty ROOM body")
    flags = body[0]
    pos = 1
    start_ms = end_ms = None
    if flags & _ROOM_HAS_START:
        (start_ms,) = struct.unpack_from("<I", body, pos)
        pos += 4
    if flags & _ROOM_HAS_END:
        (end_ms,) = struct.unpack_from("<I", body, pos)
        pos += 4
    url = body[pos:].decode("utf-8")
    if not url:
        raise ValueError("ROOM body has no URL")
    return Room(url, start_ms, end_ms,
                bool(flags & _ROOM_LOOP), bool(flags & _ROOM_SYNC))


@dataclass(frozen=True)
class Caps:
    want_video: bool
    want_audio: bool
    video_mask: int              # optional/future encodings; raw/delta/repeat mandatory
    audio_mask: int              # bit0 pcm8, bit1 dfpwm
    channels: int                # one-hot channel roles; bit0 = mono
    compress_mask: int           # bit0 none (mandatory)
    width: int
    height: int
    fps: int


_CAPS_FMT = "<BBBHBHHB"
_CAPS_SIZE = struct.calcsize(_CAPS_FMT)     # 11 bytes


def build_caps(want_video: bool = True, want_audio: bool = True,
               video_mask: int = 0,
               audio_mask: int = CAP_AUDIO_PCM8 | CAP_AUDIO_DFPWM,
               channels: int = CAP_CHANNEL_MONO,
               compress_mask: int = CAP_COMPRESS_NONE,
               width: int = 51, height: int = 19, fps: int = 24) -> bytes:
    flags = (0x01 if want_video else 0) | (0x02 if want_audio else 0)
    return struct.pack(_CAPS_FMT, flags, video_mask, audio_mask, channels,
                       compress_mask, width, height, fps)


def parse_caps(body: bytes) -> Caps:
    if len(body) < _CAPS_SIZE:
        raise ValueError(f"CAPS body too short: {len(body)} < {_CAPS_SIZE}")
    (flags, video_mask, audio_mask, channels, compress_mask,
     width, height, fps) = struct.unpack_from(_CAPS_FMT, body)
    return Caps(bool(flags & 0x01), bool(flags & 0x02), video_mask, audio_mask,
                channels, compress_mask, width, height, fps)

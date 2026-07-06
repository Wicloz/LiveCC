---
title: ComputerCraft Media Format (CCMF)
docname: draft-ccmf-01
category: info
status: Working Draft
---

# ComputerCraft Media Format (CCMF)

Working Draft 01 — replaces the legacy "32vid" transport used by LiveCC.

Stored files use the extension `.ccmf`.

## Abstract

CCMF defines two independent wire formats:

1. A **Container Format** — a self-describing byte stream of media chunks,
   usable identically as an on-disk file or a live stream.
2. A **Stream Format** — a thin, all-binary control protocol that carries
   container chunks and a text side-channel over any reliable, ordered,
   bidirectional transport (e.g. WebSocket, TCP).

Server and client internals — transcoding, room management, and any
resolution/fps/codec reconciliation — are explicitly out of scope.

## Status of This Memo

This is a working draft under active design; it tracks the LiveCC reference
implementation and MAY change incompatibly. There is no version field —
compatibility is capability-driven (Section 6). Implementers should pin to a
specific draft.

---

## 1. Introduction

CCMF targets ComputerCraft (CC:Tweaked) terminals: a grid of character cells,
each a 5-bit drawing glyph plus a 4-bit foreground and 4-bit background index
into a 16-entry RGB palette. The Lua/CC client is the primary optimization
target; other clients (e.g. a standalone player) are permitted but MUST NOT be
optimized for at the CC client's expense.

Three roles participate:

- **Producer** — transcodes source media into container chunks. Knows nothing
  of transport.
- **Streamer** — carries chunks (and control/text) from a producer or a file to
  clients over a transport. Treats chunk payloads as opaque.
- **Client** — consumes chunks from a file, a streamer, or a producer directly.
  Only the streamer path involves the Stream Format.

A stored file is a bare sequence of container chunks; a live stream is the same
bytes wrapped in the Stream Format. The two are byte-compatible.

## 2. Terminology

The key words **MUST**, **MUST NOT**, **SHOULD**, **MAY** are to be interpreted
as in RFC 2119.

- **Chunk** — the atomic unit of the container (Section 4.1).
- **Unit** — a sub-element inside a video payload (Section 4.5).
- **RAP** (Random-Access Point) — a position from which decoding can begin with
  no prior state. Every video chunk is a RAP.
- **PTS** — Presentation Timestamp, absolute, in 48 kHz samples (Section 4.2).
- **Room** — a shared production identified by `(url, start, end, loop)` under
  `sync` (Section 5).

## 3. Conventions

- All multi-byte integers are **unsigned little-endian**. The choice is
  arbitrary for correctness provided it is fixed; LE is mandated because it
  matches the existing 32vid decoder and sanjuuni, is the natural form of
  `string.byte(s, i, j)` reads (`b1 + b2·256 + …`), and is native to x86/Python/
  C++ producers — deliberately over IETF's big-endian "network byte order"
  convention, which would only add CC decode cost.
- All header fields are **byte-aligned**; sub-byte packing is permitted only
  *within* a single byte (flag bits / nibbles). Payload bodies MAY bit-pack.
- Bit 7 is the most significant bit of a byte.
- **Human-readable text** (`ERROR` bodies; subtitle payloads when defined) is
  **ISO-8859-1 (Latin-1)** restricted to **printable** code points (`0x20–0x7E`
  and `0xA0–0xFF`); bytes in the control/graphics ranges (`0x00–0x1F`, `0x7F`,
  `0x80–0x9F`) MUST NOT appear. UTF-8 is **not** used — a multi-byte sequence
  renders as several glyphs on a CC terminal. (A CC font additionally maps
  `0x00–0x1F` to CP437 symbols and `0x80–0x9F` to the block glyphs of
  Section 4.5; those are implementation-specific and outside this spec's text
  definition.)

---

## 4. Container Format

### 4.1. Chunk

```
+--------+------------+----------+--------+===============+
| marker |    PTS     |  length  |  type  |    payload    |
|  1 B   |    6 B      |   3 B    |  1 B   |  length bytes |
+--------+------------+----------+--------+===============+
```

| Field  | Size | Description                                              |
|--------|------|----------------------------------------------------------|
| marker | 1 B  | Fixed resync byte: `67` (`0x43`, ASCII `C`).             |
| PTS    | 6 B  | Absolute presentation timestamp (Section 4.2).           |
| length | 3 B  | Payload byte count, `u24` (≤ 16 MiB).                    |
| type   | 1 B  | `0` = video, `1` = audio. Others reserved.               |

A reader locates chunk N+1 at `offset(N) + 11 + length`. The 11-byte header is
fixed; a decoder encountering an unknown `type` MUST skip `length` bytes and
continue.

#### 4.1.1. Marker, length, and seeking

The `marker` enables **resync** after a blind seek: scan for a candidate marker,
then validate by *chaining* (read its `length`, jump, expect another marker;
false hits do not chain). Correctness comes from chaining, so a 1-byte marker
suffices — `P(random byte) = 1/256`, and the CC client never scans (WebSocket
delivers one chunk per binary message; file reading chains from offset 0).

`length` is a fixed `u24` (≤ 16 MiB) counting the **payload only**; chunk N+1
begins at `offset(N) + 11 + length`. This covers a busy 1–2 s chunk (~1 MB) with
wide margin.

### 4.2. Timestamps

`PTS` is a 48-bit absolute count of **48 kHz samples** since the start of the
stream/file (origin 0). Audio PTS is a sample index (exact); video chunk PTS is
that of its first frame. 48 bits gives ~186 years and stays below Lua's 2^53
exact-integer limit; a full u64 MUST NOT be used (it breaks CC arithmetic).

All chunk types share **one** timeline: a video frame and an audio sample with
equal PTS are simultaneous.  A producer transcoding audio and video through
separate pipelines (or fetching them separately from a live source) **MUST**
place both on this common timeline — synchronization is expressed *only*
through PTS; receivers are entitled to trust it and have no other signal to
correct with.

Common frame intervals are exact: 24 fps = 2000, 30 = 1600, 25 = 1920 samples.

### 4.3. Chunk Types

| Value | Type    | Status              |
|-------|---------|---------------------|
| 0     | Video   | Section 4.4         |
| 1     | Audio   | Section 4.6         |
| 2     | Subtitle| `[ED: deferred]`    |
| 3–255 | —       | Reserved            |

### 4.4. Video Payload

A video chunk is a **self-contained GOP** and thus a RAP.

```
+---------+---------+-------------+========================+
|  width  | height  | compression |     unit stream        |
|  u16    |  u16    |    u8       |  (compression applied) |
+---------+---------+-------------+========================+
```

- `compression`: `0` none, `1` deflate, `2` lz4, `3` zstd `[ED: deferred; CC
  decodes `0` only for now]`. It covers the **entire unit stream as one blob**,
  not individual frames; the 5-byte head stays uncompressed. Whole-payload is
  chosen because consecutive `delta` frames are highly redundant (one blob
  compresses far better than isolated frames), the client decompresses once per
  chunk, and per-frame random access is unnecessary — the chunk is the
  random-access unit.
- The unit stream is decoded until `length` is consumed (no unit count).
- The stream **MUST** begin with a `palette` unit followed by a `raw` frame
  (the keyframe), and **MUST NOT** end with a `palette` unit.

### 4.5. Units

```
+--------+========+
| flags  |  body  |
|  1 B   |        |
+--------+========+
```

`flags` bit 7: `0` = palette unit, `1` = frame unit.

**Palette unit** (`bit7=0`): body is exactly 48 bytes (16 × RGB, 8-bit
channels). It replaces the current palette for subsequent frames and is applied
instantaneously (no timestamp). Palette-only changes (e.g. fades) need no frame.

**Frame unit** (`bit7=1`): bits 6–4 select the encoding; bits 3–0 reserved.

| Enc | Name      | Body                                             |
|-----|-----------|--------------------------------------------------|
| 0   | raw       | `duration u16`, then packed planes (keyframe)    |
| 1   | delta     | `duration u16`, then changed spans (Section 4.5.2)|
| 2   | repeat    | `duration u16` only — hold current frame         |

`duration` is in 48 kHz samples (≤ 65535 ≈ 1.36 s). Longer holds use successive
`repeat` units. A frame is a **keyframe/RAP** iff its encoding is `raw`.

#### 4.5.1. raw planes

Three planes in order **chars, fg, bg** for `W·H` cells (row-major):

- **chars**: 5-bit glyph indices, 8 cells packed MSB-first into 5 bytes
  (`ceil(W·H/8)·5` B), the packing inherited from sanjuuni's 32vid [SANJUUNI].
  Blit char = `0x80 + index`.
- **fg**, **bg**: 4-bit palette indices, 2 cells/byte, high nibble first
  (`ceil(W·H/2)` B each).

#### 4.5.2. delta

```
+-------------+-------------------------------------------+
| span count  |            span × count                   |
|    u16      |                                           |
+-------------+-------------------------------------------+

span = [ start u16 ] [ length u8 ] [ cell × length ]
cell = [ char 1 B ] [ colour 1 B ]
```

- `start` = linear cell index; `x = start % W`, `y = start // W`. A span **MUST
  NOT** cross a row boundary (`start % W + length ≤ W`).
- `char` is **blit-ready**: `0x80 + glyph index` (used directly as `blit` text).
- `colour` = `(bg << 4) | fg` (nibble → hex via a 16-entry table).

A decoder redraws by, per span, `setCursorPos(x+1, y+1)` then `blit`. Untouched
cells persist in the terminal, so **no frame buffer is required**. Encoders
SHOULD emit `raw` when a `delta` would exceed a full frame.

> Note: `start` (u16) addresses ≤ 65535 cells; CC's maximum (335×124 = 41540)
> fits. Larger non-CC grids are out of scope for this field width.

### 4.6. Audio Payload

```
+--------+===============+
| a-hdr  |    samples    |
|  1 B   |               |
+--------+===============+
```

`a-hdr`: high nibble = codec (`0` pcm8, `1` dfpwm), low nibble = channel role.
Fixed 48 kHz. Channel roles (shared numbering with the CAPS `channels` bitmask,
Section 5.4): `0` mono · `1` front-left · `2` front-right · `3` center · `4` LFE
· `5` surround-left · `6` surround-right · `7` rear-left · `8` rear-right ·
`9`–`15` reserved.

- **pcm8**: unsigned 8-bit, 1 byte/sample (amplitude = byte − 128), the native
  ComputerCraft speaker format [CCSPEAKER].
- **dfpwm**: 1 bit/sample, DFPWM1a [DFPWM]. The decoder state **MUST** be reset
  at the start of every chunk (so each chunk is independently decodable /
  seekable).

Chunk PTS is the first sample's index; sample count is derived from `length`.
Stereo is two audio chunks (roles front-left + front-right) sharing PTS; a
client advertises which roles it accepts via the CAPS `channels` bitmask
(Section 5.4).

---

## 5. Stream Format

An all-binary control protocol. It carries container chunks unmodified and adds
a text side-channel. Semantically SMTP/IMAP-like (verbs + a lock-step
handshake), but every message is binary — which is also the cheapest to parse in
Lua.

### 5.1. Framing

Each transport message begins with one byte:

- If it equals the container `marker`, the message **is** a container chunk,
  verbatim (a `MEDIA` message; the marker doubles as its opcode). Streamers
  forward these without inspection.
- Otherwise it is a control frame:

```
+--------+----------+========+
| opcode |  length  |  body  |
|  1 B   |   u16    |        |
+--------+----------+========+
```

Control opcode values **MUST** differ from `marker`.

### 5.2. Opcodes

| Val | Opcode | Dir  | Cast     | Body                              |
|-----|--------|------|----------|-----------------------------------|
| 1   | ROOM   | C→S  | unicast  | Section 5.4                       |
| 2   | CAPS   | C→S  | unicast  | Section 5.4                       |
| 3   | START  | C→S  | unicast  | empty                             |
| 4   | QUIT   | C→S  | unicast  | empty                             |
| 5   | ACK    | S→C  | unicast  | empty                             |
| 6   | ERROR  | S→C  | unicast  | Latin-1 text, sanitized (§3)      |
| 7   | STATUS | S→C  | multicast| Section 5.6                       |
| 67  | MEDIA  | S→C  | multicast| = a container chunk (marker-led)  |

`MEDIA` has no distinct opcode: its value **is** the container `marker` (67),
which leads every chunk, so all other opcodes differ from 67. There are no
numeric reply codes; an `ACK`/`ERROR` opcode is self-explanatory.

### 5.3. Handshake

```
C→S  ROOM   (url, start, end, loop, sync)
S→C  ACK  |  ERROR
C→S  CAPS   (capabilities + preferences)
S→C  ACK  |  ERROR          <- accept-or-drop decision
C→S  START
S→C  STATUS / MEDIA / ...   (until STATUS state = ended)
```

**Step 1 (ROOM)** determines routing. **Step 2 (CAPS)** is the first point the
server knows the client's capabilities, so its reply **is** the accept-or-drop
decision. After `START`, the client is a passive subscriber to the media hose.

The server **MUST NOT** echo the chosen configuration; clients learn actual
`W`/`H`/palette/codec from the self-describing media.

### 5.4. HELLO Bodies

**ROOM**:

```
[ flags u8 ]  bit0 loop · bit1 sync · bit2 has-start · bit3 has-end
[ start u32 ]  present iff has-start — ms into source
[ end   u32 ]  present iff has-end   — ms into source
[ url   ... ]  remaining bytes, UTF-8
```

**CAPS**:

```
[ flags u8 ]     bit0 want-video · bit1 want-audio
[ video u8 ]     bitmask, optional/future encodings (raw/delta/repeat are mandatory)
[ audio u8 ]     bitmask: bit0 pcm8 · bit1 dfpwm
[ channels u16 ] one-hot per channel role (Section 4.6): bit N = accepts role N
                 (bit0 mono · bit1 front-left · bit2 front-right · … up to 7.1)
[ compress u8 ]  bitmask: bit0 none · bit1 deflate · bit2 lz4 · bit3 zstd
[ width  u16 ]
[ height u16 ]
[ fps    u8  ]
```

Every client **MUST** support all three frame encodings (`raw`, `delta`,
`repeat`) and compression `none`. Audio is **optional**: a client MAY support
`pcm8`, `dfpwm`, both, or neither — supporting neither (or `want-audio = 0`)
means no audio is sent. A client requesting audio **MUST** set the
`mono` bit (role 0) as a fallback; positional roles (front-left/right for stereo,
plus surround/LFE for 5.1/7.1) are optional. The server emits one audio stream
per selected role (stereo = front-left + front-right chunks sharing PTS) and MAY
change the set mid-stream like the codec. `channels` shares its numbering with
the container `channel` field (Section 4.6). The other bitmasks advertise
*optional/future* features.

A role's samples need not originate in a discrete source channel: `channels`
declares the receiver's speaker layout, and the producer SHOULD resolve any
mismatch with the source's layout by *deriving* the role — a nearby channel,
a downmix, ultimately the mono mix — rather than omitting it.  Duplicated
content across roles is the correct rendering of a mismatch (a mono source on
a stereo layout plays the mono on both sides); a silent, never-delivered role
is not.  How a producer mixes is implementation-defined.

### 5.5. Rooms, Cast, and Dropping

- `sync = 0`: the client always gets a **private** room (its own production,
  independently seekable). `sync = 1`: create or join the **shared** room keyed
  by `(url, start, end, loop)`.
- Handshake and per-client `ERROR` are **unicast**; `MEDIA` and `STATUS` are
  **multicast** — a streamer serializes each chunk once and writes identical
  bytes to every subscriber.
- A server **MAY** drop a client with `ERROR` at CAPS or later. This is
  **required** for genuine capability mismatch (a shared-room joiner with no
  common encoding) and **MAY** be used to decline any divergence the server
  chooses not to reconcile. A dropped client MAY retry with `sync = 0`.
- Reconciliation of divergent resolution/fps/codec within a room is
  **implementation-defined** and out of scope; the container's self-describing
  fields make any such mid-stream change expressible without signaling.

### 5.6. STATUS and the Playback Clock

```
+-------+============================+
| state |  origin (state = 1 only)   |
|  1 B  |  PTS u48                   |
+-------+============================+
```

`state`: `0` buffering · `1` playing · `2` ended.  Only `playing` carries a body
(**origin**); `buffering` and `ended` are the `state` byte alone.  `ended` is
terminal — the stream is complete; a receiver finishes any buffered media, then
stops.  A `playing` STATUS carries **origin**, the PTS (Section 4.2) that is
*due for presentation now* — it maps the shared PTS timeline onto the receiver's
local clock.

A streamer **MUST** send `playing` + origin:

1. when playback starts (after its initial prebuffer),
2. whenever the mapping changes — resuming after a rebuffer (`buffering` was
   sent, the timeline is later than wall time suggested) or skipping toward
   the live edge (the timeline jumped forward),
3. unicast to a subscriber that joins a room mid-stream, so a late joiner
   anchors to the room's running clock instead of guessing from arrival times.

Receivers **SHOULD** anchor a single media clock to origin at receipt (plus a
small fixed startup delay of their choosing) and present *both* video frames
and audio samples against that clock.  On a new origin, queued media behind it
is stale and SHOULD be dropped (whole GOPs for video — resume at a keyframe).

> Why origin exists: it is a *per-receiver* necessity, not a room-sync
> nicety.  Because delivery leads presentation (below), the first chunk a
> receiver sees may sit anywhere within the sender's delivery lead — so a
> clock anchored on arrival times is wrong by up to that lead, and every
> stream mis-schedules together: audio ahead of its due time gets discarded
> as stale, video burst-renders.  Recovering the mapping by *heuristics*
> (draft-00 behaviour: re-anchor when chunks run early/late) conflates
> network jitter, rebuffering, and live-edge skips — three things that need
> three different reactions.  origin is the one number that ties the PTS
> timeline to "now", cheap to send and exact.  That it also lets a
> mid-stream room joiner adopt the running clock is a side benefit.

**Delivery is not presentation.**  A streamer MAY deliver chunks ahead of
their PTS (jitter slack); a receiver MUST buffer early chunks and schedule
them by PTS, never play them on arrival.  In particular, audio for a role
whose PTS is not yet due MUST be held (or fed to a device whose own buffering
preserves the timing), and audio whose PTS has entirely passed MUST be
discarded — this, not arrival order, is what keeps multiple channel roles
sample-aligned with each other and with video.

While a role is active its audio chunks are contiguous in PTS (chunk N+1's
PTS = chunk N's PTS + its sample count).  Roles MAY join or leave mid-stream
(Section 5.4); a receiver realigns a (re)joining role by its PTS against the
same clock.

---

## 6. Versioning

This document has **no version field**, by design.

While the format is under development it tracks the LiveCC reference
implementation and MAY change incompatibly; implementers should pin to a
specific draft. Once stable, compatibility is **capability-driven**, not
versioned:

- **Streams** negotiate features in CAPS (Section 5.4): a server emits only the
  encodings, codecs, channels, and compression the client advertised, so a
  client never receives a feature it cannot decode.
- **Evolution is additive** — new features are new capability bits, chunk
  `type`s, or enum values. A decoder MUST ignore reserved bits and MUST skip
  (via `length`) any chunk whose `type` or contents it does not support,
  continuing with the next chunk.
- **Files** have no negotiation, so the baseline-mandatory set (Section 5.4) is
  their interop floor; unsupported optional features degrade to skipped chunks,
  not failure.
- An incompatible successor would use a different `marker` byte, so files and
  streams self-identify from their first byte — no version number is needed to
  distinguish them.

## 7. Security Considerations

Servers **MUST** sanitize `ERROR` bodies: internal detail (stack traces, paths)
MUST NOT be sent to clients. Decoders **MUST** bound-check `length`, `span`
counts, and `start`/`length` against `W·H` before allocating or drawing.

---

## 8. References

### 8.1. Normative

- **[RFC2119]** Bradner, S., "Key words for use in RFCs to Indicate Requirement
  Levels", BCP 14, RFC 2119, March 1997.

### 8.2. Informative

- **[SANJUUNI]** MCJack123, "sanjuuni: Converts images and videos into a format
  suitable for display in ComputerCraft", <https://github.com/MCJack123/sanjuuni>.
  Origin of the 32vid format and the 8-glyphs-into-5-bytes character packing
  (Section 4.5.1).
- **[DFPWM]** "cc.audio.dfpwm — Convert between DFPWM and PCM audio", CC:Tweaked,
  <https://tweaked.cc/library/cc.audio.dfpwm.html>. The DFPWM1a codec of
  Section 4.6.
- **[CCSPEAKER]** "speaker — the speaker peripheral", CC:Tweaked,
  <https://tweaked.cc/peripheral/speaker.html>. Defines the 48 kHz mono 8-bit
  amplitude contract underlying `pcm8` (Section 4.6).

---

## Appendix A. Open Issues

1. ~~Name / file extension~~ — resolved: **CCMF**, `.ccmf`.
2. ~~opcode values~~ — resolved (Section 5.2).
3. ~~length width~~ — resolved: `u24`, payload-only (Section 4.1).
4. ~~Versioning~~ — resolved: none; capability-driven and additive (Section 6).
5. **Compression** (4.4) and **subtitles** (4.3) — deferred.
6. ~~Endianness~~ — resolved: little-endian (§3), verified against the current
   decoder ([player.lua](../lua/player.lua)).

## Appendix B. Changes from draft-00

- STATUS `playing` now carries the clock **origin** (u48 PTS) and is re-sent at
  every timeline discontinuity and to mid-stream room joiners (Section 5.6).
  draft-00 receivers that read only the first body byte are unaffected;
  draft-00 *senders* leave a draft-01 receiver anchoring heuristically.
- Section 4.2 now states explicitly that audio and video share one timeline
  and that producers must reconcile separately-fetched streams onto it.
- Section 5.6 defines the delivery-vs-presentation contract: receivers
  schedule media by PTS against the STATUS-anchored clock; early delivery is
  buffered, stale media is dropped, and channel roles are aligned by PTS.
- The former `END` opcode (8) is folded into STATUS as the terminal state
  `2` (ended); opcode 8 is free again.
- Section 5.4 now frames `channels` as the receiver's speaker layout and has
  producers resolve source-layout mismatches by deriving roles (nearest mix,
  ultimately mono) instead of omitting them — a mapped speaker always plays.

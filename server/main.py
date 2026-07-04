"""
LiveCC server — YouTube/livestream -> ComputerCraft monitor transcoder.

CC's http.get caps responses at http.max_response_size (16 MiB) and buffers the
whole body, so streaming uses a single multiplexed WebSocket.  Server-side
buffering, A/V sync and pacing live in session.StreamSession.

The socket speaks the CCMF Stream Format (docs/cc-media-format.md §5): every
message is binary.  The client opens with a lock-step handshake —

    C→S  ROOM   (url, start, end, loop, sync)
    S→C  ACK | ERROR
    C→S  CAPS   (capabilities + preferences: streams, codecs, W/H/fps)
    S→C  ACK | ERROR          <- accept-or-drop decision
    C→S  START

— after which it is a passive subscriber to the media hose: CCMF container
chunks (video GOPs / audio) pass through verbatim as marker-led MEDIA messages,
interleaved with STATUS (buffering/playing), ERROR and END control frames.
A client may send QUIT to leave cleanly.

Rooms (spec §5.5): sync=0 gives the client a private production; sync=1 joins
the shared room keyed by (url, start, end, loop).  A joiner whose CAPS diverge
from the room's running production is dropped with ERROR (it may retry with
sync=0) — reconciliation is out of scope.

Both Lua scripts are served with this server's own URL baked in, so the player
is installed and run as `livecc <url>` (no address argument).

HTTP endpoints:
  GET /player   → player.lua  (server URL injected)
  GET /install  → install.lua (wget run <server>/install)
  GET /health   → {"status":"ok"}
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse

import ccmf
from session import StreamSession
from transcoder import DFPWM, PCM

app = FastAPI(title="LiveCC", version="4.0.0")

_LUA_DIR = Path(__file__).parent.parent / "lua"


class _BroadcastSocket:
    """StreamSession sink that fan-outs messages to all subscribed websockets."""

    def __init__(self, group: "_SyncGroup") -> None:
        self._group = group

    async def send_bytes(self, payload: bytes) -> None:
        await self._group.send_bytes(payload)


class _SyncGroup:
    """A shared production for sync=1 clients in the same room."""

    def __init__(self, key: tuple, signature: tuple, session: StreamSession) -> None:
        self.key = key                       # (url, start, end, loop), spec §5.5
        self.signature = signature           # production settings from CAPS
        self.session = session
        self._subs: set[WebSocket] = set()
        self._subs_lock = asyncio.Lock()
        self._run_task: Optional[asyncio.Task] = None

    async def subscribe(self, ws: WebSocket) -> None:
        async with self._subs_lock:
            self._subs.add(ws)

    async def unsubscribe(self, ws: WebSocket) -> bool:
        async with self._subs_lock:
            self._subs.discard(ws)
            empty = not self._subs
        if empty and self._run_task and not self._run_task.done():
            self._run_task.cancel()
        return empty

    async def send_bytes(self, payload: bytes) -> None:
        # One serialization, identical bytes to every subscriber (spec §5.5).
        async with self._subs_lock:
            subs = list(self._subs)
        if not subs:
            return

        results = await asyncio.gather(*(ws.send_bytes(payload) for ws in subs),
                                       return_exceptions=True)
        dead = [ws for ws, res in zip(subs, results) if isinstance(res, Exception)]
        if dead:
            async with self._subs_lock:
                for ws in dead:
                    self._subs.discard(ws)

    def start_if_needed(self) -> None:
        if self._run_task is None or self._run_task.done():
            self._run_task = asyncio.create_task(self.session.run(_BroadcastSocket(self)))


_sync_groups: dict[tuple, _SyncGroup] = {}
_sync_groups_lock = asyncio.Lock()


def _clamp(value: int, default: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi)) if value else default


def _negotiate(room: ccmf.Room, caps: ccmf.Caps) -> Optional[dict]:
    """CAPS + ROOM -> StreamSession kwargs, or None when there is nothing the
    server could send (the accept-or-drop decision, spec §5.3).

    Audio is optional: the codec is picked from the client's mask (pcm8
    preferred — the speaker's native format — else dfpwm); a client that
    advertises no usable codec, or no mono channel role (the mandatory
    fallback, spec §5.4), simply gets no audio.  Positional channel roles
    (front_left/right, 5.1, 7.1) aren't decided here — the client's raw
    `channels` bitmask is just forwarded as caps_channels; StreamSession.run()
    negotiates the actual role set once it knows the source's real channel
    count (transcoder.negotiate_channel_roles).
    """
    codec = None
    if caps.audio_mask & ccmf.CAP_AUDIO_PCM8:
        codec = PCM
    elif caps.audio_mask & ccmf.CAP_AUDIO_DFPWM:
        codec = DFPWM
    want_audio = (caps.want_audio and codec is not None
                  and bool(caps.channels & ccmf.CAP_CHANNEL_MONO))
    if not caps.want_video and not want_audio:
        return None
    return {
        "url": room.url,
        "w": _clamp(caps.width, 51, 1, 500),
        "h": _clamp(caps.height, 19, 1, 500),
        "fps": _clamp(caps.fps, 24, 1, 30),
        "want_audio": want_audio,
        "want_video": caps.want_video,
        "start": (room.start_ms or 0) / 1000.0,
        "end": room.end_ms / 1000.0 if room.end_ms is not None else None,
        "loop": room.loop,
        "audio_codec": codec or PCM,
        "caps_channels": caps.channels,
    }


def _sync_signature(kwargs: dict) -> tuple:
    """Production settings a shared room is locked to.  (url/start/end/loop are
    already the room key; this is the CAPS-derived remainder.)"""
    return (
        kwargs["w"],
        kwargs["h"],
        kwargs["fps"],
        kwargs["want_audio"],
        kwargs["want_video"],
        kwargs["audio_codec"].name,
        kwargs["caps_channels"],
    )


async def _acquire_sync_group(room: ccmf.Room, kwargs: dict) -> Optional[_SyncGroup]:
    key, sig = room.key(), _sync_signature(kwargs)
    async with _sync_groups_lock:
        group = _sync_groups.get(key)
        if group is None:
            group = _SyncGroup(key, sig, StreamSession(**kwargs))
            _sync_groups[key] = group
            return group
        if group.signature != sig:
            return None
        return group


async def _release_sync_group(key: tuple, ws: WebSocket) -> None:
    async with _sync_groups_lock:
        group = _sync_groups.get(key)
        if group is None:
            return

    empty = await group.unsubscribe(ws)
    if not empty:
        return

    async with _sync_groups_lock:
        if _sync_groups.get(key) is group:
            _sync_groups.pop(key, None)


# --------------------------------------------------------------------------- #
# Stream Format plumbing
# --------------------------------------------------------------------------- #

async def _recv_control(ws: WebSocket) -> Optional[tuple[int, bytes]]:
    """Receive one control frame -> (opcode, body); None once the socket closes
    or the client sends something unparseable."""
    try:
        while True:
            message = await ws.receive()
            if message.get("type") != "websocket.receive":
                return None                     # websocket.disconnect
            data = message.get("bytes")
            if data is None:
                continue                        # text frames aren't part of CCMF
            return ccmf.parse_message(data)
    except Exception:
        return None


async def _send_control(ws: WebSocket, opcode: int, body: bytes = b"") -> None:
    await ws.send_bytes(ccmf.control(opcode, body))


async def _reject(ws: WebSocket, message: str) -> None:
    """ERROR + drop.  Bodies stay sanitized (spec §7): protocol-level phrasing
    only, never internal detail."""
    try:
        await _send_control(ws, ccmf.OP_ERROR, message.encode("utf-8"))
        await ws.close(code=1008)
    except Exception:
        pass


async def _handshake(ws: WebSocket) -> Optional[tuple[ccmf.Room, ccmf.Caps]]:
    """Run the lock-step ROOM/CAPS/START handshake; None if it fails (the socket
    is already ERROR'd + closed).  The CAPS ACK (accept-or-drop) is deferred to
    the caller, which knows whether the requested room can take this client."""
    got = await _recv_control(ws)
    if got is None or got[0] != ccmf.OP_ROOM:
        await _reject(ws, "Expected ROOM.")
        return None
    try:
        room = ccmf.parse_room(got[1])
    except ValueError:
        await _reject(ws, "Malformed ROOM.")
        return None
    await _send_control(ws, ccmf.OP_ACK)

    got = await _recv_control(ws)
    if got is None or got[0] != ccmf.OP_CAPS:
        await _reject(ws, "Expected CAPS.")
        return None
    try:
        caps = ccmf.parse_caps(got[1])
    except ValueError:
        await _reject(ws, "Malformed CAPS.")
        return None
    return room, caps


async def _await_start(ws: WebSocket) -> bool:
    got = await _recv_control(ws)
    if got is None or got[0] != ccmf.OP_START:
        await _reject(ws, "Expected START.")
        return False
    return True


async def _until_quit_or_disconnect(ws: WebSocket) -> None:
    """Resolve when the client goes away or sends QUIT.  After START the client
    is a passive subscriber, so anything else it sends is ignored."""
    while True:
        got = await _recv_control(ws)
        if got is None or got[0] == ccmf.OP_QUIT:
            return


@app.websocket("/ws/play")
async def ws_play(websocket: WebSocket):
    await websocket.accept()

    hello = await _handshake(websocket)
    if hello is None:
        return
    room, caps = hello

    kwargs = _negotiate(room, caps)
    if kwargs is None:
        await _reject(websocket, "Nothing to play (no usable stream requested).")
        return

    if room.sync:
        group = await _acquire_sync_group(room, kwargs)
        if group is None:
            await _reject(websocket,
                          "Room settings mismatch; retry without sync.")
            return
        await _send_control(websocket, ccmf.OP_ACK)      # CAPS accepted

        if not await _await_start(websocket):
            await _release_sync_group(room.key(), websocket)
            return
        await group.subscribe(websocket)
        group.start_if_needed()
        try:
            await _until_quit_or_disconnect(websocket)
        finally:
            await _release_sync_group(room.key(), websocket)
            try:
                await websocket.close()
            except RuntimeError:
                pass
        return

    await _send_control(websocket, ccmf.OP_ACK)          # CAPS accepted
    if not await _await_start(websocket):
        return

    session = StreamSession(**kwargs)

    # Run the session alongside a disconnect/QUIT watcher.  Whichever finishes
    # first (stream ends, or the client leaves) cancels the other — so an aborted
    # CC program promptly tears the session down (kills yt-dlp/ffmpeg, frees the
    # temp file) instead of leaving it running against a dead socket.
    run_task = asyncio.create_task(session.run(websocket))
    watch_task = asyncio.create_task(_until_quit_or_disconnect(websocket))
    try:
        await asyncio.wait({run_task, watch_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in (run_task, watch_task):
            if not t.done():
                t.cancel()
        await asyncio.gather(run_task, watch_task, return_exceptions=True)
        try:
            await websocket.close()
        except RuntimeError:
            pass


def _server_base_url(request: Request) -> str:
    """Public base URL of this server, honouring reverse-proxy headers.

    Behind a TLS-terminating proxy uvicorn only sees http, so trust
    X-Forwarded-Proto/Host when present to keep https (and thus wss) intact.
    """
    fwd_proto = request.headers.get("x-forwarded-proto", "")
    fwd_host = request.headers.get("x-forwarded-host", "")
    scheme = fwd_proto.split(",")[0].strip() or request.url.scheme
    host = fwd_host.split(",")[0].strip() or request.headers.get("host", "")
    if host:
        return f"{scheme}://{host}"
    return str(request.base_url).rstrip("/")


def _serve_lua(name: str, request: Request) -> PlainTextResponse:
    """Serve a Lua script with the server's own URL baked in."""
    template = (_LUA_DIR / name).read_text()
    return PlainTextResponse(
        template.replace('"{{SERVER}}"', f'"{_server_base_url(request)}"')
    )


@app.get("/player")
async def player(request: Request):
    # The server URL is hardcoded into the saved script, so users run
    # `livecc <url>` without passing the address.
    return _serve_lua("player.lua", request)


@app.get("/install")
async def install(request: Request):
    return _serve_lua("install.lua", request)


@app.get("/health")
async def health():
    return {"status": "ok"}

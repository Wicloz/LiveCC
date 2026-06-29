"""
LiveCC server — YouTube/livestream -> ComputerCraft monitor transcoder.

CC's http.get caps responses at http.max_response_size (16 MiB) and buffers the
whole body, so streaming uses a single multiplexed WebSocket.  Server-side
buffering, A/V sync and pacing live in session.StreamSession.

WebSocket endpoint (used by the CC player):
    /ws/play?url=<url>[&width=51&height=19&fps=10&audio=1&video=1&start=&end=&loop=0&crunchy=0&sync=0]
      audio/video : "0" to disable that stream (audio-only / video-only).
      start/end : timestamps ("90", "1m30s", "3h2m"); seek a VOD section.
      loop      : "1" to cache the section once and replay it forever.
      crunchy   : "1" for low-bandwidth/low-fidelity mode — 1-bit DFPWM audio and
                  CC's fixed default palette instead of the adaptive per-frame one
                  (the client also lowers video resolution via a smaller width/height).
            sync      : "1" to share one provider across clients with the same URL.
  The binary side of the socket is a sanjuuni 32vid stream: a 12-byte "32VD" file
  header (W, H, fps, nstreams, flags) followed by self-delimiting chunks, each
  <size, datalength, type> + data, interleaving video and audio:
      video chunk -> binary  32vid chunk, type 0 — one uncompressed frame (packed
                     5-bit screen + per-cell colour byte + 48-byte palette)
      audio chunk -> binary  32vid chunk, type 1 — raw 8-bit PCM, or DFPWM if crunchy
      status      -> text    "BUFFERING" / "PLAYING" / "ERROR ..."  (out-of-band)

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

from session import StreamSession

app = FastAPI(title="LiveCC", version="3.0.0")

_LUA_DIR = Path(__file__).parent.parent / "lua"


class _BroadcastSocket:
    """StreamSession sink that fan-outs messages to all subscribed websockets."""

    def __init__(self, group: "_SyncGroup") -> None:
        self._group = group

    async def send_text(self, message: str) -> None:
        await self._group.send_text(message)

    async def send_bytes(self, payload: bytes) -> None:
        await self._group.send_bytes(payload)


class _SyncGroup:
    """A shared stream for sync=1 clients watching the same URL."""

    def __init__(self, url: str, signature: tuple, session: StreamSession) -> None:
        self.url = url
        self.signature = signature
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

    async def send_text(self, message: str) -> None:
        await self._broadcast("text", message)

    async def send_bytes(self, payload: bytes) -> None:
        await self._broadcast("bytes", payload)

    async def _broadcast(self, mode: str, payload) -> None:
        async with self._subs_lock:
            subs = list(self._subs)
        if not subs:
            return

        async def _send(ws: WebSocket):
            if mode == "text":
                await ws.send_text(payload)
            else:
                await ws.send_bytes(payload)

        results = await asyncio.gather(*(_send(ws) for ws in subs), return_exceptions=True)

        dead = [ws for ws, res in zip(subs, results) if isinstance(res, Exception)]
        if dead:
            async with self._subs_lock:
                for ws in dead:
                    self._subs.discard(ws)

    def start_if_needed(self) -> None:
        if self._run_task is None or self._run_task.done():
            self._run_task = asyncio.create_task(self.session.run(_BroadcastSocket(self)))

    async def wait(self) -> None:
        if self._run_task is not None:
            await self._run_task


_sync_groups: dict[str, _SyncGroup] = {}
_sync_groups_lock = asyncio.Lock()


def _int(qp, key: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(int(qp.get(key, default)), hi))
    except (TypeError, ValueError):
        return default


async def _until_disconnect(ws: WebSocket) -> None:
    """Resolve when the client goes away.  Our protocol is server->client only,
    so receiving simply blocks until the socket closes."""
    try:
        while True:
            await ws.receive()
    except Exception:
        return


def _session_kwargs(qp, url: str) -> dict:
    return {
        "url": url,
        "w": _int(qp, "width", 51, 1, 500),
        "h": _int(qp, "height", 19, 1, 500),
        "fps": _int(qp, "fps", 24, 1, 30),
        "want_audio": qp.get("audio", "1") != "0",
        "want_video": qp.get("video", "1") != "0",
        "start": qp.get("start", ""),
        "end": qp.get("end", ""),
        "loop": qp.get("loop", "0") == "1",
        "crunchy": qp.get("crunchy", "0") == "1",
    }


def _sync_signature(kwargs: dict) -> tuple:
    return (
        kwargs["w"],
        kwargs["h"],
        kwargs["fps"],
        kwargs["want_audio"],
        kwargs["want_video"],
        kwargs["start"],
        kwargs["end"],
        kwargs["loop"],
        kwargs["crunchy"],
    )


async def _acquire_sync_group(url: str, kwargs: dict) -> Optional[_SyncGroup]:
    sig = _sync_signature(kwargs)
    async with _sync_groups_lock:
        group = _sync_groups.get(url)
        if group is None:
            group = _SyncGroup(url, sig, StreamSession(**kwargs))
            _sync_groups[url] = group
            return group
        if group.signature != sig:
            return None
        return group


async def _release_sync_group(url: str, ws: WebSocket) -> None:
    async with _sync_groups_lock:
        group = _sync_groups.get(url)
        if group is None:
            return

    empty = await group.unsubscribe(ws)
    if not empty:
        return

    async with _sync_groups_lock:
        if _sync_groups.get(url) is group:
            _sync_groups.pop(url, None)


@app.websocket("/ws/play")
async def ws_play(websocket: WebSocket):
    await websocket.accept()
    qp = websocket.query_params
    url = qp.get("url")
    if not url:
        await websocket.close(code=1008)
        return

    kwargs = _session_kwargs(qp, url)

    if qp.get("sync", "0") == "1":
        group = await _acquire_sync_group(url, kwargs)
        if group is None:
            await websocket.send_text(
                "ERROR Sync clients on the same URL must use identical playback settings."
            )
            await websocket.close(code=1008)
            return

        await group.subscribe(websocket)
        group.start_if_needed()
        try:
            await _until_disconnect(websocket)
        finally:
            await _release_sync_group(url, websocket)
            try:
                await websocket.close()
            except RuntimeError:
                pass
        return

    session = StreamSession(**kwargs)

    # Run the session alongside a disconnect watcher.  Whichever finishes first
    # (stream ends, or the client terminates) cancels the other — so an aborted
    # CC program promptly tears the session down (kills yt-dlp/ffmpeg, frees the
    # temp file) instead of leaving it running against a dead socket.
    run_task = asyncio.create_task(session.run(websocket))
    watch_task = asyncio.create_task(_until_disconnect(websocket))
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

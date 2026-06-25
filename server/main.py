"""
LiveCC server — YouTube/livestream -> ComputerCraft monitor transcoder.

CC's http.get caps responses at http.max_response_size (16 MiB) and buffers the
whole body, so streaming uses a single multiplexed WebSocket.  Server-side
buffering, A/V sync and pacing live in session.StreamSession.

WebSocket endpoint (used by the CC player):
  /ws/play?url=<url>[&width=51&height=19&fps=10&audio=1&start=&end=&loop=0]
      start/end : timestamps ("90", "1m30s", "3h2m"); seek a VOD section.
      loop      : "1" to cache the section once and replay it forever.
      video frame -> binary  b"\\x01" + frame
      audio chunk -> binary  b"\\x02" + dfpwm
      status      -> text    "META ..." / "BUFFERING" / "PLAYING" / "ERROR ..."

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

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse

from session import StreamSession

app = FastAPI(title="LiveCC", version="3.0.0")

_LUA_DIR = Path(__file__).parent.parent / "lua"


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


@app.websocket("/ws/play")
async def ws_play(websocket: WebSocket):
    await websocket.accept()
    qp = websocket.query_params
    url = qp.get("url")
    if not url:
        await websocket.close(code=1008)
        return

    session = StreamSession(
        url=url,
        w=_int(qp, "width", 51, 1, 500),
        h=_int(qp, "height", 19, 1, 500),
        fps=_int(qp, "fps", 24, 1, 30),
        want_audio=qp.get("audio", "1") != "0",
        start=qp.get("start", ""),
        end=qp.get("end", ""),
        loop=qp.get("loop", "0") == "1",
    )

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

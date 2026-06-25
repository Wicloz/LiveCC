import asyncio

import main


class _FakeURL:
    def __init__(self, scheme):
        self.scheme = scheme


class _FakeRequest:
    """Minimal stand-in for a Starlette Request for the URL helpers."""

    def __init__(self, headers=None, scheme="http", base_url="http://fallback:8080/"):
        self.headers = headers or {}
        self.url = _FakeURL(scheme)
        self.base_url = base_url


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
# Disconnect watcher
# --------------------------------------------------------------------------- #

def test_until_disconnect_returns_when_socket_closes():
    class WS:
        async def receive(self):
            raise RuntimeError("connection closed")

    # Must resolve (not hang or raise) so the session gets torn down on disconnect.
    asyncio.run(asyncio.wait_for(main._until_disconnect(WS()), timeout=1))


# --------------------------------------------------------------------------- #
# Query param parsing / clamping
# --------------------------------------------------------------------------- #

def test_int_default_clamp_and_bad_value():
    assert main._int({}, "fps", 24, 1, 30) == 24          # default
    assert main._int({"fps": "100"}, "fps", 24, 1, 30) == 30   # clamp high
    assert main._int({"fps": "0"}, "fps", 24, 1, 30) == 1      # clamp low
    assert main._int({"fps": "x"}, "fps", 24, 1, 30) == 24     # non-numeric

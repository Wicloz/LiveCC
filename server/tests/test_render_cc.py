"""
render_cc preview tool — argument wiring.

Just the plumbing from main() to render(): sample discovery, grid parsing, and
the failure exit code.  The heavy ffmpeg render itself is exercised by the
sample tests / run by hand.  No ffmpeg needed here: `render` is stubbed so we
only check what main() asks it to do.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_RENDER_CC = Path(__file__).resolve().parent.parent / "tools" / "render_cc.py"
_spec = importlib.util.spec_from_file_location("render_cc", _RENDER_CC)
render_cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_cc)


def _run(monkeypatch, tmp_path, argv_extra, result="stub.mp4"):
    seen = []

    def fake_render(sample, w, h, fps, seconds, outdir):
        seen.append((sample, w, h, fps, seconds))
        return (outdir / result) if result else None   # non-None == "succeeded"

    monkeypatch.setattr(render_cc, "render", fake_render)
    monkeypatch.setattr(render_cc, "have_ffmpeg", lambda: True)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a clip")        # only needs to exist (render stubbed)
    rc = render_cc.main([str(clip), "--seconds", "1",
                         "--outdir", str(tmp_path), *argv_extra])
    return rc, seen


def test_renders_each_requested_grid(monkeypatch, tmp_path):
    rc, seen = _run(monkeypatch, tmp_path, ["--grids", "pocket,10x4", "--fps", "12"])
    assert rc == 0
    assert [(w, h) for _s, w, h, _f, _sec in seen] == [(26, 20), (10, 4)]
    assert all(fps == 12 for _s, _w, _h, fps, _sec in seen)


def test_failed_render_exits_nonzero(monkeypatch, tmp_path):
    rc, seen = _run(monkeypatch, tmp_path, ["--grids", "pocket"], result=None)
    assert rc == 1
    assert len(seen) == 1

"""
render_cc preview tool — argument wiring.

Just the policy that --crunchy selects the fixed palette (adaptive off) while the
default uses the adaptive per-frame palette; the heavy ffmpeg render itself is
exercised by the sample tests / run by hand.  No ffmpeg needed here: `render` is
stubbed so we only check what main() asks it to do.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_RENDER_CC = Path(__file__).resolve().parent.parent / "tools" / "render_cc.py"
_spec = importlib.util.spec_from_file_location("render_cc", _RENDER_CC)
render_cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_cc)


def _capture_adaptive(monkeypatch, tmp_path, argv_extra):
    seen = {}

    def fake_render(sample, w, h, fps, seconds, crunchy, outdir, adaptive=True):
        seen["crunchy"] = crunchy
        seen["adaptive"] = adaptive
        return outdir / "stub.mp4"               # non-None == "render succeeded"

    monkeypatch.setattr(render_cc, "render", fake_render)
    monkeypatch.setattr(render_cc, "have_ffmpeg", lambda: True)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a clip")        # only needs to exist (render stubbed)
    rc = render_cc.main([str(clip), "--grids", "pocket", "--seconds", "1",
                         "--outdir", str(tmp_path), *argv_extra])
    assert rc == 0
    return seen


def test_default_uses_adaptive_palette(monkeypatch, tmp_path):
    seen = _capture_adaptive(monkeypatch, tmp_path, [])
    assert seen["crunchy"] is False
    assert seen["adaptive"] is True


def test_crunchy_uses_fixed_palette(monkeypatch, tmp_path):
    seen = _capture_adaptive(monkeypatch, tmp_path, ["--crunchy"])
    assert seen["crunchy"] is True
    assert seen["adaptive"] is False             # --crunchy drops the adaptive palette

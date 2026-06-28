"""
ComputerCraft preview renderer.

Emulates exactly what a CC monitor + speaker would show for a clip, and writes it
back out as an MP4 you can scrub through — handy for eyeballing the transcoder at
different resolutions without a Minecraft world.

Pipeline per (sample, grid), reusing the *real* server code:
  ffmpeg decode + area-scale + letterbox  (transcoder._video_ffmpeg_cmd)
   -> _FrameSplitter                       (same split as iter_video)
   -> encode_frame                         (the actual blit encoder)
   -> decode_frame                         (blit back to the 16-colour pixels the
                                            client paints: 2 colours/cell in the
                                            glyph pattern, dither and all)
   -> nearest-neighbour upscale            (so the chunky monitor pixels are visible)
   -> ffmpeg x264 mux                       (+ audio emulated as the CC speaker hears
                                            it: 8-bit 48 kHz mono PCM, or 1-bit DFPWM
                                            with --crunchy)

Output goes to media/cc_preview/<clip>_<W>x<H>.mp4 (git-ignored).

Examples:
  python tools/render_cc.py                       # every sample, default grids
  python tools/render_cc.py big_jungus.mp4 --grids 51x19,82x41
  python tools/render_cc.py --grids all --seconds 5 --crunchy
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make the server modules importable when run as a standalone script.
_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from cc_media import (  # noqa: E402  (needs the sys.path insert above)
    GRIDS, MEDIA_DIR, PREVIEW_DIR, decode_frame, find_media, have_ffmpeg)
from cc_encoder import encode_frame  # noqa: E402
from transcoder import _FrameSplitter, _video_ffmpeg_cmd  # noqa: E402

# Named grid presets (from cc_media.GRIDS) plus free-form WxH parsing.
_PRESETS = {label.split()[0]: (w, h) for label, w, h in GRIDS}
_DEFAULT_GRIDS = "pocket,164×81,143x52"
_TARGET_WIDTH = 480       # upscale each preview to roughly this many pixels wide


def _parse_grids(spec: str) -> list[tuple[int, int]]:
    if spec.strip().lower() == "all":
        return [(w, h) for _, w, h in GRIDS]
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in _PRESETS:
            out.append(_PRESETS[tok])
        elif "x" in tok:
            w, h = tok.split("x")
            out.append((int(w), int(h)))
        else:
            raise SystemExit(f"bad grid '{tok}' (use WxH, a preset name, or 'all')")
    return out


def _even_factor(w: int, h: int) -> int:
    """Integer upscale so the W*2-wide monitor lands near _TARGET_WIDTH, forcing
    even output dims (x264 + yuv420p require them)."""
    k = max(1, round(_TARGET_WIDTH / (w * 2)))
    if (h * 3 * k) % 2:        # W*2*k is always even; only H*3*k can be odd
        k += 1
    return k


def _crunchy_dfpwm(sample: Path) -> Path | None:
    """Pre-render the sample's audio to 1-bit DFPWM (what --crunchy sends), so the
    mux can decode it back and we hear the real crunch.  None if no audio."""
    fd, name = tempfile.mkstemp(prefix="ccprev_", suffix=".dfpwm")
    os.close(fd)               # close our handle or Windows won't let us unlink it
    tmp = Path(name)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(sample),
           "-vn", "-ar", "48000", "-ac", "1", "-c:a", "dfpwm", "-f", "dfpwm", str(tmp)]
    r = subprocess.run(cmd, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return None
    return tmp


def _output_cmd(out: Path, ow: int, oh: int, fps: int, sample: Path,
                crunchy_audio: Path | None) -> list[str]:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ow}x{oh}", "-r", str(fps),
           "-i", "pipe:0"]
    if crunchy_audio is not None:                  # 1-bit DFPWM, decoded back
        cmd += ["-f", "dfpwm", "-ar", "48000", "-ac", "1", "-i", str(crunchy_audio),
                "-map", "0:v:0", "-map", "1:a:0?"]
    else:                                          # 8-bit PCM crunch via the filter
        cmd += ["-i", str(sample), "-map", "0:v:0", "-map", "1:a:0?",
                "-af", "aformat=sample_fmts=u8:sample_rates=48000:channel_layouts=mono"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            "-preset", "veryfast", "-c:a", "aac", "-b:a", "96k",
            "-shortest", str(out)]
    return cmd


def render(sample: Path, w: int, h: int, fps: int, seconds: float,
           crunchy: bool, outdir: Path) -> Path | None:
    px_w, px_h = w * 2, h * 3
    k = _even_factor(w, h)
    ow, oh = px_w * k, px_h * k
    max_frames = max(1, int(round(fps * seconds)))
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{sample.stem}_{w}x{h}{'_crunchy' if crunchy else ''}.mp4"

    crunchy_audio = _crunchy_dfpwm(sample) if crunchy else None
    dec = subprocess.Popen(
        _video_ffmpeg_cmd(px_w, px_h, fps, source=str(sample)),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(
        _output_cmd(out, ow, oh, fps, sample, crunchy_audio),
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    splitter = _FrameSplitter(px_w, px_h)
    written = 0
    try:
        while written < max_frames:
            chunk = dec.stdout.read(65536)
            if not chunk:
                break
            for arr in splitter.push(chunk):
                if written >= max_frames:
                    break
                img = decode_frame(encode_frame(arr))      # what the monitor paints
                up = np.repeat(np.repeat(img, k, axis=0), k, axis=1)
                enc.stdin.write(up.tobytes())
                written += 1
    except BrokenPipeError:
        pass
    finally:
        dec.kill()
        dec.wait()
        if enc.stdin:
            enc.stdin.close()
        err = enc.stderr.read().decode("utf-8", "replace") if enc.stderr else ""
        enc.wait()
        if crunchy_audio is not None:
            try:
                crunchy_audio.unlink(missing_ok=True)
            except OSError:
                pass           # best-effort temp cleanup

    if enc.returncode != 0 or written == 0:
        print(f"  ! failed {out.name}: {err.strip()[:200] or 'no frames'}")
        return None
    print(f"  -> {out.relative_to(MEDIA_DIR.parent)}  "
          f"({written} frames, {ow}x{oh}, x{k})")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render CC monitor/speaker previews to MP4.")
    ap.add_argument("samples", nargs="*", help="clip name(s) in media/, or paths "
                    "(default: every video in media/)")
    ap.add_argument("--grids", default=_DEFAULT_GRIDS,
                    help=f"comma list of WxH or preset names, or 'all' "
                         f"(default: {_DEFAULT_GRIDS})")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=10.0, help="duration cap")
    ap.add_argument("--crunchy", action="store_true",
                    help="emulate --crunchy audio (1-bit DFPWM round-trip)")
    ap.add_argument("--outdir", default=str(PREVIEW_DIR))
    args = ap.parse_args(argv)

    if not have_ffmpeg():
        print("render_cc: ffmpeg not found on PATH.")
        return 1

    if args.samples:
        samples = []
        for s in args.samples:
            p = Path(s)
            if not p.is_file():
                p = MEDIA_DIR / s
            if not p.is_file():
                print(f"render_cc: sample not found: {s}")
                return 1
            samples.append(p)
    else:
        samples = find_media("video")

    if not samples:
        print(f"render_cc: no samples in {MEDIA_DIR} (pass a path, or drop clips there).")
        return 1

    grids = _parse_grids(args.grids)
    outdir = Path(args.outdir)
    print(f"Rendering {len(samples)} sample(s) x {len(grids)} grid(s) "
          f"-> {outdir}")
    for sample in samples:
        print(sample.name)
        for w, h in grids:
            render(sample, w, h, args.fps, args.seconds, args.crunchy, outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Frame splitter benchmark (transcoder._FrameSplitter).

ffmpeg writes raw rgb24 to a pipe; the splitter accumulates bytes and emits each
complete frame as an (H, W, 3) array.  It runs on the read path before the encode
offload, so its throughput bounds how fast frames can be fed in.  Measured two
ways: one whole frame at a time, and in 64 KiB chunks (ffmpeg's real read size,
which exercises the buffering/copy path).

Run:  python benchmarks/bench_splitter.py
"""

from __future__ import annotations

import numpy as np

import _bench
from _bench import GRIDS, fmt, measure, section, table

from transcoder import _FrameSplitter

_CHUNK = 65536   # matches iter_video's ffmpeg read size


def main() -> None:
    section("Frame splitter throughput")
    rows = []
    rng = np.random.default_rng(0)
    for label, w, h in GRIDS:
        px_w, px_h = w * 2, h * 3
        frame_bytes = px_w * px_h * 3
        raw = rng.integers(0, 256, frame_bytes, np.uint8).tobytes()

        def whole():
            s = _FrameSplitter(px_w, px_h)
            return list(s.push(raw))

        def chunked():
            s = _FrameSplitter(px_w, px_h)
            out = []
            for i in range(0, len(raw), _CHUNK):
                out += list(s.push(raw[i:i + _CHUNK]))
            return out

        rw = measure(whole)
        rc = measure(chunked)
        mbps = (frame_bytes / 1e6) / (rc["mean_ms"] / 1e3)
        rows.append([
            label, f"{frame_bytes / 1024:.0f}",
            fmt(rw["mean_ms"], 3), fmt(rc["mean_ms"], 3), fmt(mbps, 0),
        ])
    print(table(
        ["grid", "KB/frm", "whole ms", "chunked ms", "MB/s"], rows))
    print("\n(MB/s is the chunked path; frames decode far faster than they encode.)")


if __name__ == "__main__":
    main()

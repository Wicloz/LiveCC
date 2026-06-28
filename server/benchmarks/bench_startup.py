"""
Startup cost benchmark.

Importing cc_encoder precomputes the OKLab lookup table (64^3 x 3 floats), which
dominates first-import time.  The server almost never restarts, so this is a
one-time cost — measured here so it stays visible if it ever grows.

  * cold import — fresh interpreter, so the LUT build is included (subprocess).
  * LUT build   — the table builder alone, in-process.

Run:  python benchmarks/bench_startup.py
"""

from __future__ import annotations

import subprocess
import sys
import time

import _bench
from _bench import _SERVER_DIR, fmt, measure, section, table

import cc_encoder


def _cold_import_ms(module: str, runs: int = 3) -> float:
    code = f"import time;t=time.perf_counter();import {module};" \
           f"print(time.perf_counter()-t)"
    best = float("inf")
    for _ in range(runs):
        out = subprocess.run([sys.executable, "-c", code],
                             cwd=str(_SERVER_DIR), capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            best = min(best, float(out.stdout.strip()))
    return best * 1e3


def main() -> None:
    section("Startup / one-time costs")
    rows = [
        ["cold import cc_encoder", fmt(_cold_import_ms("cc_encoder"))],
        ["cold import transcoder", fmt(_cold_import_ms("transcoder"))],
    ]
    lut = measure(cc_encoder._build_oklab_lut, target_s=1.0, warmup=1)
    rows.append(["_build_oklab_lut (in-proc)", fmt(lut["mean_ms"])])
    nbytes = cc_encoder._OKLAB_LUT.nbytes
    print(table(["item", "ms"], rows))
    print(f"\nOKLab LUT footprint: {nbytes / 1024 / 1024:.1f} MiB resident.")


if __name__ == "__main__":
    main()

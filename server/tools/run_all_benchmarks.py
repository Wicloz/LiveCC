"""
Run the whole benchmark suite.

  python tools/run_benchmarks.py            # everything
  python tools/run_benchmarks.py encoder    # one or more sections by name
  python tools/run_benchmarks.py --profile  # add the encoder + gop cProfile dumps

Sections: encoder (primary), gop, quality, samples, compression, splitter,
buffer, startup.  ("samples"/"compression" use clips in media/; they self-skip
without them / ffmpeg.)
"""

from __future__ import annotations

import sys
from pathlib import Path

# The bench_*/harness modules live in ../benchmarks, not next to this script.
_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

import harness  # noqa: F401  (puts the server dir on sys.path)

import bench_buffer
import bench_compression
import bench_encoder
import bench_gop
import bench_quality
import bench_samples
import bench_splitter
import bench_startup

SECTIONS = {
    "encoder": bench_encoder.main,
    "gop": bench_gop.main,
    "quality": bench_quality.main,
    "samples": bench_samples.main,
    "compression": bench_compression.main,
    "splitter": bench_splitter.main,
    "buffer": bench_buffer.main,
    "startup": bench_startup.main,
}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    wanted = args or list(SECTIONS)
    print("LiveCC server benchmarks")
    print("========================")
    for name in wanted:
        fn = SECTIONS.get(name)
        if fn is None:
            print(f"\n[skip] unknown section: {name} (have: {', '.join(SECTIONS)})")
            continue
        fn()


if __name__ == "__main__":
    main()

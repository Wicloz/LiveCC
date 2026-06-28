"""
Run the whole benchmark suite.

  python benchmarks/run_all.py            # everything
  python benchmarks/run_all.py encoder    # one or more sections by name
  python benchmarks/run_all.py --profile  # add the encoder cProfile dump

Sections: encoder (primary), quality, samples, splitter, buffer, startup.
("samples" needs clips in media/ and ffmpeg; it self-skips otherwise.)
"""

from __future__ import annotations

import sys

import harness  # noqa: F401  (puts the server dir on sys.path)

import bench_buffer
import bench_encoder
import bench_quality
import bench_samples
import bench_splitter
import bench_startup

SECTIONS = {
    "encoder": bench_encoder.main,
    "quality": bench_quality.main,
    "samples": bench_samples.main,
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

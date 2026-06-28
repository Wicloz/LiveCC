"""
Scheduler buffer benchmark (session.TimedBuffer).

TimedBuffer is on the per-tick hot path of StreamSession.run: every release pops
all due items, and every produced frame/chunk is put.  It's plain in-memory deque
work, so this just confirms it's nowhere near the cost of encoding — i.e. pacing
overhead is negligible next to the transcoder.

Run:  python benchmarks/bench_buffer.py
"""

from __future__ import annotations

import asyncio
import time

import harness
from harness import fmt, section, table

from session import TimedBuffer


async def _run() -> list:
    rows = []
    n = 200_000

    # Live path: put() with drop_oldest never awaits, so it's a tight loop.
    buf = TimedBuffer(max_items=120, drop_oldest=True)
    t = time.perf_counter()
    for i in range(n):
        await buf.put(i / 24.0, b"x")
    put_ns = (time.perf_counter() - t) / n * 1e9

    # Drain in due-batches the way the scheduler does.
    buf = TimedBuffer(max_items=n + 1, drop_oldest=True)
    for i in range(n):
        await buf.put(i / 24.0, b"x")
    t = time.perf_counter()
    popped = 0
    now = 0.0
    while not buf.empty():
        now += 1.0
        popped += len(buf.pop_due(now))
    pop_ns = (time.perf_counter() - t) / max(popped, 1) * 1e9

    rows.append(["put (live, drop)", fmt(put_ns, 0), fmt(1e9 / put_ns / 1e6, 1)])
    rows.append(["pop_due (per item)", fmt(pop_ns, 0), fmt(1e9 / pop_ns / 1e6, 1)])
    return rows


def main() -> None:
    section("TimedBuffer throughput")
    rows = asyncio.run(_run())
    print(table(["op", "ns/item", "M items/s"], rows))


if __name__ == "__main__":
    main()

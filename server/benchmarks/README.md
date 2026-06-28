# LiveCC server benchmarks

Performance benchmarks for the server's CPU-bound work. The **video transcoder**
(`cc_encoder.encode_frame`) is the primary concern — it's the only per-frame hot
loop — so it gets the most coverage; the rest confirm nothing else is close to it.

Pure stdlib + numpy (already a server dependency). No extra installs.

## Run

```sh
# from the server/ directory
python benchmarks/run_all.py              # everything
python benchmarks/run_all.py encoder      # one or more sections
python benchmarks/run_all.py --profile    # add the encoder cProfile dump

# or a single module directly
python benchmarks/bench_encoder.py
```

Sections: `encoder` (primary), `quality`, `samples`, `splitter`, `buffer`,
`startup`. The `samples` section uses developer clips in the repo-root `media/`
folder and self-skips when it's empty.

> Looking to *see* the output rather than time it? That's the preview renderer,
> `../tools/render_cc.py` (see `media/README.md`) — it's a dev tool, not a
> benchmark, so it lives outside this folder.

## What each measures

| Module | Measures |
| --- | --- |
| `bench_encoder.py` | `encode_frame` cost vs **grid size** (pocket→max monitor) and vs **content** (flat/gradient/edges/photo/random), plus a per-primitive breakdown of where the time goes. Reports the fps ceiling, the adaptive pacer's steady fps, and concurrent-streams-per-core. |
| `bench_quality.py` | Transcoder **fidelity**: decodes the blit wire format back to pixels and reports PSNR (raw and after a 2×2 box blur, which approximates the eye integrating dithered sub-pixels) plus % cells dithered — for synthetic content **and** for any real `media/` samples. |
| `bench_samples.py` | `encode_frame` on **real decoded frames** from `media/` clips — by grid and by clip. Needs ffmpeg + samples; self-skips otherwise. |
| `bench_splitter.py` | `_FrameSplitter` throughput (whole-frame and 64 KiB-chunked) — the read path before the encode offload. |
| `bench_buffer.py` | `TimedBuffer` put / pop_due — the scheduler's per-tick deque work. |
| `bench_startup.py` | One-time import + OKLab LUT build cost and its resident footprint. |

## Shared helpers

`harness.py` holds the timing/table/synthetic-frame helpers used by the benches.
Media discovery, real-frame extraction, and the reference blit *decoder* live in
`../cc_media.py` (shared with the preview renderer and the sample tests);
`harness` re-exports them, so the benches still `from harness import …`. The
preview renderer itself is `../tools/render_cc.py` — see `media/README.md`.

## Reading the numbers

- **Headline:** `bench_encoder` size sweep on `photo` content. `eff fps` is the
  steady rate the transcoder's adaptive pacer (`_encode_stride`) holds at a 24 fps
  target — when a grid is too big to encode in its frame slot it drops to a lower
  but stutter-free fps instead of out-running the encoder into re-buffering.
  `strm@24` is the per-core concurrency budget. Terminals and small/medium
  monitors hold full fps; only the largest monitors pace down.
- **`min ms`** is the cleanest signal (least OS scheduling noise); `mean ms` is
  typical cost. Numbers are single-core; `encode_frame` releases the GIL, so the
  worker pool scales across cores for concurrent streams.
- **PSNR is low in absolute terms** — the output has only 16 colours, so this is
  expected. Use it to compare *relatively* (content types, parameter changes like
  `_DITHER_WEIGHT`). The `PSNR 2x2` column is the better perceptual proxy: it's
  where dithering's benefit over banding shows up.
- The splitter and buffer run in **micro/nanoseconds** — included to confirm the
  pacing/read path is negligible next to encoding, not because they're a concern.

## Not benchmarked (and why)

yt-dlp download, ffmpeg decode/scale/resample, the WebSocket transport, and audio
(raw PCM / DFPWM is produced by ffmpeg) are **I/O- and subprocess-bound**, not
Python CPU work — their cost is network and ffmpeg, not anything this suite could
time meaningfully. The transcoder's own ffmpeg `scale` filter choice lives in
`transcoder._video_ffmpeg_cmd`.

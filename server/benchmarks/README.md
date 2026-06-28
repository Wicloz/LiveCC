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

Sections: `encoder` (primary), `quality`, `splitter`, `buffer`, `startup`.

## What each measures

| Module | Measures |
| --- | --- |
| `bench_encoder.py` | `encode_frame` cost vs **grid size** (pocket→max monitor) and vs **content** (flat/gradient/edges/photo/random), plus a per-primitive breakdown of where the time goes. Reports the fps ceiling and how many concurrent 24 fps streams fit on one core. |
| `bench_quality.py` | Transcoder **fidelity**: decodes the blit wire format back to pixels and reports PSNR (raw and after a 2×2 box blur, which approximates the eye integrating dithered sub-pixels), plus the fraction of cells that dither. |
| `bench_splitter.py` | `_FrameSplitter` throughput (whole-frame and 64 KiB-chunked) — the read path before the encode offload. |
| `bench_buffer.py` | `TimedBuffer` put / pop_due — the scheduler's per-tick deque work. |
| `bench_startup.py` | One-time import + OKLab LUT build cost and its resident footprint. |

## Reading the numbers

- **Headline:** `bench_encoder` size sweep on `photo` content. `strm@24` is the
  per-core concurrency budget; large monitors fall below 24 fps and lean on the
  session's frame-dropping (see the `adaptive-quality` design note). Terminals and
  small/medium monitors have ample headroom.
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

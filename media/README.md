# media/

Drop your own sample clips here — video (`.mp4`, `.mov`, `.mkv`, `.webm`, `.gif`,
…) and/or audio. **Nothing in this folder is committed** except this README and
`.gitignore`; the samples are developer-provided (large and often copyrighted).

What picks them up:

- **Tests** — `server/tests/test_media_samples.py` runs the real transcode
  pipeline over every sample present (skipped entirely if the folder is empty or
  ffmpeg isn't installed).
- **Benchmarks** — `server/benchmarks/bench_samples.py` measures `encode_frame`
  on real decoded frames from your samples (more representative than synthetic).
- **Preview renderer** — `server/tools/render_cc.py` emulates exactly what a
  ComputerCraft monitor/speaker would show at various resolutions and writes MP4s
  to `media/cc_preview/` for manual inspection (8-bit PCM audio, or 1-bit DFPWM
  with `--crunchy`).

```sh
# from the repo root, after putting a clip or two in media/
cd server
python tools/render_cc.py                 # render every sample at default grids
python tools/render_cc.py --help          # options (grids, duration, crunchy, ...)
pytest tests/test_media_samples.py
python benchmarks/bench_samples.py
```

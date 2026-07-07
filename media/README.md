# media/

Drop your own sample clips here — video (`.mp4`, `.mov`, `.mkv`, `.webm`, `.gif`,
…) and/or audio (`.mp3`, …). **Nothing in this folder is committed** except this
README and `.gitignore`; the samples are developer-provided (large and often
copyrighted).

Every file you drop here (except this README and dotfiles) is treated as a sample
you expect to work — there's **no extension allowlist**. Each file is routed by
its actual streams, probed with ffprobe: a clip with a video stream is run through
the video path, one with audio through the audio path (a music file with embedded
cover art counts as audio-only, not video). A file the matching pipeline can't
handle, or one with no audio/video stream at all, is a **genuine bug** and fails
loudly rather than being silently skipped.

What picks them up:

- **Tests** — `server/tests/test_media_samples.py` runs the real transcode
  pipeline over every sample present (skipped entirely if the folder is empty or
  ffmpeg isn't installed).
- **Benchmarks** — `server/benchmarks/bench_samples.py` measures `encode_frame`
  on real decoded frames from your samples (more representative than synthetic).
- **CCMF exporter** — `server/tools/render_cc.py` runs a sample through the real
  transcode pipeline at full speed and writes a `.ccmf` file — the same container
  a live session streams, just produced once instead of paced to a client.

```sh
# from the repo root, after putting a clip or two in media/
cd server
python tools/render_cc.py ../media/<clip> --grid pocket # convert one sample
python tools/render_cc.py --help                       # options (grid, fps, audio, ...)
pytest tests/test_media_samples.py
python benchmarks/bench_samples.py
```

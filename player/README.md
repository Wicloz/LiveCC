# player

A stand-alone desktop media player for local `.ccmf` files ([CCMF](../docs/cc-media-format.md),
the ComputerCraft Media Format LiveCC streams to CC:Tweaked terminals). Built
in C++20 with [raylib](https://www.raylib.com/) for the window/graphics/audio
layer. Plays only the **Container Format** (a bare sequence of chunks) — no
stream/handshake protocol, since a stored file needs none of that.

Video renders exactly as a CC monitor would paint it: each cell's "blit
char" is one of 32 fixed 2x3 sub-pixel dither patterns (not a text glyph),
computed directly from the file's `(glyph, fg, bg, palette)` data — no font
asset is needed.

## Project layout

```
src/engine/   Format parsing, decoding, and the playback/seek engine.
              Zero raylib dependency; 100% unit-testable headlessly.
src/app/      Thin raylib adapter: window, texture upload, audio streaming,
              on-screen controls. Not unit-tested (needs a live window/GPU/
              audio device) -- kept deliberately small; correctness lives
              in the tested engine, this layer just displays it.
tests/        GoogleTest suite for src/engine (CTest-registered), plus a
              few small real .ccmf fixtures rendered with the server's
              convert_to_ccmf.py tool (some cross-checked against the Python
              reference decoder in ../server).
```

## Building

Requires a C++20 compiler and CMake >= 3.21. Dependencies (raylib 5.5,
GoogleTest v1.17.0) are fetched automatically by CMake's `FetchContent` from
GitHub on first configure — no manual install, just internet access once.

### Visual Studio 2022 (recommended)

From a "Developer PowerShell for VS 2022" (or plain PowerShell, pointing at
the bundled `cmake.exe` directly):

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Debug -j 8
```

### MinGW-w64 (g++)

```sh
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j 8
```

> Some community MinGW-w64 builds ship a `libmingw32.a` whose exit-wrapper
> object references UCRT import symbols the rest of that build doesn't
> provide, which fails to link *only* the test binary (GoogleTest is the
> first thing in this project to touch `_Exit`/`quick_exit`). This is a
> toolchain packaging issue, not a project one — if you hit it, either use
> MSVC for the test target or a UCRT-consistent MinGW distribution; the
> `player` app itself is unaffected either way.

## Running

```sh
build/src/app/Debug/player.exe path/to/clip.ccmf     # MSVC (Debug config)
build/src/app/player.exe path/to/clip.ccmf            # MinGW
```

Need a sample file? Render one from the repo's `server/` tooling:

```sh
cd ../server
python tools/convert_to_ccmf.py ../media/<clip> --grid mon4x2 --fps 24 --duration 8 \
    --channels stereo -o ../player/sample.ccmf
```

### Controls

| Input | Action |
|---|---|
| Space, or click the play/pause button | Toggle play/pause |
| L, or click the LOOP button | Toggle looping (restarts from 0 at end-of-file instead of pausing) |
| Left / Right arrow | Seek -5s / +5s |
| Home / End | Seek to start / end |
| Click or drag the scrub bar | Seek to that position (live while dragging) |
| Esc / close window | Quit |

## Testing

```sh
cd build
ctest -C Debug --output-on-failure     # MSVC
ctest --output-on-failure              # MinGW
```

Coverage (GCC/Clang only): configure with `-DPLAYER_COVERAGE=ON` to add
`--coverage` instrumentation to `ccmf_engine` (verified it compiles/links
cleanly on its own); run the tests and inspect the `.gcda`/`.gcno` output
with `gcov` (ships with GCC) or `pip install gcovr` for an HTML/summary
report. If your MinGW distribution hits the linker issue noted above, it
blocks the *test binary* the same way with or without coverage on — that's
the toolchain bug, not this option.

## Known limitations (v1)

- Audio: plays mono, or true interleaved stereo (front-left + front-right).
  A file with only surround/center/LFE roles (e.g. rendered with
  `--channels 5.1`) has no audio output — no downmix is implemented.
- No file-open dialog or drag-and-drop; the file path is a CLI argument.
- A/V sync: audio is the master clock, so the displayed video frame trails
  "what you're hearing" by roughly the audio device's buffer depth (tens of
  milliseconds) — a small constant latency, not drift.

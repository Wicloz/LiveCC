import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

import ccmf
import transcoder
from transcoder import _FrameSplitter, _kill_wait, _ytdlp_cmd


# --------------------------------------------------------------------------- #
# _FrameSplitter
# --------------------------------------------------------------------------- #

def _px_dims(term_w, term_h):
    return term_w * 2, term_h * 3   # 2x3 sub-pixels per cell


def _encode_one_gop(frame, fps=10):
    """One frame through the real GOP encoder -> a parsed CCMF video chunk."""
    from cc_encoder import GopEncoder
    gop = GopEncoder(nominal_duration=round(ccmf.SAMPLE_RATE / fps))
    gop.add(0, frame)
    _pts, chunk = gop.flush()
    pts, ctype, payload, _ = ccmf.parse_chunk(chunk)
    assert (pts, ctype) == (0, ccmf.TYPE_VIDEO)
    return ccmf.parse_video_payload(payload)


def test_splitter_emits_one_complete_frame():
    px_w, px_h = _px_dims(4, 2)
    s = _FrameSplitter(px_w, px_h)
    frames = list(s.push(bytes([17]) * (px_w * px_h * 3)))   # solid
    assert len(frames) == 1
    assert s.count == 1
    # push() yields the raw (H, W, 3) array; encoding is done separately.
    assert frames[0].shape == (px_h, px_w, 3)
    # The split frame encodes into a well-formed CCMF GOP for the 4x2 cell grid.
    w, h, decoded = _encode_one_gop(frames[0])
    assert (w, h) == (4, 2)
    assert decoded[0].encoding == ccmf.ENC_RAW


def test_splitter_buffers_partial_frames():
    px_w, px_h = _px_dims(2, 1)        # frame_bytes = 4*3*3 = 36
    s = _FrameSplitter(px_w, px_h)
    fb = px_w * px_h * 3
    assert list(s.push(bytes(fb - 1))) == []
    assert s.count == 0
    frames = list(s.push(bytes(1)))
    assert len(frames) == 1
    assert s.count == 1


def test_splitter_handles_multiple_frames_across_byte_chunks():
    px_w, px_h = _px_dims(2, 1)
    s = _FrameSplitter(px_w, px_h)
    two = bytes(px_w * px_h * 3 * 2)
    out = []
    for b in two:
        out.extend(s.push(bytes([b])))
    assert len(out) == 2
    assert s.count == 2


# --------------------------------------------------------------------------- #
# Command builders
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("enc_ms,fps,expected", [
    (5, 24, 1),       # fast encode (5 ms << 42 ms slot) -> every frame, full fps
    (40, 24, 2),      # 0.040*24*1.15 = 1.10 -> ceil 2 -> ~12 fps
    (70, 24, 2),      # 0.070*24*1.15 = 1.93 -> ceil 2 -> ~12 fps
    (200, 24, 6),     # 0.200*24*1.15 = 5.52 -> ceil 6 -> ~4 fps
    (5000, 24, 24),   # absurdly slow -> clamped to fps (1 fps floor)
    (0.1, 30, 1),     # trivial -> 1, never finer than every frame
])
def test_encode_stride_tracks_load(enc_ms, fps, expected):
    s = transcoder._encode_stride(enc_ms / 1000.0, fps)
    assert s == expected
    assert 1 <= s <= fps          # never finer than every frame, never below 1 fps


def test_ytdlp_cmd_streams_to_stdout():
    cmd = _ytdlp_cmd("https://example.com/v", transcoder._VIDEO_FMT)
    assert cmd[0] == "yt-dlp"
    assert "-o" in cmd and cmd[cmd.index("-o") + 1] == "-"
    assert cmd[-1] == "https://example.com/v"
    assert "--no-playlist" in cmd
    assert "--download-sections" not in cmd       # no offset by default


def test_ytdlp_cmd_section_start_only():
    cmd = _ytdlp_cmd("https://example.com/v", transcoder._VIDEO_FMT, start=125)
    assert cmd[cmd.index("--download-sections") + 1] == "*125-inf"
    assert cmd[-1] == "https://example.com/v"


def test_ytdlp_cmd_section_start_and_end():
    cmd = _ytdlp_cmd("https://example.com/v", transcoder._VIDEO_FMT, start=30, end=90)
    assert cmd[cmd.index("--download-sections") + 1] == "*30-90"


def test_ytdlp_cmd_section_subsecond():
    # ROOM start/end arrive in ms, so fractional seconds must survive the round
    # trip into the yt-dlp argument.
    cmd = _ytdlp_cmd("https://x", transcoder._VIDEO_FMT, start=90.5, end=120.25)
    assert cmd[cmd.index("--download-sections") + 1] == "*90.5-120.25"


def test_audio_fmt_prefers_best_source():
    # We resample to 8-bit PCM; feeding it the *worst* YouTube stream stacks a
    # second lossy pass on an already-crushed one (audible artifacts).  Pin
    # bestaudio so the resampler always gets a clean source.
    fmt = transcoder._AUDIO_FMT
    assert "bestaudio" in fmt
    assert "worstaudio" not in fmt
    # webm/opus still preferred first for streaming-pipe compatibility.
    assert fmt.startswith("bestaudio[ext=webm]")


def test_video_ffmpeg_cmd_reads_pipe_outputs_rgb24():
    cmd = transcoder._video_ffmpeg_cmd(10, 8, 12)
    assert "pipe:0" in cmd and "pipe:1" in cmd
    assert "rgb24" in cmd
    assert "fps=12" in " ".join(cmd)
    assert "-stream_loop" not in cmd


def test_video_ffmpeg_cmd_loops_a_file():
    cmd = transcoder._video_ffmpeg_cmd(10, 8, 12, source="/tmp/s.mkv", loop=True)
    assert cmd[cmd.index("-stream_loop") + 1] == "-1"
    assert "/tmp/s.mkv" in cmd
    assert "0:v:0" in cmd
    assert "pipe:0" not in cmd


def test_video_ffmpeg_cmd_decodes_file_without_loop():
    # A downloaded MP4 (moov-at-end) is decoded once from the seekable file —
    # source set but no -stream_loop unless --loop was requested.
    cmd = transcoder._video_ffmpeg_cmd(10, 8, 12, source="/tmp/s.mkv")
    assert "/tmp/s.mkv" in cmd
    assert "0:v:0" in cmd
    assert "pipe:0" not in cmd
    assert "-stream_loop" not in cmd


def test_video_ffmpeg_cmd_aligns_letterbox_to_cell_grid():
    # Regression: content edges must snap to the 2x3 cell grid, otherwise
    # boundary cells mix content with black bar and the corners go black.
    vf = " ".join(transcoder._video_ffmpeg_cmd(20, 8, 10))
    assert "trunc(iw/2)*2:trunc(ih/3)*3" in vf      # content snapped to grid
    assert "trunc((ow-iw)/4)*2" in vf               # even x pad offset
    assert "trunc((oh-ih)/6)*3" in vf               # multiple-of-3 y pad offset


def test_download_cmd_merges_to_mkv():
    # Regression: yt-dlp rejects "matroska"; the merge format must be "mkv".
    cmd = transcoder._download_cmd("https://x", "/tmp/d", 0, None, True)
    assert cmd[cmd.index("--merge-output-format") + 1] == "mkv"


def test_requirements_bundle_ejs_solver():
    # Regression: yt-dlp needs the bundled JS challenge solver (yt-dlp-ejs) to
    # solve YouTube's n/sig; the [default] extra pulls it in.
    req = (Path(__file__).parent.parent / "requirements.txt").read_text()
    assert "yt-dlp[default]" in req


def test_audio_ffmpeg_cmd_is_raw_pcm_mono():
    # ffmpeg only ever DECODES audio (to the speaker-native u8 PCM); the wire
    # codec (DFPWM when negotiated) is packed in Python per chunk (dfpwm.py),
    # so each chunk is independently decodable as the spec requires (§4.6).
    cmd = transcoder._audio_ffmpeg_cmd(48000)
    assert cmd[cmd.index("-c:a") + 1] == "pcm_u8"   # CC speaker's native format
    assert cmd[cmd.index("-f") + 1] == "u8"         # raw, no container
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-ar") + 1] == "48000"
    assert "dfpwm" not in cmd


def test_audio_codecs_share_chunk_duration():
    # Both codecs carry the same samples per chunk (~0.1 s); only the packing
    # differs (PCM 1 sample/byte, DFPWM 8 samples/byte — so the chunk sample
    # count must pack into whole DFPWM bytes).
    assert transcoder.PCM.samples_per_byte == 1
    assert transcoder.DFPWM.samples_per_byte == 8
    assert transcoder.AUDIO_CHUNK_SAMPLES % 8 == 0


def test_audio_ffmpeg_cmd_loops_a_file():
    cmd = transcoder._audio_ffmpeg_cmd(48000, source="/tmp/s.mkv", loop=True)
    assert cmd[cmd.index("-stream_loop") + 1] == "-1"
    assert "0:a:0?" in cmd            # optional audio map (source may be video-only)


def test_audio_ffmpeg_cmd_decodes_file_without_loop():
    cmd = transcoder._audio_ffmpeg_cmd(48000, source="/tmp/s.mkv")
    assert "0:a:0?" in cmd
    assert "-stream_loop" not in cmd


def test_needs_seekable_source_flags_mp4_family_only():
    # The whole ISOBMFF / MP4 family may hide moov at the end -> can't pipe-stream.
    for ext in ("mp4", "m4v", "m4a", "m4b", "mov", "qt",
                "3gp", "3g2", "f4v", "mj2", "mjp2", "MP4"):
        assert transcoder.needs_seekable_source(ext)
    for ext in ("webm", "mkv", "", "ts", "avi", "flv", "ogg"):
        assert not transcoder.needs_seekable_source(ext)


def test_needs_download_flags_gif():
    # The server's ffmpeg can't demux a GIF from the pipe -> must download first.
    assert transcoder.needs_download("gif")
    assert transcoder.needs_download("GIF")
    for ext in ("mp4", "webm", "mkv", "mov", ""):
        assert not transcoder.needs_download(ext)


def _box(btype, payload=b""):
    return (8 + len(payload)).to_bytes(4, "big") + btype + payload


def test_scan_moov_position():
    ftyp = _box(b"ftyp", b"isom\x00\x00\x02\x00")
    # moov before mdat -> faststart -> False
    assert transcoder._scan_moov_position(ftyp + _box(b"moov", b"\x00" * 32)) is False
    # mdat before moov -> moov-at-end -> True (decided from the mdat header alone)
    assert transcoder._scan_moov_position(ftyp + _box(b"mdat", b"\x00" * 9999)) is True
    # leading boxes are skipped by size before the decisive one
    assert transcoder._scan_moov_position(
        ftyp + _box(b"free", b"\x00" * 16) + _box(b"moov")) is False
    # not enough yet to reach the next box header -> undetermined
    assert transcoder._scan_moov_position(ftyp) is None
    assert transcoder._scan_moov_position(b"\x00\x00\x00") is None


def test_audio_chunk_is_short_to_avoid_periodic_video_stall():
    # The CC client decodes each audio chunk inline on the coroutine that also
    # renders video.  A ~1 s chunk blocked rendering for the whole decode once per
    # second (visible stutter), so chunks must stay short enough to interleave.
    assert transcoder.AUDIO_CHUNK_SECONDS <= 0.2


def test_audio_ffmpeg_cmd_has_no_server_side_filtering():
    # Server-side audio *processing* was reported to make audio worse, so the
    # -af chain may only inspect (ashowinfo: first-frame source PTS for
    # SourceTimeline) — never touch the signal.  Channel work is -ac plus
    # Python-side cutting (role_source_channel / mono_downmix), not filters.
    for cmd in (transcoder._audio_ffmpeg_cmd(48000),
                transcoder._audio_ffmpeg_cmd(48000, source="/tmp/s.mkv"),
                transcoder._multichannel_pcm_cmd(48000, 2)):
        assert "-filter:a" not in cmd
        af = cmd[cmd.index("-af") + 1] if "-af" in cmd else ""
        for part in filter(None, af.split(",")):
            assert part == "ashowinfo", part


@pytest.mark.parametrize("role,source_channels,expected", [
    (1, 2, 0), (2, 2, 1),           # fronts: discrete whenever there are two
    (1, 6, 0), (2, 8, 1),
    (3, 2, None), (3, 6, 2),        # center: phantom (mono) on stereo
    (4, 2, None), (4, 6, 3),        # lfe: mono on sources without one
    (5, 2, 0), (6, 2, 1),           # surrounds mirror the fronts on stereo
    (5, 6, 4), (6, 6, 5),           # …and are discrete on 5.1
    (7, 2, 0), (8, 2, 1),           # rears mirror the fronts on stereo
    (7, 6, 4), (8, 6, 5),           # …the surrounds on 5.1
    (7, 8, 6), (8, 8, 7),           # …and are discrete on 7.1
    (1, 1, 0),                      # 1-channel decode: channel 0 IS the mono
])
def test_role_source_channel_fallback_chains(role, source_channels, expected):
    # The channel-mismatch table: every role resolves to the best available
    # source channel, or to the mono downmix (None) — never to nothing.  This
    # is what lets any speaker layout play any source layout.
    assert transcoder.role_source_channel(role, source_channels) == expected


def test_video_ffmpeg_cmd_taps_first_frame_pts():
    # showinfo sits LAST in the chain (after fps), so its first report is the
    # source timestamp of output frame 0 exactly — the SourceTimeline base.
    cmd = transcoder._video_ffmpeg_cmd(10, 8, 12)
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.endswith("fps=12,showinfo")


def test_first_pts_probe_captures_only_the_first_frame():
    p = transcoder._FirstPtsProbe()
    assert not p.feed("[vp9 @ 0x1] some decoder line")       # not showinfo: logged
    assert p.feed("[Parsed_showinfo_0 @ 0x55e] n:   0 pts:  12345 pts_time:257.32 "
                  "duration:512")
    assert p.seen and p.value == pytest.approx(257.32)
    # later frames must not overwrite: the base maps output 0, nothing else
    assert p.feed("[Parsed_showinfo_0 @ 0x55e] n:   1 pts: 12857 pts_time:258.0")
    assert p.value == pytest.approx(257.32)


def test_first_pts_probe_nopts_first_frame_stays_unknown():
    p = transcoder._FirstPtsProbe()
    assert p.feed("[Parsed_ashowinfo_0 @ 0x1] n:0 pts:NOPTS pts_time:NOPTS")
    assert p.seen and p.value is None
    p.feed("[Parsed_ashowinfo_0 @ 0x1] n:1 pts:1024 pts_time:0.021")
    assert p.value is None   # a later frame's pts would map the wrong position


# --------------------------------------------------------------------------- #
# SourceTimeline
# --------------------------------------------------------------------------- #

def test_source_timeline_aligns_two_pipelines():
    # The live A/V case: video's fetch joined the stream 2.5 s after audio's.
    # Audio (earliest base) becomes the zero; video is shifted +2.5 s.
    async def go():
        tl = transcoder.SourceTimeline(["video", "audio"], timeout=5)
        audio, video = await asyncio.gather(
            tl.offset_samples("audio", 97.0),
            tl.offset_samples("video", 99.5))
        assert audio == 0
        assert video == round(2.5 * transcoder.SAMPLE_RATE)

    asyncio.run(go())


def test_source_timeline_falls_back_when_any_base_unknown():
    # Half-corrections are worse than none: one unknown base disables both.
    async def go():
        tl = transcoder.SourceTimeline(["video", "audio"], timeout=5)
        audio, video = await asyncio.gather(
            tl.offset_samples("audio", None),
            tl.offset_samples("video", 99.5))
        assert (audio, video) == (0, 0)

    asyncio.run(go())


def test_source_timeline_falls_back_on_implausible_skew():
    # Bases further apart than a plausible live-fetch skew are two different
    # timestamp epochs (e.g. an HLS/DASH mix), not an offset to correct.
    async def go():
        tl = transcoder.SourceTimeline(["video", "audio"], timeout=5)
        audio, video = await asyncio.gather(
            tl.offset_samples("audio", 12.0),
            tl.offset_samples("video", 5000.0))
        assert (audio, video) == (0, 0)

    asyncio.run(go())


def test_source_timeline_single_pipeline_needs_no_offset():
    async def go():
        tl = transcoder.SourceTimeline(["audio"], timeout=5)
        assert await tl.offset_samples("audio", 12345.6) == 0

    asyncio.run(go())


def test_source_timeline_timeout_releases_the_waiter():
    # A pipeline that never produces (broken video) must not hold audio
    # hostage: the waiter falls back to raw counters after the timeout.
    async def go():
        tl = transcoder.SourceTimeline(["video", "audio"], timeout=0.05)
        assert await asyncio.wait_for(tl.offset_samples("audio", 97.0), 2) == 0
        # the late reporter gets the same (already-frozen) decision
        assert await tl.offset_samples("video", 99.5) == 0

    asyncio.run(go())


# --------------------------------------------------------------------------- #
# mono downmix
# --------------------------------------------------------------------------- #

def test_mono_downmix_averages_channels():
    import numpy as np
    stereo = np.array([[10, 30], [20, 40]], np.uint8)
    assert list(transcoder.mono_downmix(stereo)) == [20, 30]


def test_mono_downmix_excludes_lfe():
    import numpy as np
    # 5.1 decode order FL FR FC LFE BL BR: index 3 (value 250) must not fold in.
    six = np.tile(np.array([[10, 20, 30, 250, 40, 50]], np.uint8), (3, 1))
    assert list(transcoder.mono_downmix(six)) == [30, 30, 30]


@pytest.mark.parametrize("stdout,expected", [
    ("2\n", 2),           # normal YouTube VOD
    ("6\n", 6),           # 5.1 source
    ("1\n", 1),           # honestly mono: trusted
    ("NA\n", None),       # generic extractor (direct URL) / most live streams
    ("", None),           # no output at all (extraction failed)
    ("none\n", None),     # junk
    ("0\n", None),        # zero channels is not a real answer
    ("-1\n", None),
])
def test_parse_audio_channels_trusts_only_positive_integers(stdout, expected):
    # yt-dlp reports audio_channels as "NA" for every direct-URL source and
    # for most live streams.  The parser must say "unknown", not guess — the
    # layered probe then asks ffprobe, and only assumes as a last resort.
    assert transcoder.parse_audio_channels(stdout) == expected


def test_probe_layers_metadata_then_ffprobe_then_assumption(monkeypatch):
    # The channel probe is layered: yt-dlp format metadata first, ffprobe on
    # the URL itself second (exactly the direct-URL class whose metadata is
    # "NA"), and only then the stereo assumption.  A wrong guess no longer
    # silences roles (fallback mixes) — but the direction still matters:
    # assuming stereo of a mono source is dual-mono; assuming mono of a
    # stereo source throws real channels away.
    class _NoMetadata:
        stdout = "NA\n"
        returncode = 0

    monkeypatch.setattr(transcoder.subprocess, "run",
                        lambda *a, **k: _NoMetadata())
    monkeypatch.setattr(transcoder, "_ffprobe_channels", lambda url: 6)
    assert transcoder._probe_audio_channels_blocking("u") == 6

    monkeypatch.setattr(transcoder, "_ffprobe_channels", lambda url: None)
    assert (transcoder._probe_audio_channels_blocking("u")
            == transcoder._ASSUMED_CHANNELS)
    assert transcoder._ASSUMED_CHANNELS == 2   # never mono; see the constant


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="ffmpeg/ffprobe not installed")
def test_ffprobe_channels_reads_direct_media(tmp_path):
    path = tmp_path / "stereo.wav"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", "aevalsrc=0.25|0.75:c=stereo:s=48000:d=0.2",
           str(path)]
    if subprocess.run(cmd, stdout=subprocess.DEVNULL,
                      stderr=subprocess.PIPE).returncode != 0:
        pytest.skip("ffmpeg can't build a stereo source")
    assert transcoder._ffprobe_channels(str(path)) == 2
    # Wider layouts report their real width too (drives the 5.1 decode).
    six = tmp_path / "surround.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "aevalsrc=0.1|0.2|0.3|0.4|0.5|0.6:c=5.1:s=48000:d=0.2",
                    str(six)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if six.exists():
        assert transcoder._ffprobe_channels(str(six)) == 6
    # Non-media (an extractor page, a dead link): unknown, not an exception.
    assert transcoder._ffprobe_channels(str(tmp_path / "missing.wav")) is None


def test_source_timeline_live_falls_back_to_wall_alignment():
    # LIVE + unalignable timestamps: raw counters would pin each pipeline's
    # zero to its own first output — with drop-oldest buffers that STARVES the
    # earlier stream (see test_session.test_live_audio_flows_despite_
    # unaligned_skewed_pipelines for the end-to-end proof).  The fallback must
    # instead approximate the skew with first-output wall times.
    async def go():
        tl = transcoder.SourceTimeline(["video", "audio"], timeout=5, live=True)

        async def audio():
            return await tl.offset_samples("audio", None)

        async def video():
            await asyncio.sleep(0.3)               # video decode starts later
            return await tl.offset_samples("video", None)

        audio_off, video_off = await asyncio.gather(audio(), video())
        assert audio_off == 0                       # earliest output = zero
        # video is placed ~0.3 s later on the timeline (generous CI margins)
        assert 0.2 * transcoder.SAMPLE_RATE <= video_off <= 0.8 * transcoder.SAMPLE_RATE

    asyncio.run(go())


def test_source_timeline_live_prefers_real_timestamps_over_wall():
    # The wall fallback must NOT kick in when source timestamps align fine.
    async def go():
        tl = transcoder.SourceTimeline(["video", "audio"], timeout=5, live=True)

        async def audio():
            return await tl.offset_samples("audio", 100.0)

        async def video():
            await asyncio.sleep(0.2)               # wall skew that must be ignored
            return await tl.offset_samples("video", 102.5)

        audio_off, video_off = await asyncio.gather(audio(), video())
        assert audio_off == 0
        assert video_off == round(2.5 * transcoder.SAMPLE_RATE)   # from bases

    asyncio.run(go())


def test_source_timeline_vod_fallback_stays_raw_counters():
    # VOD pipelines decode the same file from zero, faster than realtime —
    # wall times would be wrong there; unknown bases must mean offsets 0.
    async def go():
        tl = transcoder.SourceTimeline(["video", "audio"], timeout=5, live=False)

        async def audio():
            return await tl.offset_samples("audio", None)

        async def video():
            await asyncio.sleep(0.2)
            return await tl.offset_samples("video", None)

        assert await asyncio.gather(audio(), video()) == [0, 0]

    asyncio.run(go())


def test_negotiate_channel_roles_serves_every_requested_role():
    # Negotiation is purely "what did the client(s) ask for" — never "what
    # does the source have".  Any role can be produced from any source via
    # the fallback table, so filtering here could only silence a mapped
    # speaker (the old behaviour: a stereo rig on a mono-probed source heard
    # NOTHING; a 7.1 rig on a stereo source had six dead speakers).
    caps = ccmf.CAP_CHANNEL_MONO | ccmf.CAP_CHANNEL_FRONT_LEFT | ccmf.CAP_CHANNEL_FRONT_RIGHT
    assert transcoder.negotiate_channel_roles(caps) == [0, 1, 2]
    everything = sum(1 << r for r in range(9))
    assert transcoder.negotiate_channel_roles(everything) == list(range(9))


def test_negotiate_channel_roles_has_no_bundling_requirement():
    # Only front_left requested (not front_right) -> that role alone is
    # served, not bundled up to a full stereo pair or dropped to mono. This
    # matters for a sync room's union: one client's mono+lfe combined with
    # another's front_left+front_right must yield exactly those four roles,
    # not get rounded up/down to a canonical stereo/5.1/7.1 group.
    assert transcoder.negotiate_channel_roles(ccmf.CAP_CHANNEL_FRONT_LEFT) == [1]
    union = (ccmf.CAP_CHANNEL_MONO | ccmf.CAP_CHANNEL_LFE
             | ccmf.CAP_CHANNEL_FRONT_LEFT | ccmf.CAP_CHANNEL_FRONT_RIGHT)
    assert transcoder.negotiate_channel_roles(union) == [0, 1, 2, 4]


def test_negotiate_channel_roles_empty_request_falls_back_to_mono():
    assert transcoder.negotiate_channel_roles(0) == [0]
    # reserved bits above the defined roles are ignored, not served
    assert transcoder.negotiate_channel_roles(1 << 12) == [0]


# --------------------------------------------------------------------------- #
# _kill_wait
# --------------------------------------------------------------------------- #

def test_kill_wait_is_safe_on_finished_process_and_none():
    p = subprocess.Popen(
        ["python", "-c", "pass"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    p.wait()
    _kill_wait(p, None)   # must tolerate both a dead process and None


# --------------------------------------------------------------------------- #
# Integration: real ffmpeg pipeline (skipped if ffmpeg is unavailable)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_rawvideo_feeds_splitter():
    term_w, term_h, fps = 8, 4, 5
    px_w, px_h = _px_dims(term_w, term_h)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=size=64x48:rate={fps}:duration=1",
        "-vf", f"scale={px_w}:{px_h}",
        "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.returncode == 0
    assert len(proc.stdout) == px_w * px_h * 3 * fps

    s = _FrameSplitter(px_w, px_h)
    raw_frames = list(s.push(proc.stdout))
    assert len(raw_frames) == fps
    w, h, decoded = _encode_one_gop(raw_frames[0], fps)
    assert (w, h) == (term_w, term_h)
    assert decoded[0].encoding == ccmf.ENC_RAW


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_raw_pcm_output_is_one_byte_per_sample():
    # Push a 1 s sine through the audio command's output args and confirm raw
    # u8 PCM: exactly 48000 mono bytes (1 byte/sample), the client's unpack input.
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=48000",
        "-ar", "48000", "-ac", "1", "-c:a", "pcm_u8", "-f", "u8", "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert len(proc.stdout) == 48000   # 1 s * 48 kHz * 1 byte/sample


# --------------------------------------------------------------------------- #
# Integration: decode real container files generated by ffmpeg on the fly.
# Exercises the seekable-file path (iter_video/iter_audio with source_path) that
# plain MP4s (moov-at-end) are routed through, across several containers.
# --------------------------------------------------------------------------- #

def _ffmpeg_make(path, vcodec, *, acodec=None, extra=()):
    """Generate a 1 s test clip in whatever container `path`'s suffix implies."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=1"]
    if acodec:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=48000"]
    cmd += ["-c:v", vcodec]
    if acodec:
        cmd += ["-c:a", acodec]
    cmd += list(extra) + ["-shortest", str(path)]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _collect_video_from_file(path, limit=None):
    chunks = []

    async def go():
        agen = transcoder.iter_video("ignored", term_w=8, term_h=4, fps=5,
                                     source_path=str(path))
        try:
            async for pts, chunk in agen:     # iter_video yields (pts, GOP chunk)
                chunks.append((pts, chunk))
                if limit and len(chunks) >= limit:
                    break
        finally:
            await agen.aclose()

    asyncio.run(go())
    return chunks


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_adaptive_pacing_skips_frames_when_encode_is_slow(tmp_path, monkeypatch):
    # When the encoder can't keep up, iter_video must encode only a subset of
    # source frames (so the stream doesn't out-run the encoder) while the frame
    # PTS/durations it emits stay on the true source grid (i/fps) so A/V sync is
    # preserved.
    import time

    import cc_encoder

    path = tmp_path / "clip.mkv"
    if _ffmpeg_make(path, "mpeg4").returncode != 0:     # ~10 frames at 10 fps
        pytest.skip("ffmpeg can't build the clip")

    class SlowGop(cc_encoder.GopEncoder):
        def add(self, pts, frame):
            time.sleep(0.15)     # 0.15*10*1.15 = 1.7 -> stride 2 -> ~half the frames
            return super().add(pts, frame)

    monkeypatch.setattr(transcoder, "GopEncoder", SlowGop)

    frames = []

    async def go():
        agen = transcoder.iter_video("ignored", term_w=8, term_h=4, fps=10,
                                     source_path=str(path))
        try:
            async for pts, chunk in agen:
                cpts, _t, payload, _ = ccmf.parse_chunk(chunk)
                assert cpts == pts
                cur = cpts
                for f in ccmf.parse_video_payload(payload)[2]:
                    frames.append((cur, f.duration))
                    cur += f.duration
        finally:
            await agen.aclose()

    asyncio.run(go())

    assert len(frames) >= 2, "should still emit some frames"
    # Frames were skipped: fewer emitted than the ~10 source frames.
    assert len(frames) <= 7
    grid = ccmf.SAMPLE_RATE // 10                     # 4800 samples per source frame
    # Frame PTS stay on the source grid and at least one hold spans >1 source
    # frame (a skip actually happened).
    assert all(pts % grid == 0 for pts, _ in frames)
    assert any(d >= 2 * grid for _, d in frames)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.parametrize("fname,vcodec,extra", [
    ("moov_end.mp4", "mpeg4", ()),                       # default mp4: moov at END
    ("faststart.mp4", "mpeg4", ("-movflags", "+faststart")),
    ("clip.mkv", "mpeg4", ()),
    ("clip.mov", "mpeg4", ()),
    ("clip.webm", "libvpx", ()),
])
def test_decode_video_from_generated_file(tmp_path, fname, vcodec, extra):
    path = tmp_path / fname
    res = _ffmpeg_make(path, vcodec, extra=extra)
    if res.returncode != 0 or not path.exists():
        pytest.skip(f"ffmpeg can't build {fname} ({vcodec}): "
                    f"{res.stderr.decode('utf-8', 'replace')[:200]}")
    chunks = _collect_video_from_file(path)
    assert len(chunks) > 0
    # Every chunk is a well-formed, self-contained GOP for the 8x4 grid.
    for _pts, chunk in chunks:
        _cpts, ctype, payload, _ = ccmf.parse_chunk(chunk)
        assert ctype == ccmf.TYPE_VIDEO
        w, h, frames = ccmf.parse_video_payload(payload)
        assert (w, h) == (8, 4)
        assert frames[0].encoding == ccmf.ENC_RAW        # opens with a keyframe


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_moov_at_end_mp4_decodes_from_seekable_file(tmp_path):
    # The core of the feature: a moov-at-end MP4 (unstreamable from a pipe) decodes
    # fine once it's a seekable local file — which is why such VODs are downloaded.
    path = tmp_path / "trailing_moov.mp4"
    res = _ffmpeg_make(path, "mpeg4")              # default muxing -> moov after mdat
    if res.returncode != 0:
        pytest.skip("ffmpeg can't build mp4")
    data = path.read_bytes()
    mdat, moov = data.find(b"mdat"), data.find(b"moov")
    assert 0 < mdat < moov, "expected a genuine moov-at-end file for this test"
    assert len(_collect_video_from_file(path)) > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_scan_moov_position_on_real_files(tmp_path):
    # The box scanner must classify ffmpeg's real output: default muxing puts moov
    # at the end (download), +faststart puts it up front (stream).
    end = tmp_path / "end.mp4"
    fast = tmp_path / "fast.mp4"
    if _ffmpeg_make(end, "mpeg4").returncode != 0 or \
       _ffmpeg_make(fast, "mpeg4", extra=("-movflags", "+faststart")).returncode != 0:
        pytest.skip("ffmpeg can't build mp4")
    assert transcoder._scan_moov_position(end.read_bytes()) is True     # moov at end
    assert transcoder._scan_moov_position(fast.read_bytes()) is False   # faststart


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_decode_audio_from_generated_file(tmp_path):
    path = tmp_path / "with_audio.mp4"
    res = _ffmpeg_make(path, "mpeg4", acodec="aac")
    if res.returncode != 0:
        pytest.skip("ffmpeg can't build mp4+aac")

    chunks = []

    async def go():
        agen = transcoder.iter_audio("ignored", sample_rate=48000, source_path=str(path))
        try:
            async for c in agen:
                chunks.append(c)
                if chunks:
                    break
        finally:
            await agen.aclose()

    asyncio.run(go())
    assert sum(len(c) for c in chunks) > 0         # produced raw PCM bytes


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_iter_audio_roles_deinterleaves_stereo_without_crosstalk(tmp_path):
    # Regression for the channel-drift bug class: independent per-role (or
    # separate mono/positional) pipelines each fetch+decode the source
    # separately, which is fine for a seekable file but skews on a live source
    # (two fetches of "now" don't land on the same sample). iter_audio_roles
    # decodes once and cuts every role — including the mono downmix — from the
    # same frames. Verify with a distinct constant level per channel
    # (front_left ~0.25, front_right ~0.75): any swap or off-by-one in the
    # de-interleave would show up as a wrong level, and mono must sit between.
    path = tmp_path / "stereo.wav"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", "aevalsrc=0.25|0.75:c=stereo:s=48000:d=0.5", str(path)]
    if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE).returncode != 0:
        pytest.skip("ffmpeg can't build a synthetic stereo source")

    roles = [ccmf.CHANNEL_MONO, ccmf.CHANNEL_FRONT_LEFT, ccmf.CHANNEL_FRONT_RIGHT,
             ccmf.CHANNEL_CENTER, ccmf.CHANNEL_SURROUND_LEFT,
             ccmf.CHANNEL_REAR_RIGHT]
    ptss = []
    got = {r: [] for r in roles}

    async def go():
        agen = transcoder.iter_audio_roles(
            "ignored", sample_rate=48000, roles=roles,
            decode_channels=2, source_path=str(path))
        try:
            async for pts, chunks in agen:
                ptss.append((pts, len(chunks[ccmf.CHANNEL_FRONT_LEFT])))
                for r in roles:
                    got[r].append(chunks[r])
        finally:
            await agen.aclose()

    asyncio.run(go())
    joined = {r: b"".join(c) for r, c in got.items()}
    assert all(joined.values())
    assert len({len(v) for v in joined.values()}) == 1   # sample-aligned per role
    avg = {r: sum(v) / len(v) for r, v in joined.items()}
    left, right = avg[ccmf.CHANNEL_FRONT_LEFT], avg[ccmf.CHANNEL_FRONT_RIGHT]
    assert left < 180 < right                    # 0.25 -> ~160, 0.75 -> ~224: not swapped
    assert left < avg[ccmf.CHANNEL_MONO] < right  # mono = downmix of the same frames
    # Mismatch resolution on a stereo source: surrounds/rears mirror the
    # fronts, center gets the (phantom) mono — byte-identical, not silent.
    assert joined[ccmf.CHANNEL_SURROUND_LEFT] == joined[ccmf.CHANNEL_FRONT_LEFT]
    assert joined[ccmf.CHANNEL_REAR_RIGHT] == joined[ccmf.CHANNEL_FRONT_RIGHT]
    assert joined[ccmf.CHANNEL_CENTER] == joined[ccmf.CHANNEL_MONO]
    # PTS is contiguous: chunk N+1 starts exactly where chunk N ended.
    for (p1, n1), (p2, _n2) in zip(ptss, ptss[1:]):
        assert p2 == p1 + n1


# The user-visible mismatch matrix: SOURCE layout (rows below) x a full 7.1
# speaker layout requesting every role.  Levels are distinct per source
# channel, so each role's mean identifies exactly which mix it carries; the
# identity groups pin the aliasing (mirrored roles share the same bytes).
# u8 level for amplitude a is 128 + a*128.
_LAYOUT_MATRIX = [
    # mono source: every role of every rig plays THE mono.
    ("mono", "aevalsrc=0.5:c=mono:s=48000:d=0.4", 1,
     {r: 192.0 for r in range(9)},
     [set(range(9))]),
    # stereo source: fronts discrete, surrounds/rears mirror them,
    # center/LFE/mono are the downmix.
    ("stereo", "aevalsrc=0.25|0.75:c=stereo:s=48000:d=0.4", 2,
     {0: 192.0, 1: 160.0, 2: 224.0, 3: 192.0, 4: 192.0,
      5: 160.0, 6: 224.0, 7: 160.0, 8: 224.0},
     [{1, 5, 7}, {2, 6, 8}, {0, 3, 4}]),
    # 5.1 source: everything discrete except the rears, which mirror the
    # surrounds; mono excludes the (loud, 0.9) LFE — its level proves it.
    ("5.1", "aevalsrc=0.1|0.2|0.3|0.9|0.5|0.6:c=5.1:s=48000:d=0.4", 6,
     {0: 171.5, 1: 140.8, 2: 153.6, 3: 166.4, 4: 243.2,
      5: 192.0, 6: 204.8, 7: 192.0, 8: 204.8},
     [{5, 7}, {6, 8}]),
]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.parametrize("name,src,channels,levels,identical",
                         _LAYOUT_MATRIX, ids=[m[0] for m in _LAYOUT_MATRIX])
def test_layout_matrix_every_role_from_every_source(tmp_path, name, src,
                                                    channels, levels, identical):
    path = tmp_path / f"{name.replace('.', '_')}.wav"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", src, str(path)]
    if subprocess.run(cmd, stdout=subprocess.DEVNULL,
                      stderr=subprocess.PIPE).returncode != 0:
        pytest.skip(f"ffmpeg can't build the {name} source")

    got = {r: [] for r in range(9)}

    async def go():
        agen = transcoder.iter_audio_roles(
            "ignored", sample_rate=48000, roles=list(range(9)),
            decode_channels=channels, source_path=str(path))
        try:
            async for _pts, chunks in agen:
                for r in range(9):
                    got[r].append(chunks[r])
        finally:
            await agen.aclose()

    asyncio.run(go())
    joined = {r: b"".join(c) for r, c in got.items()}
    assert all(len(v) > 9600 for v in joined.values())   # every role has audio
    for role, want in levels.items():
        mean = sum(joined[role]) / len(joined[role])
        assert mean == pytest.approx(want, abs=3), \
            f"{name}: role {role} carries the wrong mix ({mean:.1f} != {want})"
    for group in identical:
        first = joined[min(group)]
        assert all(joined[r] == first for r in group), \
            f"{name}: roles {sorted(group)} should alias the same bytes"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_iter_audio_roles_mono_source_aliases_every_role(tmp_path):
    # decode_channels == 1 (mono or mono-treated source) needs no
    # de-interleave: the plain downmix pipeline feeds EVERY requested role the
    # same samples — a stereo/7.1 rig on a mono source plays the mono
    # everywhere instead of going silent.
    path = tmp_path / "with_audio.mp4"
    if _ffmpeg_make(path, "mpeg4", acodec="aac").returncode != 0:
        pytest.skip("ffmpeg can't build mp4+aac")

    roles = [ccmf.CHANNEL_MONO, ccmf.CHANNEL_FRONT_LEFT, ccmf.CHANNEL_FRONT_RIGHT]
    got = []

    async def go():
        agen = transcoder.iter_audio_roles(
            "ignored", sample_rate=48000, roles=roles,
            decode_channels=1, source_path=str(path))
        try:
            async for pts, chunks in agen:
                got.append((pts, chunks))
                if len(got) >= 2:
                    break
        finally:
            await agen.aclose()

    asyncio.run(go())
    assert got and set(got[0][1]) == set(roles)
    assert got[0][0] == 0                                  # counts from zero
    first = got[0][1]
    assert first[ccmf.CHANNEL_FRONT_LEFT] == first[ccmf.CHANNEL_MONO]
    assert first[ccmf.CHANNEL_FRONT_RIGHT] == first[ccmf.CHANNEL_MONO]
    assert got[1][0] == len(first[ccmf.CHANNEL_MONO])      # contiguous PTS


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_source_timeline_resolves_from_real_pipelines(tmp_path):
    # End-to-end alignment plumbing: both pipelines must parse their first
    # source PTS out of real ffmpeg showinfo/ashowinfo stderr and resolve the
    # shared timeline from it — not via the None/timeout fallback.  Catches a
    # showinfo tap missing from a command, the drain not feeding the probe,
    # or the pts_time regex drifting from ffmpeg's actual output.
    path = tmp_path / "clip.mkv"
    if _ffmpeg_make(path, "mpeg4", acodec="aac").returncode != 0:
        pytest.skip("ffmpeg can't build the clip")

    async def go():
        tl = transcoder.SourceTimeline(["video", "audio"], timeout=10)
        video_pts, audio_pts = [], []

        async def video():
            agen = transcoder.iter_video("ignored", term_w=8, term_h=4, fps=5,
                                         source_path=str(path), timeline=tl)
            try:
                async for pts, _chunk in agen:
                    video_pts.append(pts)
            finally:
                await agen.aclose()

        async def audio():
            agen = transcoder.iter_audio_roles("ignored", 48000, roles=[0],
                                               decode_channels=1,
                                               source_path=str(path), timeline=tl)
            try:
                async for pts, _chunks in agen:
                    audio_pts.append(pts)
            finally:
                await agen.aclose()

        await asyncio.wait_for(asyncio.gather(video(), audio()), 30)
        assert video_pts and audio_pts
        # Both probes parsed a real base (no fallback): the timeline decided
        # on an actual zero point.
        assert tl._zero is not None
        # Same file, same start: both streams begin within a frame of zero.
        assert 0 <= video_pts[0] <= ccmf.SAMPLE_RATE // 5
        assert 0 <= audio_pts[0] <= transcoder.AUDIO_CHUNK_SAMPLES

    asyncio.run(go())


# --------------------------------------------------------------------------- #
# Integration: real live streams (regression guard)
# --------------------------------------------------------------------------- #

# NASA's 24/7 live channels.  These exercise the full live pipeline end to end.
NASA_LIVE_STREAMS = [
    "https://www.youtube.com/watch?v=FuuC4dpSQ1M",
    "https://www.youtube.com/watch?v=uwXgcTc8oY8",
]

# YouTube needs a JS runtime (deno/node) to resolve formats; without one the
# pipeline can't fetch anything, so skip rather than report a bogus failure.
_HAS_JS_RUNTIME = any(shutil.which(x) for x in ("deno", "node", "bun"))


@pytest.mark.youtube
@pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("yt-dlp") and _HAS_JS_RUNTIME),
    reason="live test needs ffmpeg, yt-dlp and a JS runtime (deno/node) on PATH",
)
@pytest.mark.parametrize("url", NASA_LIVE_STREAMS)
def test_live_stream_produces_frames(url):
    """A real live stream must produce chunks through the video pipeline.

    Confirms the stream is actually live first; if it isn't (taken down, or no
    network right now) the test skips, so it only fails on a genuine regression.
    """
    state = {"chunks": 0}

    async def go():
        if not await transcoder.probe_is_live(url):
            pytest.skip(f"not live right now (or unreachable): {url}")

        async def collect():
            agen = transcoder.iter_video(url, term_w=20, term_h=8, fps=5)
            try:
                async for _pts, _chunk in agen:   # iter_video yields (pts, chunk)
                    state["chunks"] += 1
                    if state["chunks"] >= 2:
                        break
            finally:
                await agen.aclose()

        try:
            await asyncio.wait_for(collect(), timeout=60)
        except asyncio.TimeoutError:
            pass

    asyncio.run(go())
    assert state["chunks"] > 0, f"live stream produced no chunks: {url}"

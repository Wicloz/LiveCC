import numpy as np
import pytest

import ccmf
from cc_encoder import _CC_RGB, GopEncoder, encode_frame, generate_palette

# palette indices used in the assertions
_WHITE, _LGRAY, _BLUE, _RED = 0, 8, 11, 14


def _solid(rgb, ph, pw):
    return np.tile(np.array(rgb, dtype=np.uint8), (ph, pw, 1))


def _cell(pixels):
    # pixels is a 3-row x 2-col list of RGB triples -> (3, 2, 3) uint8.
    return np.array(pixels, dtype=np.uint8).reshape(3, 2, 3)


def _enc(frame, adaptive=False):
    """encode_frame -> (glyph, fg, bg, palette) at the frame's grid.

    Defaults to the FIXED CC palette so the palette-index assertions below have a
    known meaning; the adaptive (default) path is covered by its own tests."""
    return encode_frame(frame, adaptive=adaptive)


def test_grids_match_frame_dimensions():
    W, H = 4, 2
    glyph, fg, bg, pal = encode_frame(_solid((17, 17, 17), H * 3, W * 2))
    assert glyph.shape == fg.shape == bg.shape == (H, W)
    assert pal.shape == (16, 3) and pal.dtype == np.uint8


def test_solid_cell_is_empty_glyph():
    # A solid cell needs no sub-pixel structure: empty glyph (0x80) with the colour
    # carried in the background.
    glyph, _fg, bg, _pal = _enc(_solid((204, 76, 76), 3, 2))   # solid red
    assert glyph[0, 0] == 0x80                  # empty drawing glyph
    assert bg[0, 0] == _RED


def test_glyphs_and_colours_in_range():
    rng = np.random.default_rng(0)
    W, H = 6, 5
    glyph, fg, bg, pal = _enc(rng.integers(0, 256, size=(H * 3, W * 2, 3), dtype=np.uint8))
    assert glyph.shape == (H, W)
    assert ((glyph >= 0x80) & (glyph <= 0x9F)).all()   # valid 2x3 drawing chars
    assert ((fg <= 15).all() and (bg <= 15).all())     # valid palette indices
    assert pal.shape == (16, 3)


def test_pair_search_brackets_a_split_cell():
    # A cell that is red over the top two rows and blue across the bottom row must
    # resolve to the {red, blue} pair (not collapse both to one "average" colour).
    glyph, fg, bg, _pal = _enc(_cell([
        [(204, 76, 76), (204, 76, 76)],   # red
        [(204, 76, 76), (204, 76, 76)],   # red
        [(51, 102, 204), (51, 102, 204)],  # blue
    ]))
    assert 0x80 <= glyph[0, 0] <= 0x9F
    assert {int(fg[0, 0]), int(bg[0, 0])} == {_RED, _BLUE}


def test_between_colors_region_is_dithered():
    # A flat field whose colour sits between white and light_gray can't be matched
    # by a single palette entry, so the encoder dithers between the two — both
    # colours appear across the region rather than everything snapping to one.
    W, H = 8, 8
    _glyph, fg, bg, _pal = _enc(_solid((196, 196, 196), H * 3, W * 2))
    used = set(fg.ravel().tolist()) | set(bg.ravel().tolist())
    assert _WHITE in used and _LGRAY in used, f"expected white/light_gray dither, got {used}"


def test_fixed_palette_is_cc_default():
    # adaptive=False returns CC's fixed default palette.
    _glyph, _fg, _bg, pal = _enc(_solid((0, 0, 0), 3, 2), adaptive=False)
    assert np.array_equal(pal, _CC_RGB.astype(np.uint8))   # RGB order, 16 entries


def test_adaptive_palette_is_valid_block():
    # The default (adaptive) path yields a valid, content-derived 16x3 RGB palette.
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, size=(5 * 3, 6 * 2, 3), dtype=np.uint8)
    _glyph, _fg, _bg, pal = encode_frame(frame)
    assert pal.shape == (16, 3) and pal.dtype == np.uint8


def test_generate_palette_adapts_to_two_colours():
    # A frame made of two distinct out-of-gamut colours should yield a palette whose
    # entries cluster near those two colours (the quantiser found both).
    teal, gold = (0, 130, 130), (210, 110, 0)
    frame = np.concatenate([_solid(teal, 9, 8), _solid(gold, 9, 8)], axis=1)
    pal = generate_palette(frame).astype(np.int32)
    near_teal = np.abs(pal - np.array(teal)).sum(1).min()
    near_gold = np.abs(pal - np.array(gold)).sum(1).min()
    assert near_teal < 24 and near_gold < 24, f"palette missed a colour: {pal}"


def test_adaptive_beats_fixed_on_out_of_gamut_frame():
    # On content far from CC's fixed palette, the adaptive palette reconstructs the
    # frame more faithfully — measured with S-CIELAB (the perceptual metric the
    # encoder optimises for; lower ΔE = closer) — than the fixed palette does.
    from cc_media import render_cells
    from cc_metrics import mean_scielab

    w, h = 24, 16
    yy, xx = np.mgrid[0:h * 3, 0:w * 2].astype(np.float32)
    frame = np.empty((h * 3, w * 2, 3), np.uint8)          # teal->gold gradient
    t = xx / (w * 2 - 1)
    frame[..., 0] = (t * 210).astype(np.uint8)
    frame[..., 1] = (130 - t * 20 + yy / (h * 3) * 10).astype(np.uint8)
    frame[..., 2] = ((1 - t) * 130).astype(np.uint8)

    adaptive = mean_scielab(frame, render_cells(*encode_frame(frame, adaptive=True)))
    fixed = mean_scielab(frame, render_cells(*encode_frame(frame, adaptive=False)))
    assert adaptive < fixed - 1.0, f"adaptive ΔE {adaptive:.1f} not < fixed {fixed:.1f}"


def test_each_palette_color_round_trips_solid():
    # A solid cell of any palette colour must come back as that colour, empty glyph.
    for i, rgb in enumerate(_CC_RGB.astype(np.uint8)):
        glyph, _fg, bg, _pal = _enc(_solid(tuple(int(c) for c in rgb), 3, 2))
        assert glyph[0, 0] == 0x80
        assert bg[0, 0] == i, f"palette {i} did not round-trip"


# --------------------------------------------------------------------------- #
# GopEncoder: chunking, mid-GOP re-keys, palette resend
# --------------------------------------------------------------------------- #

_FPS = 24


def _gop(gop_samples=ccmf.SAMPLE_RATE * 2):
    return GopEncoder(gop_samples=gop_samples,
                      nominal_duration=round(ccmf.SAMPLE_RATE / _FPS))


def _decode(chunk_bytes):
    _pts, _ctype, payload, _next = ccmf.parse_chunk(chunk_bytes)
    return ccmf.parse_video_payload(payload)


def test_gop_opens_with_palette_and_raw_keyframe():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(20 * 3, 10 * 2, 3), dtype=np.uint8)
    gop = _gop()
    gop.add(0, frame)
    _pts, chunk_bytes = gop.flush(next_pts=2000)
    _w, _h, frames = _decode(chunk_bytes)
    assert [f.encoding for f in frames] == [ccmf.ENC_RAW]


def test_unchanged_frame_becomes_a_repeat_unit():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(20 * 3, 10 * 2, 3), dtype=np.uint8)
    gop = _gop()
    gop.add(0, frame)
    gop.add(2000, frame)              # byte-identical: no content change
    _pts, chunk_bytes = gop.flush(next_pts=4000)
    _w, _h, frames = _decode(chunk_bytes)
    assert [f.encoding for f in frames] == [ccmf.ENC_RAW, ccmf.ENC_REPEAT]


def test_unchanged_frame_skips_the_re_encode(monkeypatch):
    # The repeat-unit *output* above doesn't prove the cheap path was taken --
    # add() could reach the same result by re-encoding and then diffing to a
    # no-op delta.  Assert the actual point of the pixel-identity check: a
    # byte-identical frame must never re-enter the quantiser at all.
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(20 * 3, 10 * 2, 3), dtype=np.uint8)
    gop = _gop()
    gop.add(0, frame)                 # opening keyframe: encode is expected here

    calls = []
    monkeypatch.setattr(GopEncoder, "_encode",
                        lambda self, *a, **kw: calls.append(1) or (None, None, None))
    gop.add(2000, frame)              # byte-identical: must not call _encode
    assert calls == []


def test_scene_cut_appends_a_rekey_without_flushing_the_chunk():
    # A big scene cut mid-GOP must land as another palette+raw pair in the
    # SAME chunk, not force a new one -- the point of the spec's "any order
    # after the opening pair" flexibility (Section 4.4/4.5).
    rng = np.random.default_rng(0)
    dark = np.zeros((20 * 3, 10 * 2, 3), dtype=np.uint8)
    bright = rng.integers(180, 256, size=(20 * 3, 10 * 2, 3), dtype=np.uint8)
    gop = _gop()
    assert gop.add(0, dark) is None
    assert gop.add(2000, bright) is None      # scene cut: must NOT flush
    _pts, chunk_bytes = gop.flush(next_pts=4000)
    _w, _h, frames = _decode(chunk_bytes)
    assert [f.encoding for f in frames] == [ccmf.ENC_RAW, ccmf.ENC_RAW]
    # The new content really did need a new palette -- confirms the resend
    # path still fires when the colours genuinely changed.
    assert not np.array_equal(frames[0].palette, frames[1].palette)


def test_rekey_does_not_resend_an_unchanged_palette():
    # generate_palette() quantises the colour HISTOGRAM, not pixel positions,
    # so a spatial rearrange of the exact same colours forces a re-key (via
    # the RGB scene-cut check) while reproducing byte-identical palette
    # output.  Spec: the palette isn't locked to the raw frame beside it, so
    # an unchanged one must not be queued twice.
    rng = np.random.default_rng(0)
    frame1 = rng.integers(0, 256, size=(20 * 3, 10 * 2, 3), dtype=np.uint8)
    frame2 = np.fliplr(frame1).copy()
    gop = _gop()
    gop.add(0, frame1)
    gop.add(2000, frame2)
    pal_entries = [e for e in gop._entries if e[0] == "pal"]
    assert len(pal_entries) == 1

    _pts, chunk_bytes = gop.flush(next_pts=4000)
    _w, _h, frames = _decode(chunk_bytes)
    assert [f.encoding for f in frames] == [ccmf.ENC_RAW, ccmf.ENC_RAW]
    assert np.array_equal(frames[0].palette, frames[1].palette)


@pytest.mark.parametrize("adaptive", [False, True], ids=["fixed", "adaptive"])
def test_numba_and_numpy_cores_agree(adaptive):
    # The compiled core is the active path; the numpy core is the reference.  They
    # implement the same algorithm, so they must reach the same per-cell colour pair
    # (the dither realisation can differ by float precision).  Checked for BOTH
    # palette strategies: the fixed CC palette and an adaptive palette.
    import cc_encoder as cc
    if not cc._HAVE_NUMBA:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, size=(41 * 3, 82 * 2, 3), dtype=np.uint8)
    idx, h, w = cc._prepare(frame)
    pal_rgb = cc.generate_palette(frame) if adaptive else cc._CC_RGB.astype(np.uint8)
    tables = cc._palette_tables(pal_rgb)                    # (pal, pal2, lin_pal, dot)
    _g_np, fg_np, bg_np = cc._encode_numpy(idx, h, w, *tables)
    _g_nb, fg_nb, bg_nb = cc._encode_numba(idx, h, w, *tables)
    agree = np.mean([
        frozenset((int(fg_np[y, x]), int(bg_np[y, x]))) ==
        frozenset((int(fg_nb[y, x]), int(bg_nb[y, x])))
        for y in range(h) for x in range(w)])
    assert agree > 0.99, f"cores disagree on {(1 - agree) * 100:.1f}% of cells"

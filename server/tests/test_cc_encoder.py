import struct

import numpy as np
import pytest

from cc_encoder import (_CC_RGB, V32_FLAG_BASE, V32_TYPE_VIDEO, decode_32vid,
                        encode_frame, v32_chunk, v32_stream_header)

# palette indices used in the assertions
_WHITE, _LGRAY, _BLUE, _RED = 0, 8, 11, 14


def _solid(rgb, ph, pw):
    return np.tile(np.array(rgb, dtype=np.uint8), (ph, pw, 1))


def _cell(pixels):
    # pixels is a 3-row x 2-col list of RGB triples -> (3, 2, 3) uint8.
    return np.array(pixels, dtype=np.uint8).reshape(3, 2, 3)


def _enc(frame):
    """encode_frame -> decoded (glyph, fg, bg, palette) at the frame's grid."""
    h, w = frame.shape[0] // 3, frame.shape[1] // 2
    return decode_32vid(encode_frame(frame), w, h)


def test_frame_size_matches_32vid_formula():
    W, H = 4, 2
    out = encode_frame(_solid((17, 17, 17), H * 3, W * 2))
    # 32vid uncompressed: screen ceil(W*H/8)*5 + colour W*H + palette 48.
    assert len(out) == ((W * H + 7) // 8) * 5 + W * H + 48


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


def test_palette_block_is_cc_default():
    _glyph, _fg, _bg, pal = _enc(_solid((0, 0, 0), 3, 2))
    assert np.array_equal(pal, _CC_RGB.astype(np.uint8))   # RGB order, 16 entries


def test_each_palette_color_round_trips_solid():
    # A solid cell of any palette colour must come back as that colour, empty glyph.
    for i, rgb in enumerate(_CC_RGB.astype(np.uint8)):
        glyph, _fg, bg, _pal = _enc(_solid(tuple(int(c) for c in rgb), 3, 2))
        assert glyph[0, 0] == 0x80
        assert bg[0, 0] == i, f"palette {i} did not round-trip"


def test_v32_stream_header():
    h = v32_stream_header(82, 41, 24, 2, V32_FLAG_BASE)
    assert h[:4] == b"32VD"
    w, ht, fps, ns, flags = struct.unpack("<HHBBH", h[4:])
    assert (w, ht, fps, ns) == (82, 41, 24, 2)
    assert flags & 0x10                       # bit 4 is always set


def test_v32_chunk_layout():
    data = bytes(range(50))
    c = v32_chunk(V32_TYPE_VIDEO, 1, data)
    size, datalength, ctype = struct.unpack("<IIB", c[:9])
    assert size == len(data) and datalength == 1 and ctype == V32_TYPE_VIDEO
    assert c[9:] == data                      # header is exactly 9 bytes


def test_numba_and_numpy_cores_agree():
    # The compiled core is the active path; the numpy core is the reference.  They
    # implement the same algorithm, so they must reach the same per-cell colour pair
    # (the dither realisation can differ by float precision).
    import cc_encoder as cc
    if not cc._HAVE_NUMBA:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, size=(41 * 3, 82 * 2, 3), dtype=np.uint8)
    idx, h, w = cc._prepare(frame)
    _g_np, fg_np, bg_np = cc._encode_numpy(idx, h, w)
    _g_nb, fg_nb, bg_nb = cc._encode_numba(idx, h, w)
    agree = np.mean([
        frozenset((int(fg_np[y, x]), int(bg_np[y, x]))) ==
        frozenset((int(fg_nb[y, x]), int(bg_nb[y, x])))
        for y in range(h) for x in range(w)])
    assert agree > 0.99, f"cores disagree on {(1 - agree) * 100:.1f}% of cells"

import numpy as np
import pytest

from cc_encoder import _BLIT_LUT, _CC_RGB, encode_frame


def _solid(rgb, ph, pw):
    return np.tile(np.array(rgb, dtype=np.uint8), (ph, pw, 1))


def _cell(pixels):
    # pixels is a 3-row x 2-col list of RGB triples -> (3, 2, 3) uint8.
    return np.array(pixels, dtype=np.uint8).reshape(3, 2, 3)


def test_encode_frame_header_and_size():
    W, H = 4, 2                                   # pixel grid 8 wide x 6 tall
    out = encode_frame(_solid((17, 17, 17), H * 3, W * 2))
    assert out[0:2] == bytes((0, W))              # W uint16 BE
    assert out[2:4] == bytes((0, H))              # H uint16 BE
    assert len(out) == 4 + H * 3 * W              # H rows * 3 strings * W bytes


def test_solid_cell_is_empty_glyph():
    # A solid cell needs no sub-pixel structure: empty glyph 0x80 with the colour
    # carried in the background slot.
    out = encode_frame(_solid((204, 76, 76), 3, 2))   # solid red
    assert out[4] == 0x80                          # text byte: empty glyph
    assert chr(out[4 + 2]) == "e"                   # bg hex: red -> 'e'


def test_glyphs_stay_in_drawing_range():
    rng = np.random.default_rng(0)
    W, H = 6, 5
    out = encode_frame(rng.integers(0, 256, size=(H * 3, W * 2, 3), dtype=np.uint8))
    pos = 4
    for _ in range(H):
        text = out[pos:pos + W]
        assert all(0x80 <= b <= 0x9F for b in text)          # valid 2x3 glyphs
        assert all(b in _BLIT_LUT for b in out[pos + W:pos + 2 * W])     # fg hex
        assert all(b in _BLIT_LUT for b in out[pos + 2 * W:pos + 3 * W])  # bg hex
        pos += 3 * W


def test_pair_search_brackets_a_split_cell():
    # The whole point of scoring the full blit: a cell that is red over the top
    # two rows and blue across the bottom row must resolve to the {red, blue}
    # endpoint pair (not collapse both to one "average" colour).
    out = encode_frame(_cell([
        [(204, 76, 76), (204, 76, 76)],   # red
        [(204, 76, 76), (204, 76, 76)],   # red
        [(51, 102, 204), (51, 102, 204)],  # blue
    ]))
    assert out[0:4] == bytes((0, 1, 0, 1))
    assert 0x80 <= out[4] <= 0x9F                 # a valid drawing glyph
    assert {chr(out[5]), chr(out[6])} == {"e", "b"}   # red + blue, both kept


def test_between_colors_region_is_dithered():
    # A flat field whose colour sits between white and light_gray can't be matched
    # by a single palette entry, so the encoder must dither between the two — both
    # colours appear across the region rather than everything snapping to one.
    W, H = 8, 8
    out = encode_frame(_solid((196, 196, 196), H * 3, W * 2))
    body = out[4:]
    used = set()
    pos = 0
    for _ in range(H):
        for off in (W, 2 * W):             # fg then bg strings
            used.update(chr(c) for c in body[pos + off:pos + off + W])
        pos += 3 * W
    assert "0" in used and "8" in used, f"expected dither between white/light_gray, got {used}"


def test_each_palette_color_round_trips_solid():
    # A solid cell of any palette colour must come back as that colour (in the bg
    # slot, with the empty glyph).
    for i, rgb in enumerate(_CC_RGB.astype(np.uint8)):
        out = encode_frame(_solid(tuple(int(c) for c in rgb), 3, 2))
        assert out[4] == 0x80
        assert chr(out[6]) == "0123456789abcdef"[i], f"palette {i} did not round-trip"

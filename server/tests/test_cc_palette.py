import numpy as np
import pytest

from cc_palette import _BLIT_LUT, _CC_RGB, encode_frame, quantize


def _solid(rgb, ph, pw):
    return np.tile(np.array(rgb, dtype=np.uint8), (ph, pw, 1))


def test_quantize_shape_and_dtype():
    out = quantize(_solid((0, 0, 0), 6, 8))
    assert out.shape == (6, 8)
    assert out.dtype == np.uint8
    assert int(out.max()) <= 15


@pytest.mark.parametrize("rgb,expected", [
    ((240, 240, 240), 0),   # white
    ((204, 76, 76), 14),    # red
    ((17, 17, 17), 15),     # black
    ((51, 102, 204), 11),   # blue
])
def test_quantize_known_colors(rgb, expected):
    assert int(quantize(_solid(rgb, 3, 3))[0, 0]) == expected


def test_neutral_grays_map_to_neutral_palette_entries():
    # Regression: with RGB-Euclidean nearest-colour, neutral greys mapped to
    # chromatic entries (light greys -> pink, mid greys -> brown).  CIELAB ΔE
    # keeps a grey's a*/b* near zero, so every grey on the ramp must land on a
    # neutral CC colour: black(15), gray(7), light_gray(8) or white(0).
    NEUTRAL = {0, 7, 8, 15}
    for v in range(0, 256, 3):
        idx = int(quantize(_solid((v, v, v), 1, 1))[0, 0])
        assert idx in NEUTRAL, f"grey ({v},{v},{v}) mapped to non-neutral index {idx}"


def test_light_chromatic_colors_do_not_wash_to_white():
    # Regression: equal-weight CIELAB ΔE collapsed light, low-chroma colours to
    # white (CC's only light palette entry), washing light blues out.  The chroma
    # up-weight (_CHROMA_WEIGHT) must keep a clear light blue mapped to a blue
    # entry, not white(0).
    for rgb in [(150, 180, 235), (165, 195, 240)]:
        idx = int(quantize(_solid(rgb, 1, 1))[0, 0])
        assert idx in {3, 11}, f"light blue {rgb} washed to index {idx}, expected ltblue/blue"


def test_quantize_each_palette_entry_maps_to_itself():
    for i, rgb in enumerate(_CC_RGB.astype(np.uint8)):
        px = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
        assert int(quantize(px)[0, 0]) == i, f"palette index {i} did not round-trip"


def test_encode_frame_header_and_size():
    # 4 char-cols x 2 char-rows => pixel grid 8 wide x 6 tall.
    W, H = 4, 2
    frame = _solid((17, 17, 17), H * 3, W * 2)   # solid black
    out = encode_frame(frame)

    # header: W, H as uint16 BE
    assert out[0:2] == bytes((0, W))
    assert out[2:4] == bytes((0, H))
    # body: H rows * 3 strings * W bytes
    assert len(out) == 4 + H * 3 * W


def test_encode_frame_solid_cell_is_empty_glyph():
    # A solid cell -> drawing char 0x80 (all background) with bg = that colour.
    W, H = 1, 1
    frame = _solid((204, 76, 76), 3, 2)          # solid red
    out = encode_frame(frame)
    text = out[4]                                # first (only) text byte
    bg = chr(out[4 + 2 * W])                     # bg hex char for the cell
    assert text == 0x80                          # empty glyph
    assert bg == "e"                             # red -> 'e'


def test_encode_frame_chars_in_drawing_range():
    rng = np.random.default_rng(0)
    W, H = 6, 5
    frame = rng.integers(0, 256, size=(H * 3, W * 2, 3), dtype=np.uint8)
    out = encode_frame(frame)
    pos = 4
    for _ in range(H):
        text = out[pos:pos + W]
        assert all(0x80 <= b <= 0x9F for b in text)   # valid 2x3 glyphs
        fg = out[pos + W:pos + 2 * W]
        bg = out[pos + 2 * W:pos + 3 * W]
        assert all(b in _BLIT_LUT for b in fg)
        assert all(b in _BLIT_LUT for b in bg)
        pos += 3 * W


def test_blit_lut_is_hex_digits():
    assert bytes(_BLIT_LUT) == b"0123456789abcdef"

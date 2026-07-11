"""rANS + RLE plane codec (rans.py) round-trips."""

import numpy as np
import pytest

import rans


def _roundtrip(plane, nsym):
    plane = np.asarray(plane, dtype=np.uint8)
    blob = rans.encode_plane(plane, nsym)
    out, end = rans.decode_plane(blob, 0, plane.shape[0], nsym)
    assert end == len(blob), "decoder must consume exactly the plane blob"
    np.testing.assert_array_equal(out, plane)
    return blob


def test_uniform_plane():
    _roundtrip(np.full(1000, 7, np.uint8), 16)


def test_single_cell():
    _roundtrip(np.array([3], np.uint8), 16)


def test_two_symbols_runs():
    plane = np.array([0] * 300 + [5] * 400 + [0] * 5 + [5], np.uint8)
    blob = _roundtrip(plane, 16)
    assert blob[0] == 0            # run-heavy -> RLE mode wins


def test_noisy_plane_uses_plain_mode_and_avoids_length_token_bloat():
    # High-detail content degenerates RLE to length-1 runs; the encoder must fall
    # back to plain rANS (mode 1) so per-run length tokens don't bloat the plane
    # past the bit-packed size.
    rng = np.random.default_rng(9)
    plane = rng.integers(0, 32, 4000, dtype=np.uint8)
    blob = _roundtrip(plane, 32)
    assert blob[0] == 1            # plain rANS mode chosen
    packed = (4000 + 7) // 8 * 5   # 5-bit glyph packing size
    assert len(blob) < packed * 1.2   # near entropy, not ~3 B/cell RLE bloat


def test_all_symbols_glyph():
    rng = np.random.default_rng(1)
    _roundtrip(rng.integers(0, 32, 4096, dtype=np.uint8), 32)


def test_skewed_distribution():
    # 90% one symbol, rest scattered — the realistic flat-background case.
    rng = np.random.default_rng(2)
    plane = np.where(rng.random(5000) < 0.9, 4,
                     rng.integers(0, 16, 5000)).astype(np.uint8)
    blob = _roundtrip(plane, 16)
    # Should compress far below the raw nibble packing (~2500 B).
    assert len(blob) < 1500


@pytest.mark.parametrize("seed", range(10))
def test_random_planes(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 3000))
    nsym = 16 if seed % 2 else 32
    # Mix of runs and noise so both RLE and the entropy stage are exercised.
    if seed % 3 == 0:
        vals = rng.integers(0, nsym, size=int(rng.integers(1, 50)))
        lens = rng.integers(1, 200, size=vals.shape[0])
        plane = np.repeat(vals, lens)[:n].astype(np.uint8)
        if plane.shape[0] < n:
            plane = np.concatenate([plane, np.zeros(n - plane.shape[0], np.uint8)])
    else:
        plane = rng.integers(0, nsym, n, dtype=np.uint8)
    _roundtrip(plane, nsym)


def test_long_run_over_255():
    _roundtrip(np.full(70000 % 65536 + 1000, 2, np.uint8), 16)


def test_full_grid_run():
    # A single run spanning the max grid (u16 length token edge).
    _roundtrip(np.full(65535, 9, np.uint8), 16)

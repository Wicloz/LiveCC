import numpy as np

from cc_metrics import mean_scielab, scielab_delta_e


def test_identical_images_have_zero_delta_e():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    assert mean_scielab(img, img) == 0.0


def test_delta_e_map_shape_and_positivity():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
    b = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
    de = scielab_delta_e(a, b)
    assert de.shape == (30, 40)
    assert de.min() >= 0.0 and de.mean() > 0.0


def test_high_frequency_chroma_error_is_attenuated_more_than_luminance():
    # S-CIELAB's point: the eye has lower chroma acuity, so a fine (high-frequency)
    # error in chroma should cost LESS perceptually than the same-magnitude error in
    # luminance.  Build a 1-px checkerboard error of equal CIELAB amplitude in L vs
    # in a*, and confirm the chroma one scores lower after the spatial filter.
    h, w = 60, 60
    chk = (np.indices((h, w)).sum(0) % 2).astype(np.float32)   # 0/1 checkerboard
    base = np.full((h, w, 3), 128, np.uint8)

    # Perturb a mid-grey along L (achromatic) vs along a* (red-green) using small
    # sRGB deltas; pick deltas giving a similar per-pixel CIELAB amplitude.
    lum = base.copy()
    lum[chk > 0] = (150, 150, 150)                  # lighter squares: pure luminance
    chroma = base.copy()
    chroma[chk > 0] = (150, 110, 128)               # red/green swing at ~equal lightness

    # Equalise by the unfiltered (per-pixel) ΔE so we compare frequency response, not
    # raw amplitude: scale is removed by dividing each by its high-SPD (near-unfiltered) ΔE.
    sharp_l = mean_scielab(base, lum, samples_per_degree=200)
    sharp_c = mean_scielab(base, chroma, samples_per_degree=200)
    blur_l = mean_scielab(base, lum) / sharp_l
    blur_c = mean_scielab(base, chroma) / sharp_c
    assert blur_c < blur_l, f"chroma not attenuated more: chroma {blur_c:.3f} vs luma {blur_l:.3f}"

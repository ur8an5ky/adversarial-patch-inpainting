"""Tests for thresholding."""
import numpy as np

from src.detection.threshold import (
    edge_density, threshold_fixed, threshold_otsu,
)


def test_edge_density_higher_in_textured_region():
    rng = np.random.default_rng(0)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:50, 20:50] = rng.integers(0, 256, (30, 30, 3), dtype=np.uint8)
    d = edge_density(img, ksize=15)
    assert d.shape == (100, 100) and d.dtype == np.uint8
    assert d[35, 35] > d[80, 80]


def test_threshold_fixed_binary():
    resp = np.zeros((10, 10), dtype=np.uint8)
    resp[2:5, 2:5] = 200
    out = threshold_fixed(resp, 100)
    assert set(np.unique(out)).issubset({0, 255})
    assert (out[2:5, 2:5] == 255).all()
    assert out[0, 0] == 0


def test_threshold_otsu_separates():
    resp = np.zeros((20, 20), dtype=np.uint8)
    resp[5:15, 5:15] = 255
    out = threshold_otsu(resp)
    assert set(np.unique(out)).issubset({0, 255})
    assert (out[5:15, 5:15] == 255).all()
"""Tests for the patched-dataset generator."""
import numpy as np

from src.data.generate import square_resize, make_synthetic_patch, paste_patch


def test_square_resize_to_square():
    out = square_resize(np.zeros((40, 80, 3), dtype=np.uint8), 32)
    assert out.shape == (32, 32, 3) and out.dtype == np.uint8


def test_synthetic_patch_shape():
    p = make_synthetic_patch(16, np.random.default_rng(0))
    assert p.shape == (16, 16, 3) and p.dtype == np.uint8


def test_paste_patch_sets_mask_region():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    patch = np.full((5, 5, 3), 255, dtype=np.uint8)
    patched, mask = paste_patch(img, patch, (3, 4))
    assert mask.shape == (20, 20) and mask.dtype == np.uint8
    assert (mask[3:8, 4:9] == 255).all()
    assert mask.sum() == 5 * 5 * 255
    assert (patched[3:8, 4:9] == 255).all()
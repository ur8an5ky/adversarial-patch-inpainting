"""Tests for mask loading and overlay."""
import cv2
import numpy as np

from src.data.mask import load_mask, overlay_mask


def test_load_mask_binarizes(tmp_path):
    arr = np.array([[0, 200], [50, 255]], dtype=np.uint8)
    p = tmp_path / "m.png"
    cv2.imwrite(str(p), arr)

    m = load_mask(p)
    assert m.shape == (2, 2)
    assert m.dtype == np.uint8
    assert set(np.unique(m)).issubset({0, 255})
    assert m[0, 1] == 255 and m[1, 1] == 255   # 200, 255 -> 255
    assert m[0, 0] == 0 and m[1, 0] == 0       # 0, 50  -> 0


def test_overlay_only_changes_masked_region():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0:2, 0:2] = 255

    out = overlay_mask(img, mask)
    assert out.shape == (4, 4, 3) and out.dtype == np.uint8
    assert (out[2:, 2:] == 0).all()            # unmasked unchanged
    assert (out[0:2, 0:2] != 0).any()          # masked tinted
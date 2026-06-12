"""Tests for morphological cleanup."""
import numpy as np

from src.detection.morphology import opening, closing, clean_mask


def test_closing_fills_hole():
    mask = np.full((30, 30), 255, dtype=np.uint8)
    mask[14:16, 14:16] = 0
    out = closing(mask, 9)
    assert out[14, 14] == 255


def test_opening_removes_speck():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 5:25] = 255
    mask[0, 0] = 255
    out = opening(mask, 5)
    assert out[0, 0] == 0
    assert out[15, 15] == 255


def test_clean_mask_binary():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 255
    out = clean_mask(mask)
    assert out.shape == (50, 50)
    assert set(np.unique(out)).issubset({0, 255})
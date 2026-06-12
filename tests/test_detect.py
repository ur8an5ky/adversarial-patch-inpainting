"""Tests for the assembled detect_mask pipeline."""
import numpy as np

from src.detection.detect import detect_mask
from src.data.generate import make_synthetic_patch, paste_patch


def _gradient_base(size=120):
    row = np.linspace(0, 255, size, dtype=np.uint8)
    return np.stack([np.tile(row, (size, 1))] * 3, axis=-1)


def test_detect_mask_finds_patch():
    rng = np.random.default_rng(0)
    base = _gradient_base(120)
    patch = make_synthetic_patch(40, rng)
    patched, gt = paste_patch(base, patch, (40, 40))

    mask = detect_mask(patched, density_ksize=11)
    assert mask.shape == (120, 120) and mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})

    overlap = int(((mask == 255) & (gt == 255)).sum())
    assert overlap > 0.5 * int((gt == 255).sum())


def test_detect_mask_empty_on_smooth():
    base = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = detect_mask(base)
    assert mask.shape == (100, 100)
    assert mask.sum() == 0
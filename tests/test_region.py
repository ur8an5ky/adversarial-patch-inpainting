"""Tests for region extraction."""
import numpy as np

from src.detection.region import largest_component, bounding_box_mask


def test_largest_component_keeps_biggest():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    mask[20:35, 20:35] = 255
    out = largest_component(mask)
    assert out[27, 27] == 255
    assert out[3, 3] == 0
    assert set(np.unique(out)).issubset({0, 255})


def test_largest_component_empty():
    out = largest_component(np.zeros((10, 10), dtype=np.uint8))
    assert out.sum() == 0


def test_bounding_box_fills_rect():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5, 5] = 255
    mask[10, 12] = 255
    out = bounding_box_mask(mask)
    assert (out[5:11, 5:13] == 255).all()
    assert out[0, 0] == 0

def test_densest_component_prefers_high_density():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:8, 2:8] = 255
    mask[30:34, 30:34] = 255
    density = np.zeros((40, 40), dtype=np.uint8)
    density[2:8, 2:8] = 50
    density[30:34, 30:34] = 200

    from src.detection.region import densest_component
    out = densest_component(mask, density)
    assert out[31, 31] == 255
    assert out[3, 3] == 0
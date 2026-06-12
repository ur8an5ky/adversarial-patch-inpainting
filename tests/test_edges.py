"""Tests for edge detection."""
import numpy as np

from src.detection.edges import to_gray, canny_edges, sobel_edges


def _image_with_square():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[15:35, 15:35] = 255
    return img


def test_to_gray_shape():
    g = to_gray(np.zeros((10, 12, 3), dtype=np.uint8))
    assert g.shape == (10, 12) and g.dtype == np.uint8


def test_canny_detects_edges():
    edges = canny_edges(_image_with_square())
    assert edges.shape == (50, 50) and edges.dtype == np.uint8
    assert set(np.unique(edges)).issubset({0, 255})
    assert (edges == 255).sum() > 0


def test_sobel_magnitude_range():
    mag = sobel_edges(_image_with_square())
    assert mag.shape == (50, 50) and mag.dtype == np.uint8
    assert 0 <= mag.min() and mag.max() <= 255
    assert mag.max() > 0
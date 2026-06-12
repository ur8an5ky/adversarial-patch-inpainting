"""Tests for image loading."""
import cv2
import numpy as np
import pytest

from src.data.loading import load_image


def test_load_image_returns_rgb(tmp_path):
    bgr_red = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr_red[:, :, 2] = 255
    path = tmp_path / "red.png"
    cv2.imwrite(str(path), bgr_red)

    img = load_image(path)
    assert img.shape == (4, 4, 3)
    assert img.dtype == np.uint8
    
    assert (img[:, :, 0] == 255).all()
    assert (img[:, :, 1] == 0).all()
    assert (img[:, :, 2] == 0).all()


def test_load_image_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")
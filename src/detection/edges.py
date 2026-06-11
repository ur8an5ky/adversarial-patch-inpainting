"""Edge detection — first step of the mask-detection pipeline."""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an (H, W, 3) RGB image to (H, W) uint8 grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def canny_edges(image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """Binary Canny edge map, (H, W) uint8 {0, 255}."""
    return cv2.Canny(to_gray(image), low, high)


def sobel_edges(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Sobel gradient magnitude as (H, W) uint8, normalized to [0, 255]."""
    gray = to_gray(image)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    if mag.max() > 0:
        mag = mag / mag.max() * 255
    return mag.astype(np.uint8)
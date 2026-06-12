"""Thresholding — turn a continuous response into a binary candidate region."""
from __future__ import annotations

import cv2
import numpy as np

from src.detection.edges import canny_edges
from src.detection.edges import to_gray


def edge_density(image: np.ndarray, ksize: int = 25) -> np.ndarray:
    """Local edge density: Canny edges box-blurred over a window.

    High inside textured/patch regions, low in smooth areas. (H, W) uint8.
    """
    edges = canny_edges(image)
    return cv2.blur(edges, (ksize, ksize))


def threshold_fixed(response: np.ndarray, thresh: int = 60) -> np.ndarray:
    """Global fixed threshold. (H, W) uint8 {0, 255}."""
    _, out = cv2.threshold(response, thresh, 255, cv2.THRESH_BINARY)
    return out


def threshold_otsu(response: np.ndarray) -> np.ndarray:
    """Global Otsu threshold (auto-picked). (H, W) uint8 {0, 255}."""
    _, out = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def threshold_adaptive(response: np.ndarray, block: int = 51, c: int = -10) -> np.ndarray:
    """Local adaptive threshold. (H, W) uint8 {0, 255}."""
    return cv2.adaptiveThreshold(response, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, c)

def local_std(image: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Local standard deviation of grayscale — solid-high inside noisy
    patches, thin/low on natural structure. (H, W) uint8."""
    gray = to_gray(image).astype(np.float32)
    mean = cv2.blur(gray, (ksize, ksize))
    mean_sq = cv2.blur(gray * gray, (ksize, ksize))
    var = np.clip(mean_sq - mean * mean, 0, None)
    return np.sqrt(var).clip(0, 255).astype(np.uint8)
"""Morphological cleanup of the binary candidate region."""
from __future__ import annotations

import cv2
import numpy as np


def _kernel(ksize: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))


def opening(mask: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Erode then dilate: remove small isolated specks."""
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(ksize))


def closing(mask: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Dilate then erode: fill small holes, bridge nearby fragments."""
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(ksize))


def clean_mask(mask: np.ndarray, open_ksize: int = 7,
               close_ksize: int = 15) -> np.ndarray:
    """Opening to kill background noise, then closing to solidify the blob."""
    return closing(opening(mask, open_ksize), close_ksize)
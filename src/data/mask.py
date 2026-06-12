"""Mask loading and visualization."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_mask(path: str | Path) -> np.ndarray:
    """Load a mask as (H, W) uint8 with values {0, 255}.

    Binarizes at 127, so anti-aliased or non-binary PNGs still conform
    to the contract (255 = pixel to reconstruct).
    """
    path = Path(path)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return np.where(gray > 127, 255, 0).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0), alpha: float = 0.5) -> np.ndarray:
    """Tint the masked region of an RGB image for visualization."""
    out = image.copy()
    region = mask == 255
    tint = np.array(color, dtype=np.float32)
    out[region] = ((1 - alpha) * out[region] + alpha * tint).astype(np.uint8)
    return out
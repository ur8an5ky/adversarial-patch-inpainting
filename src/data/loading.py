"""Image loading utilities."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as (H, W, 3) uint8 RGB.

    Handles the OpenCV BGR -> RGB conversion. Grayscale images are
    expanded to 3 channels; an alpha channel, if present, is dropped.
    """
    path = Path(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
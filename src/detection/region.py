"""Region extraction — select the patch blob and produce the final mask."""
from __future__ import annotations

import cv2
import numpy as np


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component. (H, W) uint8 {0, 255}."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def bounding_box_mask(mask: np.ndarray) -> np.ndarray:
    """Fill the bounding box of the foreground (patches are rectangular)."""
    ys, xs = np.where(mask == 255)
    out = np.zeros_like(mask)
    if len(xs) == 0:
        return out
    out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 255
    return out
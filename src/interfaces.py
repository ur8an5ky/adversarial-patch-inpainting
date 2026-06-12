"""Shared interfaces (contracts) between project modules.

Conventions:
- Image: np.ndarray (H, W, 3), uint8, RGB.
- Mask:  np.ndarray (H, W), uint8, {0, 255}; 255 = pixel to reconstruct.
"""
from __future__ import annotations

import numpy as np


def detect_mask(image: np.ndarray) -> np.ndarray:
    """Detect region to reconstruct (e.g. adversarial patch). Returns mask."""
    raise NotImplementedError


def inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reconstruct image in the masked region. Returns reconstructed image."""
    raise NotImplementedError


def evaluate(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute reconstruction metrics within the mask. Returns metrics dict."""
    raise NotImplementedError
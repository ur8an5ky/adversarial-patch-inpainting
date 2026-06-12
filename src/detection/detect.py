"""Assembled patch-detection pipeline."""
from __future__ import annotations

import cv2
import numpy as np

from src.detection.threshold import (
    edge_density, local_std, threshold_fixed, threshold_otsu,
)
from src.detection.morphology import clean_mask
from src.detection.region import (
    largest_component, densest_component, bounding_box_mask,
)


def detect_mask(image: np.ndarray, *,
                signal: str = "local_std",
                selection: str = "densest",
                density_ksize: int | None = None,
                threshold: int | None = None,
                open_ksize: int = 7,
                close_ksize: int = 15,
                dilate_ksize: int = 5) -> np.ndarray:
    """Detect the patch region. Returns (H, W) uint8 {0, 255}.

    signal:    "local_std" (default) or "edge_density".
    selection: "densest" (default, picks the highest-variance blob) or "largest".
    threshold: fixed threshold, or None for Otsu.
    """
    if signal == "local_std":
        density = local_std(image, density_ksize or 7)
    else:
        density = edge_density(image, density_ksize or 25)

    if int(density.max()) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    binary = (threshold_otsu(density) if threshold is None
              else threshold_fixed(density, threshold))
    cleaned = clean_mask(binary, open_ksize, close_ksize)

    if selection == "densest":
        blob = densest_component(cleaned, density)
    else:
        blob = largest_component(cleaned)
    mask = bounding_box_mask(blob)

    if dilate_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (dilate_ksize, dilate_ksize))
        mask = cv2.dilate(mask, kernel)
    return mask


def detect_masks(images: list[np.ndarray], **kwargs) -> list[np.ndarray]:
    """Run detect_mask over a batch of images."""
    return [detect_mask(img, **kwargs) for img in images]
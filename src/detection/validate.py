"""Validate detection against ground-truth masks (IoU)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.loading import load_image
from src.data.mask import load_mask
from src.data.inventory import find_images
from src.detection.detect import detect_mask


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-union of two binary {0, 255} masks."""
    p = pred == 255
    g = gt == 255
    union = int(np.logical_or(p, g).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


def evaluate_detection(data_dir, **detect_kwargs) -> dict:
    """Run detect_mask on <data_dir>/patched, score IoU vs <data_dir>/mask.

    Returns {"mean_iou", "ious", "names"}.
    """
    patched_dir = Path(data_dir) / "patched"
    mask_dir = Path(data_dir) / "mask"
    ious, names = [], []
    for p in find_images(patched_dir):
        gt = load_mask(mask_dir / p.name)
        pred = detect_mask(load_image(p), **detect_kwargs)
        ious.append(iou(pred, gt))
        names.append(p.name)
    mean = float(np.mean(ious)) if ious else 0.0
    return {"mean_iou": mean, "ious": ious, "names": names}
"""Tests for detection validation (IoU)."""
import cv2
import numpy as np

from src.detection.validate import iou, evaluate_detection
from src.data.generate import make_synthetic_patch, paste_patch


def test_iou_identical():
    a = np.zeros((10, 10), dtype=np.uint8); a[2:6, 2:6] = 255
    assert iou(a, a) == 1.0


def test_iou_disjoint():
    a = np.zeros((10, 10), dtype=np.uint8); a[1:4, 1:4] = 255
    b = np.zeros((10, 10), dtype=np.uint8); b[6:9, 6:9] = 255
    assert iou(a, b) == 0.0


def test_iou_both_empty():
    z = np.zeros((5, 5), dtype=np.uint8)
    assert iou(z, z) == 1.0


def test_evaluate_detection(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "patched").mkdir()
    (tmp_path / "mask").mkdir()
    for i in range(3):
        row = np.linspace(0, 255, 224, dtype=np.uint8)
        base = np.stack([np.tile(row, (224, 1))] * 3, axis=-1)
        patch = make_synthetic_patch(50, rng)
        patched, gt = paste_patch(base, patch, (60, 60))
        cv2.imwrite(str(tmp_path / "patched" / f"img{i}.png"), cv2.cvtColor(patched, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(tmp_path / "mask" / f"img{i}.png"), gt)

    res = evaluate_detection(tmp_path)
    assert res["mean_iou"] > 0.5
    assert len(res["ious"]) == 3
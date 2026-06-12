"""Tests for the annotation tool (GUI-free logic)."""
import cv2
import numpy as np

from src.data.annotate import MaskAnnotator


def test_annotator_paints_on_click():
    ann = MaskAnnotator(np.zeros((50, 50, 3), dtype=np.uint8), brush=5)
    assert ann.mask.sum() == 0
    ann.on_mouse(cv2.EVENT_LBUTTONDOWN, 25, 25, 0, None)
    assert ann.mask[25, 25] == 255
    assert ann.mask.sum() > 0


def test_annotator_render_shape():
    ann = MaskAnnotator(np.zeros((30, 40, 3), dtype=np.uint8))
    out = ann.render()
    assert out.shape == (30, 40, 3)
    assert out.dtype == np.uint8
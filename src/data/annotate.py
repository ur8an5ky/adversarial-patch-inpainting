"""Interactive free-form mask annotation tool.

Usage:
    python -m src.data.annotate path/to/image.jpg [-o mask.png] [-b 15]

Controls:
    left mouse drag - paint mask
    r               - reset
    s               - save mask
    q / Esc         - quit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.data.loading import load_image


class MaskAnnotator:
    def __init__(self, image: np.ndarray, brush: int = 15):
        self.display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.mask = np.zeros(image.shape[:2], dtype=np.uint8)
        self.brush = brush
        self.drawing = False
        self.last = (0, 0)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last = (x, y)
            cv2.circle(self.mask, (x, y), self.brush, 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            cv2.line(self.mask, self.last, (x, y), 255, self.brush * 2)
            self.last = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def render(self) -> np.ndarray:
        out = self.display.copy()
        red = np.zeros_like(out)
        red[:, :, 2] = 255                  # red in BGR
        region = self.mask == 255
        out[region] = (0.5 * out[region] + 0.5 * red[region]).astype(np.uint8)
        return out


def annotate(image_path: Path, out_path: Path, brush: int) -> None:
    image = load_image(image_path)
    ann = MaskAnnotator(image, brush)
    win = "annotate (drag: paint | r: reset | s: save | q: quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, ann.on_mouse)

    while True:
        cv2.imshow(win, ann.render())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):           # q or Esc
            break
        if key == ord("r"):
            ann.mask[:] = 0
        if key == ord("s"):
            cv2.imwrite(str(out_path), ann.mask)
            print(f"Saved mask to {out_path}")
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive mask annotation.")
    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=None, help="output mask path (default: <image>_mask.png)")
    parser.add_argument("-b", "--brush", type=int, default=15, help="brush radius in pixels")
    args = parser.parse_args()
    out = args.out or args.image.with_name(args.image.stem + "_mask.png")
    annotate(args.image, out, args.brush)


if __name__ == "__main__":
    main()
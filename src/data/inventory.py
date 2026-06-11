"""Dataset inventory: counts images, sizes, formats, and checks for masks.

Usage:
    python -m src.data.inventory data/imagenet-patch
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def inventory(root: Path) -> None:
    images = find_images(root)
    if not images:
        print(f"No images in: {root}")
        return

    sizes: Counter = Counter()
    formats: Counter = Counter()
    for p in images:
        with Image.open(p) as im:
            sizes[im.size] += 1
            formats[im.format] += 1

    print(f"Directory:    {root}")
    print(f"Image count:  {len(images)}")
    print(f"Formats:      {dict(formats)}")
    print("Sizes (top 5):")
    for (w, h), n in sizes.most_common(5):
        print(f"  {w}x{h}: {n}")

    masks = [p for p in images if "mask" in p.stem.lower() or "mask" in p.parent.name.lower()]
    print(f"Files that look like masks: {len(masks)}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    inventory(target)
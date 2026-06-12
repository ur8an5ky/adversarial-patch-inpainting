"""Generate (clean, patched, mask) triples by pasting a patch onto base images.

Usage:
    python -m src.data.generate data/imagenette2-320/val --out data/generated --n 50 --size 224 --patch-size 60

By default a synthetic high-frequency patch is generated (no downloads,
no torch). Pass --patch path/to/patch.png to use a real adversarial patch
(e.g. extracted from pralab/ImageNet-Patch) for the final dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.data.inventory import find_images
from src.data.loading import load_image


def square_resize(image: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to square, then resize to (size, size)."""
    h, w = image.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    cropped = image[top:top + side, left:left + side]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def make_synthetic_patch(size: int, rng: np.random.Generator) -> np.ndarray:
    """High-frequency colorful patch — a stand-in for an adversarial patch."""
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def paste_patch(image: np.ndarray, patch: np.ndarray, top_left: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Paste patch onto image; return (patched_image, mask)."""
    y, x = top_left
    ph, pw = patch.shape[:2]
    patched = image.copy()
    patched[y:y + ph, x:x + pw] = patch
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y:y + ph, x:x + pw] = 255
    return patched, mask


def apply_random_patch(image: np.ndarray, patch: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    ph, pw = patch.shape[:2]
    y = int(rng.integers(0, h - ph + 1))
    x = int(rng.integers(0, w - pw + 1))
    return paste_patch(image, patch, (y, x))


def generate(base_dir: Path, out_dir: Path, n: int, size: int, patch_size: int, patch_path: Path | None, seed: int) -> None:
    rng = np.random.default_rng(seed)
    bases = find_images(base_dir)
    if not bases:
        raise SystemExit(f"No base images in {base_dir}")
    idx = rng.permutation(len(bases))[:n]
    bases = [bases[i] for i in idx]

    patch_img = load_image(patch_path) if patch_path else None
    for sub in ("clean", "patched", "mask"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(bases):
        clean = square_resize(load_image(p), size)
        patch = (cv2.resize(patch_img, (patch_size, patch_size))
                 if patch_img is not None
                 else make_synthetic_patch(patch_size, rng))
        patched, mask = apply_random_patch(clean, patch, rng)

        name = f"img{i:04d}.png"
        cv2.imwrite(str(out_dir / "clean" / name), cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "patched" / name), cv2.cvtColor(patched, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "mask" / name), mask)

    print(f"Wrote {len(bases)} triples to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate patched-image dataset.")
    ap.add_argument("base_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/generated"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--patch-size", type=int, default=60)
    ap.add_argument("--patch", type=Path, default=None, help="patch image (default: synthetic)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.base_dir, args.out, args.n, args.size, args.patch_size, args.patch, args.seed)


if __name__ == "__main__":
    main()
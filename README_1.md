# Part 1 - Data & Patch Detection

Author: *Jakub Urbański*

This part covers dataset preparation and the classical (OpenCV-based) detection
of adversarial patches. It produces the `(clean, patched, mask)` triples the
rest of the project runs on, and a `detect_mask` function that recovers the
patch region from an image using only classical image-analysis techniques.

## Modules

| Module | Purpose |
| --- | --- |
| `src/data/loading.py` | `load_image` - read an image as `(H, W, 3)` uint8 **RGB** (handles BGR→RGB). |
| `src/data/mask.py` | `load_mask`, `overlay_mask` - load/visualize binary masks. |
| `src/data/inventory.py` | `find_images`, `inventory` - list/summarize a dataset folder. |
| `src/data/generate.py` | Build `(clean, patched, mask)` triples by pasting a patch onto base images. |
| `src/data/annotate.py` | Interactive brush tool for drawing masks by hand. |
| `src/detection/edges.py` | `to_gray`, `canny_edges`, `sobel_edges`. |
| `src/detection/threshold.py` | `edge_density`, `local_std`, fixed / Otsu / adaptive thresholds. |
| `src/detection/morphology.py` | `opening`, `closing`, `clean_mask`. |
| `src/detection/region.py` | `largest_component`, `densest_component`, `bounding_box_mask`. |
| `src/detection/detect.py` | **`detect_mask`** - the assembled pipeline. |
| `src/detection/validate.py` | `iou`, `evaluate_detection` - score detection vs ground truth. |

## Contract (handoff for Parts 2 & 3)

All modules share these formats:

- **Image:** `np.ndarray` `(H, W, 3)`, dtype `uint8`, channel order **RGB**.
- **Mask:** `np.ndarray` `(H, W)`, dtype `uint8`, values `{0, 255}`; `255` = pixel
  to reconstruct.

Main entry point:

```python
from src.data.loading import load_image
from src.detection.detect import detect_mask

image = load_image("data/generated/patched/img0000.png")  # (H, W, 3) uint8 RGB
mask = detect_mask(image)                                  # (H, W) uint8 {0, 255}
```

Generated dataset layout (parallel filenames across folders):

```
data/generated/
  clean/    img0000.png   # original image (PSNR/SSIM target for Part 3)
  patched/  img0000.png   # image with the patch (detection / inpainting input)
  mask/     img0000.png   # ground-truth patch location (Part 1 validation, Part 2 inpainting)
```

## Generating the dataset

```bash
python -m src.data.generate data/imagenette2-320/val --out data/generated --n 50
```

Takes clean base images (Imagenette), pastes a patch at a random known location,
and writes the triple. A synthetic high-frequency patch is used by default (no
downloads, no torch). To use a real adversarial patch instead, extract a patch
to PNG and pass `--patch path/to/patch.png`.

> **Status:** `data/generated` is a working set built with a *synthetic* patch
> (a stand-in). It is sufficient for developing and validating detection and
> inpainting. For the final experiments - in particular the classifier-fooling
> test - regenerate with real **ImageNet-Patch** patches via `--patch`; the base
> images and the whole pipeline stay unchanged.

## Detection pipeline (`detect_mask`)

The patch is detected as the most anomalous high-frequency region:

1. **Response map** - `local_std` (local variance; default) or `edge_density`
   (Canny edges blurred). High inside the patch, low on smooth areas.
2. **Threshold** - Otsu (default) or a fixed threshold → binary candidate.
3. **Morphology** - opening removes small specks, closing fills holes.
4. **Region selection** - `densest_component` keeps the highest-variance blob
   (default), so a large textured *subject* doesn't outvote the patch.
5. **Bounding box + dilation** - fill the blob's box and expand a few px to
   cover the patch's sharp edge.

Defaults (`signal="local_std"`, `selection="densest"`) were chosen by IoU
comparison - see `notebooks/03_detection_validation.ipynb`. Tunable knobs:
`density_ksize`, `threshold`, `open_ksize`, `close_ksize`, `dilate_ksize`.

## Manual annotation

```bash
python -m src.data.annotate data/some_image.png
```

Drag to paint the region, `s` saves the mask, `r` resets, `q` quits. Requires a
graphical display.

## Validation

```python
from src.detection.validate import evaluate_detection
res = evaluate_detection("data/generated")        # uses patched/ vs mask/
print(res["mean_iou"])
```

`notebooks/03_detection_validation.ipynb` scores detection against the
ground-truth masks and shows the worst cases. Mean IoU over a mixed Imagenette
set (synthetic patches), by signal and region-selection rule:

| signal \ selection | largest | densest |
| --- | --- | --- |
| edge_density | 0.200 | 0.272 |
| local_std | **0.306** | 0.236 |

Findings:

- `local_std` + `largest` is the best configuration and is the default.
- The selection rule must match the signal: `densest_component` helps
  `edge_density` (which counts edges, so the patch wins) but hurts `local_std`
  (which measures contrast, so a single sharp background edge competes with the
  patch).
- Absolute IoU is moderate — detection works well on simple backgrounds and
  degrades on cluttered scenes (see "Known limitation").

## Known limitation

Classical detection works well when the patch is the dominant high-frequency
anomaly (simple / smooth backgrounds) and degrades on cluttered scenes, where a
textured subject competes with the patch. This is an honest, expected result and
**does not block the project**: the generator provides ground-truth masks, which
Parts 2 & 3 use for inpainting and for PSNR/SSIM - detection is evaluated as a
separate component via IoU.

## Tests

```bash
pytest
```

Covers loading, masks, generation, every detection step, the assembled
`detect_mask`, and IoU validation.
# Classical Inpainting Experiments

Author: *Mateusz Wardawa*

This project automates **image inpainting** (reconstruction of damaged/masked regions
of images) using classical OpenCV algorithms as well as a custom color-propagation
method.

## Prerequisites
Before running the main script, you need to download the dataset and generate the required data.

1. Download the **imagenette2-320** dataset and ensure it is placed in the `data/` directory.
2. Run the following command to generate the data:

```bash
python -m src.data.generate data/imagenette2-320/val --out data/generated --n 50
```
## How it works

The pipeline operates on a data directory with a fixed structure:

```
data/
├── clean/      # original, undamaged images
├── mask/       # binary masks (255 = pixel to be reconstructed, 0 = "healthy" pixel)
└── patched/    # images with damage applied (to be repaired)
```

For each pair (image from `patched/`, mask from `mask/`), the script runs three
reconstruction methods and saves the results in the `inpainted/` directory:

```
data/
└── inpainted/
    ├── telea/radius={r}/
    ├── navierStokes/radius={r}/
    └── propagation/radius={r}/
```

## Key components

### `Inpaint`
The class responsible for orchestrating the whole process:
- **`_check_data_dirs`** – verifies that the required directories exist (`clean`, `mask`, `patched`).
- **`_create_data_dirs`** – pre-creates the output directories for each method.
- **`generate_data`** – a generator that lazily loads image/mask pairs, so even large
  datasets don't put excessive pressure on RAM.
- **`run`** – the main loop: for each image/mask pair it calls all three inpainting
  methods and saves the results (after RGB → BGR conversion) as PNG/JPG files.

### `InpaintWrapper`
A common interface for the three inpainting methods:
- **`telea`** – OpenCV's `cv2.INPAINT_TELEA` method (Fast Marching Method).
- **`ns`** – OpenCV's `cv2.INPAINT_NS` method (Navier-Stokes).
- **`propagation`** – a custom iterative algorithm that:
  1. finds the outer contour of the mask (`cv2.findContours`),
  2. for each pixel on the contour, gathers "healthy" pixels from a local window whose
     size depends on `radius`,
  3. fills the pixel with the mean color of those neighboring pixels and removes it from
     the mask,
  4. repeats the process, "shrinking" the mask from the edges inward until the whole
     region has been reconstructed.

The `radius` parameter controls the size of the neighborhood used by all three
algorithms.

## Requirements

- Python 3.9+
- `numpy`
- `opencv-python`

Install dependencies:

```bash
pip install numpy opencv-python
```

## How to run

The script is run from the command line, passing the path to the data directory:

```bash
python src/inpainting/inpaint.py data/generated --radius 3
```

Arguments:
- `data_path` (required) – path to the root directory containing the `clean`, `mask`,
  and `patched` subdirectories.
- `--radius` (optional, default `3`) – neighborhood radius used by the inpainting
  algorithms.

Once finished, the results for each of the three methods will be located in
`data_path/inpainted/<method>/radius=<radius>/`, with the same filenames as the
original images from `patched/`.

## Experiments notebook

The `04_classical_inpainting_experiments.ipynb` notebook contains experiments using the
classes above — including a comparison of the inpainting methods for different `radius`
values and a visual assessment of reconstruction quality.

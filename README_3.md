Oczywiście! Dodałem nową sekcję opisującą proces tworzenia patcha (wraz ze wspomnieniem o notatniku `00_adversarial_patch_generation.ipynb`).

Zaktualizowany plik README zachowałem w języku angielskim, aby był w pełni spójny z resztą Twojego projektu. Dodatkowo uaktualniłem sekcję instalacji o wzmiankę dotyczącą wymogów CUDA, o których rozmawialiśmy wcześniej.

Oto gotowy tekst do wklejenia:

---

# Part 3 - Evaluation and ML Benchmark

Author: *Osoba 3*

This section introduces quantitative evaluation of the classically reconstructed images and establishes a Machine Learning benchmark using pre-trained deep neural networks. It computes objective quality metrics and estimates the classification impact of inpainting on adversarial patches.

## Modules

| Module | Purpose |
| --- | --- |
| `src/evaluation/metrics.py` | Implementation of `masked_psnr` and `masked_ssim` to compare predictions vs ground truth specifically inside the patch region. Exposes main `evaluate` function. |
| `src/evaluation/classifier.py` | `PatchClassifierBenchmark` - integrates a pre-trained ImageNet ResNet50 model (via `torchvision`) to evaluate prediction shifts upon patching and reconstruction. |
| `src/evaluation/ml_inpaint.py` | `LaMaInpainter` - Wrapper around the state-of-the-art Large Mask Inpainting (LaMa) model as the ultimate reference for classical algorithms. |

## Contract (Evaluation interface)

The core contract implemented allows standardized metric collection directly adhering to the main pipeline types:

```python
from src.evaluation.metrics import evaluate

metrics = evaluate(pred=reconstructed_img, target=clean_img, mask=mask_img)
# Returns: {"psnr": 32.41, "ssim": 0.94}

```

The computations only reflect performance over the `mask_img > 127` pixels, as evaluating the entire unmasked area would skew results (smooth backgrounds are identical).

## Setup & Requirements

Additional dependencies were introduced for deep learning integration:

```bash
pip install torch torchvision scikit-image simple-lama-inpainting

```

* `torch` & `torchvision`: for ResNet50 class prediction.
* `simple-lama-inpainting`: lightweight interface mapping strictly to the original LaMa architecture.
* `scikit-image`: structural similarity computations (`ssim`).

These requirements are embedded alongside earlier dependencies in `requirements.txt`.
*Note: If you are running this on newer hardware (e.g., RTX 50-series), ensure your `requirements.txt` points to the PyTorch Nightly CUDA index (`--extra-index-url https://download.pytorch.org/whl/nightly/cu124`) to avoid kernel execution errors.*

## Adversarial Patch Generation (Notebook 00)

Before the pipeline can be evaluated, a robust adversarial patch must be created. This is handled by the **`00_adversarial_patch_generation.ipynb`** notebook.

Instead of generating a static noisy image, this notebook visualizes the real-time PyTorch optimization of a $100 \times 100$ patch specifically engineered to fool the ResNet50 classifier (targeting high-frequency features, e.g., class `340` - Zebra).

To ensure the patch is indestructible and bypasses the "simulation-to-reality gap" during dataset generation, it is dynamically trained directly over real background images from our dataset with random scaling and translation. The final optimized image is automatically saved to `data/adversarial_patch.png` to be ingested by the main pipeline.

## ML Benchmark (LaMa)

The project leverages **LaMa** due to its exceptional texture synthesis and large-mask translation capabilities. It serves as an upper ceiling for our OpenCV classical algorithms (Telea / NS).

```python
from src.evaluation.ml_inpaint import LaMaInpainter

lama = LaMaInpainter()
img_reconstructed_lama = lama.inpaint(patched_img, mask_img)

```

## Classifier Benchmark

We determine if the reconstructed texture neutralizes the adversarial patch. Using `resnet50` from torchvision, we verify if the predicted class ID (1-1000) for the inpainted image restores or aligns consistently with the clean source, instead of converging on the patch's target class.

```python
from src.evaluation.classifier import PatchClassifierBenchmark

benchmark = PatchClassifierBenchmark()
predictions = benchmark.evaluate_triplet(clean_img, patched_img, img_reconstructed_lama)

print(predictions)
# {'clean_pred': 123, 'patched_pred': 859, 'inpainted_pred': 123}

print(f"Patched matches clean: {predictions['patched_pred'] == predictions['clean_pred']}")
print(f"Inpainted matches clean: {predictions['inpainted_pred'] == predictions['clean_pred']}")

```

If you encounter a `NotImplementedError: Could not run 'aten::empty_strided' with arguments from the 'CUDA' backend.` error during LaMa setup, ensure you instantiate LaMa with CPUDevice, or that your local torch is fully capable. In a Jupyter Notebook, **you must Restart the Kernel** after patching or reinstalling torch to apply the fix.

## Experiments Notebooks

The pipeline is fully demonstrable via interactive Jupyter Notebooks:

* **`00_adversarial_patch_generation.ipynb`**: Real-time generation of the target patch.
* **`05_evaluation_and_ml_benchmark.ipynb`**: Unites Part 1 (Detection), Part 2 (Classical Inpainting), and Part 3 (Evaluation & ML) offering a side-by-side PSNR/SSIM contrast as well as visual inspection of PyTorch results seamlessly juxtaposed with Telea, Navier-Stokes, and Propagation methods.
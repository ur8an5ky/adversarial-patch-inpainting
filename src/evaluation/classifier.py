import torch
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from PIL import Image

class PatchClassifierBenchmark:
    """
    Evaluates how adversarial patches and inpainting affect the output of a pre-trained classifier.
    """
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load a standard pre-trained ImageNet model
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights).to(self.device).eval()
        self.transforms = weights.transforms()
        
    def predict(self, image: np.ndarray) -> int:
        """
        Returns the top predicted ImageNet class index for a single RGB image.
        image: (H, W, 3) uint8 RGB
        """
        # Convert numpy (H, W, 3) uint8 to PIL for torchvision transforms
        pil_img = Image.fromarray(image)
        
        # Apply standard classification transforms (resize, crop, normalize)
        img_t = self.transforms(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_t)
            _, preds = torch.max(outputs, 1)
            
        return preds.item()

    def evaluate_triplet(self, clean: np.ndarray, patched: np.ndarray, inpainted: np.ndarray) -> dict[str, int]:
        """
        Returns the predicted classes for the original, patched, and reconstructed image.
        """
        return {
            "clean_pred": self.predict(clean),
            "patched_pred": self.predict(patched),
            "inpainted_pred": self.predict(inpainted)
        }

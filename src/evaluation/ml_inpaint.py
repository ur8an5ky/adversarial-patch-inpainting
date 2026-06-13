import numpy as np
from PIL import Image

try:
    from simple_lama_inpainting import SimpleLama
except ImportError:
    # Optional dependency if running purely classical without ML benchmark installed
    SimpleLama = None


class LaMaInpainter:
    """
    Wrapper for the LaMa (Large Mask Inpainting) model.
    Serves as the ML benchmark returning SOTA results compared to classical methods.
    """
    def __init__(self):
        if SimpleLama is None:
            raise ImportError("simple-lama-inpainting package is required to use LaMaInpainter.")
        
        self.model = SimpleLama()
        
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Reconstruct image in the masked region.
        image: (H, W, 3) uint8 RGB
        mask:  (H, W) uint8 {0, 255}
        """
        pil_img = Image.fromarray(image)
        pil_mask = Image.fromarray(mask).convert('L')
        
        result_img = self.model(pil_img, pil_mask)
        
        return np.array(result_img)

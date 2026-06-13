import numpy as np
import cv2
import os
from typing import Generator
import argparse
from inpaint_wrapper import InpaintWrapper

class Inpaint:
    """
    A class to automate the image inpainting pipeline across an entire dataset
    using various classic computer vision algorithms.
    """
    def __init__(self, data_path: str, radius: int = 3):
        """
        Initializes the Inpaint pipeline, validates inputs, and sets up directories.
        
        Args:
            data_path (str): Path to the root data directory.
            radius (int): Inpainting neighborhood radius used by the algorithms.
        """
        self.data_path = data_path
        self.radius = radius
        self._check_data_dirs()
        self._create_data_dirs()

    def _load_img(self, path: str) -> np.ndarray:
        """Loads an image from disk and converts it from BGR (OpenCV default) to RGB."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Unable to load the image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _load_mask(self, path: str) -> np.ndarray:
        """Loads a binary mask image from disk in grayscale mode (values 0 and 255)."""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Unable to load the mask: {path}")
        return mask
    
    def _check_data_dirs(self):
        """Validates that all required source directories exist before processing."""
        dirs = ["clean", "mask", "patched"]
        for dir in dirs:
            path = os.path.join(self.data_path,dir)
            if not os.path.isdir(path):
                raise FileNotFoundError(f"The directory does not exist: {path}")
            
    def _create_data_dirs(self):
        """
        Pre-creates all necessary target directories for the output results.
        Doing this upfront avoids repetitive OS I/O calls inside the processing loop.
        """
        dirs = ["telea", "navierStokes", "propagation"]
        for dir in dirs:
            path = os.path.join(self.data_path, "inpainted", dir, f"radius={self.radius}")
            os.makedirs(path, exist_ok=True)
            
            
    def generate_data(self) -> Generator[tuple[str, np.ndarray, np.ndarray], None, None]:
        """
        A data generator that lazily loads image-mask pairs from disk.
        This prevents RAM exhaustion when handling large datasets.
        
        Yields:
            tuple[str, np.ndarray, np.ndarray]: A tuple containing (file_name, RGB_image, grayscale_mask)
        """
        patched_dir = os.path.join(self.data_path, "patched")
        mask_dir = os.path.join(self.data_path, "mask")

        file_names = sorted(os.listdir(patched_dir))

        for file_name in file_names:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                patched_path = os.path.join(patched_dir, file_name)
                mask_path = os.path.join(mask_dir, file_name)

                img = self._load_img(patched_path)
                mask = self._load_mask(mask_path)

                yield file_name, img, mask
    
    def run(self):
        """
        Main execution loop. Processes each image fetched from the generator 
        using three distinct inpainting techniques and saves outputs to disk.
        """
        for file_name, img, mask in self.generate_data():
            telea = InpaintWrapper.apply_inpaint(img, mask, "telea", self.radius)
            ns = InpaintWrapper.apply_inpaint(img, mask, "ns", self.radius)
            propagation = InpaintWrapper.apply_inpaint(img, mask, "propagation", self.radius)

            for result, method_name in [(telea, "telea"), (ns, "navierStokes"), (propagation, "propagation")]:
                output_path = os.path.join(self.data_path, "inpainted", method_name, f"radius={self.radius}", file_name)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, result)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classic image inpainting using OpenCV")
    parser.add_argument("data_path", type=str, help="Path to the main data directory (e.g. data/generated)")
    parser.add_argument("--radius", type=int, default=3, help="Radius for inpainting algorithms (default: 3)")
    args = parser.parse_args()
    
    print(f"Running inpainting for: {args.data_path} with a radius of {args.radius}")

    i = Inpaint(args.data_path, radius=args.radius)
    i.run()
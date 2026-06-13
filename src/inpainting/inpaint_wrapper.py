import numpy as np
import cv2

class InpaintWrapper:
    """
    A wrapper class providing a unified interface for various image inpainting methods,
    including standard OpenCV implementations and a custom boundary propagation algorithm.
    """
    @staticmethod
    def apply_inpaint(image: np.ndarray, mask: np.ndarray, method: str = "telea", radius: int = 3):
        """
        Routes the inpainting request to the specified algorithm.

        Args:
            image (np.ndarray): The source image to patch (RGB or BGR depending on pipeline).
            mask (np.ndarray): A binary mask where 255 represents pixels to be inpainted.
            method (str): The algorithm to use ("telea", "ns", or "propagation"). Defaults to "telea".
            radius (int): The neighborhood radius for the inpainting algorithms. Defaults to 3.

        Returns:
            np.ndarray: The reconstructed (inpainted) image.

        Raises:
            ValueError: If an unrecognized inpainting method name is provided.
        """
        if method == "telea":
            return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
        elif method == "ns":
            return cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)
        elif method == "propagation":
            return InpaintWrapper.inpaint_propagation(image, mask, radius)
        else:
            raise ValueError(f"An unknown inpainting method: {method}")
        

    @staticmethod
    def inpaint_propagation(image: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
        """
        A custom iterative inpainting algorithm that propagates colors from unmasked ('healthy') 
        pixels into the masked region, stripping away the mask from the outer edges inward.

        Args:
            image (np.ndarray): The source image to be reconstructed.
            mask (np.ndarray): The binary mask (grayscale, uint8) marking damaged pixels with 255.
            radius (int): The size of the local neighborhood window to sample healthy colors from.

        Returns:
            np.ndarray: A completely inpainted copy of the source image.
        """
        image_cp = image.copy()
        mask_cp = mask.copy() 
        h, w = image_cp.shape[:2]
        while np.any(mask_cp == 255):
            contours, _ = cv2.findContours(mask_cp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            edge_contour = contours[0]
            edge_points = edge_contour.reshape(-1, 2)
            for x, y in edge_points:
                y_min = max(0, y - radius)
                y_max = min(h, y + radius + 1)
                x_min = max(0, x - radius)
                x_max = min(w, x + radius + 1)

                mask_section = mask_cp[y_min:y_max, x_min:x_max]
                image_section = image_cp[y_min:y_max, x_min:x_max]
                healthy_pixels = image_section[(mask_section == 0)]

                if len(healthy_pixels) > 0:
                    new_pixel = np.mean(healthy_pixels, axis=0)
                    image_cp[y, x] = new_pixel
                    mask_cp[y, x] = 0

        return image_cp
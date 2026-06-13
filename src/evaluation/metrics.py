import numpy as np
from skimage.metrics import structural_similarity as ssim

def masked_psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute PSNR only within the masked region.
    mask: 2D array where 255 indicates the region to evaluate.
    """
    mask_bool = mask > 127
    if not np.any(mask_bool):
        return float('inf')  # No pixels to evaluate

    diff = pred[mask_bool].astype(np.float32) - target[mask_bool].astype(np.float32)
    mse = np.mean(diff ** 2)
    
    if mse == 0:
        return float('inf')
        
    max_pixel = 255.0
    return 10.0 * np.log10((max_pixel ** 2) / mse)


def masked_ssim(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute SSIM and average only within the masked region.
    mask: 2D array where 255 indicates the region to evaluate.
    """
    mask_bool = mask > 127
    if not np.any(mask_bool):
        return 1.0

    # Compute full SSIM map for each channel
    # win_size should be smaller than image size, usually 7 or 11
    # win_size = min(7, min(pred.shape[0], pred.shape[1]))
    
    score, ssim_map = ssim(
        target, 
        pred, 
        channel_axis=-1, 
        data_range=255, 
        full=True
    )
    
    # ssim_map is (H, W, 3). Average across channels to get (H, W)
    ssim_map_gray = np.mean(ssim_map, axis=-1)
    
    return float(np.mean(ssim_map_gray[mask_bool]))


def evaluate(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """
    Compute reconstruction metrics within the mask. Returns metrics dict.
    """
    return {
        "psnr": masked_psnr(pred, target, mask),
        "ssim": masked_ssim(pred, target, mask)
    }

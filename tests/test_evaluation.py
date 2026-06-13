import numpy as np
from src.evaluation.metrics import masked_psnr, masked_ssim, evaluate


def test_psnr_perfect_match():
    clean = np.ones((10, 10, 3), dtype=np.uint8) * 100
    pred = np.ones((10, 10, 3), dtype=np.uint8) * 100
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 255  # 3x3 region
    
    psnr = masked_psnr(pred, clean, mask)
    assert psnr == float('inf')


def test_ssim_perfect_match():
    clean = np.ones((10, 10, 3), dtype=np.uint8) * 100
    pred = np.ones((10, 10, 3), dtype=np.uint8) * 100
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    
    ssim = masked_ssim(pred, clean, mask)
    assert np.isclose(ssim, 1.0)


def test_evaluate_dict():
    clean = np.ones((10, 10, 3), dtype=np.uint8) * 100
    pred = np.ones((10, 10, 3), dtype=np.uint8) * 100
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 0] = 255
    
    metrics = evaluate(pred, clean, mask)
    assert "psnr" in metrics
    assert "ssim" in metrics
    assert metrics["psnr"] == float('inf')
    assert np.isclose(metrics["ssim"], 1.0)

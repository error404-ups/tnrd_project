"""
utils.py

Evaluation metrics: PSNR and SSIM
"""

import torch
import torch.nn.functional as F


def psnr(pred, target):
    """Peak Signal-to-Noise Ratio (dB). Higher is better."""
    mse = torch.mean((pred - target) ** 2)
    return 10 * torch.log10(1.0 / (mse + 1e-8))


def ssim(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Structural Similarity Index (SSIM). Range [0, 1], higher is better.
    Works on single-channel [B, 1, H, W] tensors.
    """
    # Gaussian kernel
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g /= g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0)          # [1,1,ws,ws]

    pad = window_size // 2

    mu1  = F.conv2d(pred,   kernel, padding=pad, groups=1)
    mu2  = F.conv2d(target, kernel, padding=pad, groups=1)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred   * pred,   kernel, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad) - mu2_sq
    sigma12   = F.conv2d(pred   * target, kernel, padding=pad) - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12   + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return torch.mean(numerator / denominator)
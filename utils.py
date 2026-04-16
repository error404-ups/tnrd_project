"""
utils.py

Contains evaluation metrics
"""

import torch


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 10 * torch.log10(1.0 / (mse + 1e-8))
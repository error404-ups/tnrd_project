"""
dataset.py

- Loads grayscale images
- Normalizes to [0, 1]
- Adds stable multiplicative Gamma noise (L=5)
"""

import os
import cv2
import torch
from torch.utils.data import Dataset


def add_gamma_noise(img, L=5):
    """
    Multiplicative Gamma noise.
    L=5 → moderate noise, stable training.
    """
    gamma_dist = torch.distributions.Gamma(
        torch.tensor(float(L)),
        torch.tensor(1.0 / L),
    )
    noise = gamma_dist.sample(img.shape)
    return torch.clamp(img * noise, 0.0, 1.0)


class BSDDataset(Dataset):
    EXTENSIONS = (".png", ".jpg", ".jpeg")

    def __init__(self, folder):
        self.paths = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(self.EXTENSIONS)
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in: {folder}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not read image: {self.paths[idx]}")

        img = cv2.resize(img, (180, 180))

        # Normalize to [0, 1]
        img = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)

        noisy = add_gamma_noise(img)
        return noisy, img
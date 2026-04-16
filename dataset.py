"""
dataset.py

- Loads grayscale images
- Normalizes to [0,1]
- Adds stable multiplicative Gamma noise
"""

import os
import cv2
import torch
from torch.utils.data import Dataset


def add_gamma_noise(img, L=5):
    """
    CHANGE: Increased L from 1 → 5

    WHY:
    - L=1 = very strong noise → unstable training
    - L=5 = moderate noise → easier learning
    """
    gamma_dist = torch.distributions.Gamma(L, 1.0 / L)
    noise = gamma_dist.sample(img.shape)

    noisy = img * noise
    return torch.clamp(noisy, 0.0, 1.0)


class BSDDataset(Dataset):
    def __init__(self, folder):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (180, 180))

        # IMPORTANT: normalization to [0,1]
        img = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)

        noisy = add_gamma_noise(img)

        return noisy, img
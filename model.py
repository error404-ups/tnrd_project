"""
model.py

Balanced TNRD:
- Stable but not overly restricted
- Improved learning behavior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFInfluence(nn.Module):
    def __init__(self, num_basis=31, gamma=0.5):
        super().__init__()

        self.mu = nn.Parameter(torch.linspace(-1, 1, num_basis))
        self.gamma = gamma

        self.weights = nn.Parameter(torch.randn(num_basis) * 0.1)

    def forward(self, x):
        x_exp = x.unsqueeze(-1)
        diff = x_exp - self.mu
        rbf = torch.exp(-(diff ** 2) / (2 * self.gamma ** 2))
        return torch.sum(self.weights * rbf, dim=-1)


class TNRDStage(nn.Module):
    def __init__(self, num_filters=24):
        super().__init__()

        # CHANGE: slightly larger init (0.05 → 0.08)
        # WHY: earlier was too small → weak learning
        self.filters = nn.Parameter(
            torch.randn(num_filters, 1, 5, 5) * 0.08
        )

        self.influences = nn.ModuleList([
            RBFInfluence() for _ in range(num_filters)
        ])

        # CHANGE: λ increased
        # WHY: stronger pull toward clean image
        self.lambda_param = nn.Parameter(torch.tensor(0.15))

    def forward(self, u, f):
        # CHANGE: less aggressive clamp
        # WHY: preserve gradients
        u = torch.clamp(u, 1e-4, 1.0)
        f = torch.clamp(f, 1e-4, 1.0)

        diffusion = 0

        for i in range(len(self.filters)):
            k = self.filters[i:i+1]

            conv = F.conv2d(u, k, padding=2)

            phi = self.influences[i](conv)

            u_sigma = F.avg_pool2d(u, kernel_size=3, stride=1, padding=1)

            M = torch.mean(u_sigma) + 1e-3
            scaled_phi = (u_sigma / M) * phi

            kT = torch.flip(k, [2, 3])
            conv_T = F.conv2d(scaled_phi, kT, padding=2)

            diffusion += conv_T

        # CHANGE: stronger but safe reaction
        # OLD: u + 0.1 (too weak)
        # NEW: u + 0.05 → sharper gradient
        reaction = self.lambda_param * (u - f) / (u + 0.05)

        u_next = u - diffusion - reaction

        return torch.clamp(u_next, 0.0, 1.0)


class TNRD(nn.Module):
    def __init__(self, T=5):
        super().__init__()
        self.stages = nn.ModuleList([TNRDStage() for _ in range(T)])

    def forward(self, f):
        u = f.clone()
        outputs = []

        for stage in self.stages:
            u = stage(u, f)
            outputs.append(u)

        return u, outputs
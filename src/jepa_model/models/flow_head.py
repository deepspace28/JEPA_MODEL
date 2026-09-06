from __future__ import annotations

import torch
import torch.nn as nn


class FlowHead(nn.Module):
    """Predict dense optical flow (u, v) from token grid features."""

    def __init__(self, dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
        )

    def forward(self, feat_grid: torch.Tensor) -> torch.Tensor:
        # feat_grid: [B, D, H, W] -> flow: [B, 2, H, W]
        return self.net(feat_grid)

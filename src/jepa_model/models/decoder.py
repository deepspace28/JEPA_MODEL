import torch
import torch.nn as nn


class PatchDecoder(nn.Module):
    def __init__(self, dim: int = 768, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, patch_size * patch_size * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, t, p, _d = z.shape
        out = self.net(z)
        return out.view(b, t, p, 3, self.patch_size, self.patch_size)

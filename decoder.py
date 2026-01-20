import torch.nn as nn
import torch

class PatchDecoder(nn.Module):
    def __init__(self, dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, patch_size * patch_size * 3),
        )

    def forward(self, z):
        # z: (B, T, P, D)
        B, T, P, D = z.shape
        out = self.net(z)  # (B, T, P, 3*patch*patch)
        out = out.view(B, T, P, 3, self.patch_size, self.patch_size)
        return out

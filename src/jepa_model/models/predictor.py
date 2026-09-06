import torch
import torch.nn as nn


class TransformerPredictor(nn.Module):
    def __init__(self, dim: int = 768, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, batch_first=True)
        self.net = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, t, p, d = z.shape
        z = z.reshape(b * t, p, d)
        out = self.net(z)
        return out.reshape(b, t, p, d)

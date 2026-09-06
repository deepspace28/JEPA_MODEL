import torch
import torch.nn as nn
from torchvision.models import vit_b_16


class ViTEncoder(nn.Module):
    def __init__(self, image_size: int = 224):
        super().__init__()
        self.vit = vit_b_16(weights=None)
        self.vit.image_size = image_size
        self.vit.heads = nn.Identity()
        self._tokens = None

        def hook(_module, _inputs, output):
            self._tokens = output

        self.vit.encoder.register_forward_hook(hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        _ = self.vit(x)
        tokens = self._tokens[:, 1:]
        return tokens.view(b, t, tokens.shape[1], tokens.shape[2])

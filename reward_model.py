import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)

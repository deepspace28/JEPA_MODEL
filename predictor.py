import torch.nn as nn
import torch

class ActionPredictor(nn.Module):
    def __init__(self, dim=768, n_actions=18):
        super().__init__()
        self.embed = nn.Embedding(n_actions, dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, z, a):
        """
        z: (B, P, D)
        a: (B,)
        """
        a = self.embed(a)          
        a = a.unsqueeze(1)        
        a = a.expand_as(z)        

        z = torch.cat([z, a], dim=-1)  
        return self.net(z)


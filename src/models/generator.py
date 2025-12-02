# Dummy

import torch
import torch.nn as nn

class DummyGenerator(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 3 * 32 * 32)

    def forward(self, z, labels=None):
        x = self.fc(z)
        x = x.view(-1, 3, 32, 32)
        return x
    
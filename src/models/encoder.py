# Dummy

import torch
import torch.nn as nn

class DummyEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(3 * 32 * 32, latent_dim)
        self.fc_logvar = nn.Linear(3 * 32 * 32, latent_dim)

    def forward(self, x, labels=None):
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
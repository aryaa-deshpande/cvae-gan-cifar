# Dummy

import torch
import torch.nn as nn

class DummyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 32 * 32, 1)

    def forward(self, x, labels=None):
        x = self.flatten(x)
        return self.fc(x)
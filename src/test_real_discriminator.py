import torch
from models.discriminator import Discriminator

disc = Discriminator(num_classes=100)

x = torch.randn(4, 3, 32, 32)
labels = torch.randint(0, 100, (4,))

out = disc(x, labels)
print("Output shape:", out.shape)
print("Scores:", out)
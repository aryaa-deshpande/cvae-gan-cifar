import torch
from models.generator import Generator

BATCH = 4
LATENT = 128

z = torch.randn(BATCH, LATENT)
labels = torch.randint(0, 100, (BATCH,))

gen = Generator(latent_dim=LATENT)
fake = gen(z, labels)

print("Fake image shape:", fake.shape)
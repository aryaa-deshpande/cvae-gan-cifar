import torch
from models.encoder import DummyEncoder
from models.generator import DummyGenerator
from models.discriminator import DummyDiscriminator

x = torch.randn(4, 3, 32, 32)
labels = torch.randint(0, 100, (4,))

enc = DummyEncoder()
gen = DummyGenerator()
disc = DummyDiscriminator()

mu, logvar = enc(x, labels)
print("Mu shape:", mu.shape, "Logvar shape:", logvar.shape)

z = mu  # fake latent
fake_img = gen(z, labels)
print("Fake image shape:", fake_img.shape)

disc_out = disc(fake_img, labels)
print("Disc output shape:", disc_out.shape)
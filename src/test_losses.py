import torch
from utils.losses import kl_divergence, reconstruction_loss, gan_discriminator_loss, gan_generator_loss

B, latent_dim = 4, 128

mu = torch.zeros(B, latent_dim)
logvar = torch.zeros(B, latent_dim)

kl = kl_divergence(mu, logvar)
print("KL (should be ~0):", kl.item())

x = torch.randn(B, 3, 32, 32)
recon = x + 0.1 * torch.randn_like(x)
recon_l = reconstruction_loss(recon, x, mode="mse")
print("Recon loss:", recon_l.item())

real_scores = torch.randn(B, 1)
fake_scores = torch.randn(B, 1)

d_loss = gan_discriminator_loss(real_scores, fake_scores)
g_loss = gan_generator_loss(fake_scores)

print("D loss:", d_loss.item())
print("G loss:", g_loss.item())
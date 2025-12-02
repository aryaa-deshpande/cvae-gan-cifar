# train.py
import torch
from torch import nn, optim
from tqdm import tqdm

from dataset import get_cifar100_dataloaders
from utils.device import get_device
from models.encoder import DummyEncoder
from models.generator import DummyGenerator
from models.discriminator import DummyDiscriminator


def reparameterize(mu, logvar):
    """
    Standard VAE sampling:
        z = mu + std * eps
    For now this is just to test that everything is differentiable.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def train_one_epoch(
    encoder,
    generator,
    discriminator,
    train_loader,
    optimizer_enc,
    optimizer_gen,
    optimizer_disc,
    device,
):
    encoder.train()
    generator.train()
    discriminator.train()

    for images, labels in tqdm(train_loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # ----- Forward pass through encoder -----
        mu, logvar = encoder(images, labels)
        z = reparameterize(mu, logvar)

        # ----- Generate fake images -----
        fake_images = generator(z, labels)

        # ===== 1) DISCRIMINATOR STEP =====
        optimizer_disc.zero_grad()

        # Real scores
        real_scores = discriminator(images, labels)
        # Fake scores (detach so D doesn't backprop into G/E)
        fake_scores_d = discriminator(fake_images.detach(), labels)

        # Simple "D wants real > fake"
        disc_loss = -(real_scores.mean() - fake_scores_d.mean())
        disc_loss.backward()
        optimizer_disc.step()

        # ===== 2) ENCODER + GENERATOR STEP =====
        optimizer_enc.zero_grad()
        optimizer_gen.zero_grad()

        # Reconstruction loss (fake should look like real) – dummy version
        recon_loss = nn.functional.mse_loss(fake_images, images)

        # Recompute fake scores for generator step (now through current D)
        fake_scores_g = discriminator(fake_images, labels)
        gen_loss = -fake_scores_g.mean()  # G wants fake to look real

        total_eg_loss = recon_loss + gen_loss
        total_eg_loss.backward()
        optimizer_enc.step()
        optimizer_gen.step()


def main():
    device = get_device()

    # Data
    train_loader, val_loader = get_cifar100_dataloaders(batch_size=64)

    # Models
    encoder = DummyEncoder(latent_dim=64).to(device)
    generator = DummyGenerator(latent_dim=64).to(device)
    discriminator = DummyDiscriminator().to(device)

    # Optimizers (basic, will tune later)
    optimizer_enc = optim.Adam(encoder.parameters(), lr=1e-3)
    optimizer_gen = optim.Adam(generator.parameters(), lr=1e-3)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=1e-3)

    # Just run ONE epoch as a sanity check
    print("Starting dummy training loop for 1 epoch...")
    train_one_epoch(
        encoder,
        generator,
        discriminator,
        train_loader,
        optimizer_enc,
        optimizer_gen,
        optimizer_disc,
        device,
    )
    print("Finished dummy epoch without crashing ✅")


if __name__ == "__main__":
    main()
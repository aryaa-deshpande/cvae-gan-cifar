# src/train.py

import os
from typing import Dict, List

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_cifar100_dataloaders
from utils.device import get_device
from utils.losses import (
    kl_divergence,
    reconstruction_loss,
    gan_discriminator_loss,
    gan_generator_loss,
)
from models.encoder import Encoder
from models.generator import Generator
from models.discriminator import Discriminator


# ----------------- CONFIG -----------------

LATENT_DIM = 128
NUM_CLASSES = 100

BATCH_SIZE = 64
EPOCHS = 20           # start small, you can increase later
LR = 2e-4

BETA_KL = 0.2        # weight for KL term
LAMBDA_ADV = 0.1     # weight for adversarial term in EG loss
N_DISC_STEPS = 1     # Option B: multiple D steps per G/E step

RECON_MODE = "mse"   # reconstruction loss type

SAMPLES_DIR = "samples"
PLOTS_DIR = "plots"


# ----------------- UTILS -----------------


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick:
        z = mu + std * eps, eps ~ N(0, I)
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def ensure_dirs():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def save_loss_curves(history: Dict[str, List[float]], out_path: str):
    plt.figure(figsize=(8, 5))
    for key, values in history.items():
        plt.plot(values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


@torch.no_grad()
def save_sample_grid(generator: Generator, device: torch.device, epoch: int):
    """
    Generate a small grid of samples for a few classes to visually inspect progress.
    """
    from torchvision.utils import save_image, make_grid

    generator.eval()

    # pick 4 random classes and generate 8 samples per class
    num_classes_to_show = 4
    samples_per_class = 8

    z = torch.randn(num_classes_to_show * samples_per_class, LATENT_DIM, device=device)

    # e.g., classes [0, 1, 2, 3] repeated 8 times each
    base_labels = torch.arange(0, num_classes_to_show, device=device)
    labels = base_labels.repeat_interleave(samples_per_class)

    fake_images = generator(z, labels)
    # generator outputs in [-1, 1] (due to Tanh) -> map to [0, 1] for saving
    fake_images = (fake_images + 1) / 2.0
    fake_images = torch.clamp(fake_images, 0.0, 1.0)

    grid = make_grid(fake_images, nrow=samples_per_class)
    out_path = os.path.join(SAMPLES_DIR, f"samples_epoch_{epoch:03d}.png")
    save_image(grid, out_path)
    print(f"[samples] Saved sample grid to {out_path}")


# ----------------- TRAINING -----------------


def train_one_epoch(
    encoder: Encoder,
    generator: Generator,
    discriminator: Discriminator,
    train_loader: DataLoader,
    opt_enc: optim.Optimizer,
    opt_gen: optim.Optimizer,
    opt_disc: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    encoder.train()
    generator.train()
    discriminator.train()

    running_kl = 0.0
    running_recon = 0.0
    running_d = 0.0
    running_g = 0.0
    num_batches = 0

    for images, labels in tqdm(train_loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # ----- Encode and sample -----
        mu, logvar = encoder(images, labels)
        z = reparameterize(mu, logvar)

        # ----- Generate fake images -----
        fake_images = generator(z, labels)

        # ===== 1) DISCRIMINATOR STEP (Option B: multiple D updates) =====
        for _ in range(N_DISC_STEPS):
            opt_disc.zero_grad()

            real_scores = discriminator(images, labels)          # [B]
            fake_scores = discriminator(fake_images.detach(), labels)  # [B]

            d_loss = gan_discriminator_loss(
                real_scores.view(-1, 1),
                fake_scores.view(-1, 1),
            )
            d_loss.backward()
            opt_disc.step()

        # ===== 2) ENCODER + GENERATOR STEP (VAE + GAN) =====
        opt_enc.zero_grad()
        opt_gen.zero_grad()

        # Reconstruction loss: fake_images vs real images
        recon_l = reconstruction_loss(fake_images, images, mode=RECON_MODE)

        # KL divergence
        kl_l = kl_divergence(mu, logvar)

        # Generator adversarial loss: wants fake to be classified as real
        fake_scores_for_g = discriminator(fake_images, labels)
        g_adv_l = gan_generator_loss(fake_scores_for_g.view(-1, 1))

        total_eg_loss = recon_l + BETA_KL * kl_l + LAMBDA_ADV * g_adv_l

        total_eg_loss.backward()
        opt_enc.step()
        opt_gen.step()

        # ---- Stats ----
        running_kl += kl_l.item()
        running_recon += recon_l.item()
        running_d += d_loss.item()
        running_g += g_adv_l.item()
        num_batches += 1

    return {
        "kl": running_kl / num_batches,
        "recon": running_recon / num_batches,
        "d_loss": running_d / num_batches,
        "g_loss": running_g / num_batches,
    }


def main():
    ensure_dirs()
    device = get_device()

    # Data
    train_loader, val_loader = get_cifar100_dataloaders(batch_size=BATCH_SIZE)

    # Models
    encoder = Encoder(latent_dim=LATENT_DIM).to(device)
    generator = Generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)
    discriminator = Discriminator(num_classes=NUM_CLASSES).to(device)

    # Optimizers (classic GAN-style settings)
    opt_enc = optim.Adam(encoder.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_gen = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=LR * 0.5, betas=(0.5, 0.999))

    history = {
        "kl": [],
        "recon": [],
        "d_loss": [],
        "g_loss": [],
    }

    print("Starting real hybrid VAE–GAN training...")
    for epoch in range(1, EPOCHS + 1):
        epoch_stats = train_one_epoch(
            encoder,
            generator,
            discriminator,
            train_loader,
            opt_enc,
            opt_gen,
            opt_disc,
            device,
        )

        print(
            f"[Epoch {epoch}/{EPOCHS}] "
            f"KL: {epoch_stats['kl']:.4f} | "
            f"Recon: {epoch_stats['recon']:.4f} | "
            f"D: {epoch_stats['d_loss']:.4f} | "
            f"G: {epoch_stats['g_loss']:.4f}"
        )

        # Save stats
        for key in history:
            history[key].append(epoch_stats[key])

        # Save sample grid every epoch (you can change to every N epochs)
        save_sample_grid(generator, device, epoch)

    # Save loss curves
    loss_plot_path = os.path.join(PLOTS_DIR, "losses.png")
    save_loss_curves(history, loss_plot_path)
    print(f"[plots] Saved loss curves to {loss_plot_path}")

    # Save final model weights (optional but useful)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "history": history,
        },
        "checkpoint_final.pt",
    )
    print("[checkpoint] Saved final model checkpoint to checkpoint_final.pt")


if __name__ == "__main__":
    main()
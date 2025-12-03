# src/train_vae.py
"""
Train a conditional VAE baseline on CIFAR-100.

Uses:
  - Encoder
  - Generator
  - KL + reconstruction loss (no GAN / discriminator)

Outputs:
  - checkpoint_vae.pt
  - plots/losses_vae.png
"""

import os
from typing import Dict, List

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_cifar100_dataloaders
from utils.device import get_device
from utils.losses import kl_divergence, reconstruction_loss
from models.encoder import Encoder
from models.generator import Generator


# -------- CONFIG --------

LATENT_DIM = 128
NUM_CLASSES = 100

BATCH_SIZE = 128
EPOCHS = 20
LR = 2e-4

BETA_KL = 1.0          # VAE usually uses beta >= 1.0
RECON_MODE = "mse"

PLOTS_DIR = "plots"


# -------- UTILS --------

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick: z = mu + std * eps, eps ~ N(0, I)."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def ensure_dirs():
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


# -------- TRAINING LOOP --------

def train_one_epoch(
    encoder: Encoder,
    generator: Generator,
    train_loader: DataLoader,
    opt_enc: optim.Optimizer,
    opt_gen: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    encoder.train()
    generator.train()

    running_kl = 0.0
    running_recon = 0.0
    num_batches = 0

    for images, labels in tqdm(train_loader, desc="Train (VAE)", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        opt_enc.zero_grad()
        opt_gen.zero_grad()

        # Encode
        mu, logvar = encoder(images, labels)

        # Sample latent z
        z = reparameterize(mu, logvar)

        # Decode
        recon = generator(z, labels)

        # Losses
        recon_l = reconstruction_loss(recon, images, mode=RECON_MODE)
        kl_l = kl_divergence(mu, logvar)

        total_loss = recon_l + BETA_KL * kl_l
        total_loss.backward()

        opt_enc.step()
        opt_gen.step()

        running_kl += kl_l.item()
        running_recon += recon_l.item()
        num_batches += 1

    return {
        "kl": running_kl / num_batches,
        "recon": running_recon / num_batches,
    }


def main():
    ensure_dirs()
    device = get_device()

    # Data
    train_loader, _ = get_cifar100_dataloaders(batch_size=BATCH_SIZE)

    # Models
    encoder = Encoder(latent_dim=LATENT_DIM).to(device)
    generator = Generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)

    # Optimizers
    opt_enc = optim.Adam(encoder.parameters(), lr=LR, betas=(0.9, 0.999))
    opt_gen = optim.Adam(generator.parameters(), lr=LR, betas=(0.9, 0.999))

    history = {
        "kl": [],
        "recon": [],
    }

    print("Starting conditional VAE baseline training...")
    for epoch in range(1, EPOCHS + 1):
        stats = train_one_epoch(
            encoder,
            generator,
            train_loader,
            opt_enc,
            opt_gen,
            device,
        )

        print(
            f"[VAE Epoch {epoch}/{EPOCHS}] "
            f"KL: {stats['kl']:.4f} | "
            f"Recon: {stats['recon']:.4f}"
        )

        for key in history:
            history[key].append(stats[key])

    # Save curves
    loss_plot_path = os.path.join(PLOTS_DIR, "losses_vae.png")
    save_loss_curves(history, loss_plot_path)
    print(f"[VAE] Saved loss curves to {loss_plot_path}")

    # Save checkpoint
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "generator": generator.state_dict(),
            "history": history,
        },
        "checkpoint_vae.pt",
    )
    print("[VAE] Saved baseline checkpoint to checkpoint_vae.pt")


if __name__ == "__main__":
    main()
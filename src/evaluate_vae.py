# src/evaluate_vae.py
"""
Evaluate the conditional VAE baseline:
  - reconstructions
  - class-conditional samples
  - latent interpolation
"""

import os
import torch
from torchvision.utils import save_image, make_grid

from dataset import get_cifar100_dataloaders
from models.encoder import Encoder
from models.generator import Generator
from utils.device import get_device

LATENT_DIM = 128
NUM_CLASSES = 100

OUT_DIR = "eval_outputs_vae"
os.makedirs(OUT_DIR, exist_ok=True)


def load_vae(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    enc = Encoder(latent_dim=LATENT_DIM).to(device)
    gen = Generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)

    enc.load_state_dict(ckpt["encoder"])
    gen.load_state_dict(ckpt["generator"])

    enc.eval()
    gen.eval()
    return enc, gen


def denorm(x: torch.Tensor) -> torch.Tensor:
    # Generator outputs in [-1, 1] -> map to [0, 1] for saving
    x = (x + 1) / 2.0
    return torch.clamp(x, 0.0, 1.0)


@torch.no_grad()
def eval_reconstructions(encoder, generator, device):
    train_loader, val_loader = get_cifar100_dataloaders(batch_size=16)
    images, labels = next(iter(val_loader))

    images = images.to(device)
    labels = labels.to(device)

    mu, logvar = encoder(images, labels)
    z = mu  # deterministic recon
    recon = generator(z, labels)

    grid = make_grid(torch.cat([denorm(images), denorm(recon)], dim=0),
                     nrow=images.size(0))
    out_path = os.path.join(OUT_DIR, "reconstructions_vae.png")
    save_image(grid, out_path)
    print(f"[VAE eval] Saved reconstructions to {out_path}")

    mse = torch.mean((images - recon) ** 2).item()
    print(f"[VAE eval] Reconstruction MSE (batch): {mse:.4f}")


@torch.no_grad()
def eval_class_samples(generator, device):
    generator.eval()

    num_classes_to_show = 8
    samples_per_class = 8

    z = torch.randn(num_classes_to_show * samples_per_class,
                    LATENT_DIM, device=device)
    labels = torch.arange(0, num_classes_to_show,
                          device=device).repeat_interleave(samples_per_class)

    fake = generator(z, labels)
    grid = make_grid(denorm(fake), nrow=samples_per_class)
    out_path = os.path.join(OUT_DIR, "class_samples_0to7_vae.png")
    save_image(grid, out_path)
    print(f"[VAE eval] Saved class-conditional samples to {out_path}")


@torch.no_grad()
def eval_interpolation(encoder, generator, device):
    train_loader, val_loader = get_cifar100_dataloaders(batch_size=16)
    images, labels = next(iter(val_loader))

    images = images.to(device)
    labels = labels.to(device)

    x1, x2 = images[0:1], images[1:2]
    y1, y2 = labels[0:1], labels[1:2]

    mu1, _ = encoder(x1, y1)
    mu2, _ = encoder(x2, y2)

    num_steps = 8
    alphas = torch.linspace(0, 1, steps=num_steps).view(-1, 1).to(device)
    z_interp = (1 - alphas) * mu1 + alphas * mu2

    labels_interp = y1.repeat(num_steps)
    interp_imgs = generator(z_interp, labels_interp)

    all_imgs = torch.cat([x1, interp_imgs, x2], dim=0)
    grid = make_grid(denorm(all_imgs), nrow=num_steps + 2)
    out_path = os.path.join(OUT_DIR, "latent_interpolation_vae.png")
    save_image(grid, out_path)
    print(f"[VAE eval] Saved latent interpolation to {out_path}")


def main():
    device = get_device()
    print(f"[VAE eval] Using device: {device}")

    enc, gen = load_vae("checkpoint_vae.pt", device)

    eval_reconstructions(enc, gen, device)
    eval_class_samples(gen, device)
    eval_interpolation(enc, gen, device)


if __name__ == "__main__":
    main()
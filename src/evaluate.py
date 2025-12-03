import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from dataset import get_cifar100_dataloaders
from models.encoder import Encoder
from models.generator import Generator
from models.discriminator import Discriminator
from utils.device import get_device

LATENT_DIM = 128
NUM_CLASSES = 100

OUT_DIR = "eval_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def load_models(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    enc = Encoder(latent_dim=LATENT_DIM).to(device)
    gen = Generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)
    disc = Discriminator(num_classes=NUM_CLASSES).to(device)

    enc.load_state_dict(ckpt["encoder"])
    gen.load_state_dict(ckpt["generator"])
    disc.load_state_dict(ckpt["discriminator"])

    enc.eval()
    gen.eval()
    disc.eval()
    return enc, gen, disc


@torch.no_grad()
def eval_reconstructions(encoder, generator, device):
    train_loader, val_loader = get_cifar100_dataloaders(batch_size=16)
    images, labels = next(iter(val_loader))

    images = images.to(device)
    labels = labels.to(device)

    mu, logvar = encoder(images, labels)
    # simple deterministic recon: use mu directly
    z = mu
    recon = generator(z, labels)

    # map from [-1, 1] to [0, 1] for saving
    def denorm(x):
        x = (x + 1) / 2.0
        return torch.clamp(x, 0.0, 1.0)

    grid = make_grid(torch.cat([denorm(images), denorm(recon)], dim=0), nrow=images.size(0))
    out_path = os.path.join(OUT_DIR, "reconstructions.png")
    save_image(grid, out_path)
    print(f"[eval] Saved reconstructions to {out_path}")

    # MSE recon error
    mse = torch.mean((images - recon) ** 2).item()
    print(f"[eval] Reconstruction MSE (batch): {mse:.4f}")


@torch.no_grad()
def eval_class_samples(generator, device):
    generator.eval()

    num_classes_to_show = 8
    samples_per_class = 8

    z = torch.randn(num_classes_to_show * samples_per_class, LATENT_DIM, device=device)
    labels = torch.arange(0, num_classes_to_show, device=device).repeat_interleave(samples_per_class)

    fake = generator(z, labels)

    def denorm(x):
        x = (x + 1) / 2.0
        return torch.clamp(x, 0.0, 1.0)

    fake = denorm(fake)
    grid = make_grid(fake, nrow=samples_per_class)
    out_path = os.path.join(OUT_DIR, "class_samples_0to7.png")
    save_image(grid, out_path)
    print(f"[eval] Saved class-conditional samples to {out_path}")


@torch.no_grad()
def eval_interpolation(encoder, generator, device):
    train_loader, val_loader = get_cifar100_dataloaders(batch_size=16)
    images, labels = next(iter(val_loader))

    images = images.to(device)
    labels = labels.to(device)

    # take first two images in the batch
    x1, x2 = images[0:1], images[1:2]
    y1, y2 = labels[0:1], labels[1:2]

    mu1, logvar1 = encoder(x1, y1)
    mu2, logvar2 = encoder(x2, y2)

    # interpolate between mu1 and mu2
    num_steps = 8
    alphas = torch.linspace(0, 1, steps=num_steps).view(-1, 1).to(device)
    z_interp = (1 - alphas) * mu1 + alphas * mu2
    # fix label to y1 for simplicity
    labels_interp = y1.repeat(num_steps)

    interp_imgs = generator(z_interp, labels_interp)

    def denorm(x):
        x = (x + 1) / 2.0
        return torch.clamp(x, 0.0, 1.0)

    # also include endpoints for context
    all_imgs = torch.cat([x1, interp_imgs, x2], dim=0)
    all_imgs = denorm(all_imgs)

    grid = make_grid(all_imgs, nrow=num_steps + 2)
    out_path = os.path.join(OUT_DIR, "latent_interpolation.png")
    save_image(grid, out_path)
    print(f"[eval] Saved latent interpolation to {out_path}")


def main():
    device = get_device()
    print(f"[eval] Using device: {device}")

    enc, gen, disc = load_models("checkpoint_final.pt", device)

    eval_reconstructions(enc, gen, device)
    eval_class_samples(gen, device)
    eval_interpolation(enc, gen, device)


if __name__ == "__main__":
    main()
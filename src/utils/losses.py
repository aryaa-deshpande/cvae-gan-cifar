import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- VAE losses ----------

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between N(mu, sigma^2) and N(0, 1).
    Formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    Returns mean KL over batch.
    """
    # shape: [B, latent_dim]
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # sum over latent dim, mean over batch
    return kl.sum(dim=1).mean()


def reconstruction_loss(recon_x: torch.Tensor, x: torch.Tensor, mode: str = "mse") -> torch.Tensor:
    """
    Reconstruction loss between reconstructed images and originals.
    - mode="mse": mean squared error (good default for CIFAR)
    - mode="bce": binary cross entropy (if using sigmoid outputs)
    """
    if mode == "mse":
        return F.mse_loss(recon_x, x)
    elif mode == "bce":
        # assume recon_x in (0,1), x in (0,1)
        return F.binary_cross_entropy(recon_x, x)
    else:
        raise ValueError(f"Unknown reconstruction mode: {mode}")


# ---------- GAN losses (standard BCE-with-logits) ----------

_bce_logits = nn.BCEWithLogitsLoss()


def gan_discriminator_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Discriminator loss:
        - real_scores should be classified as 1
        - fake_scores should be classified as 0
    Uses BCEWithLogitsLoss on raw scores.
    """
    real_targets = torch.ones_like(real_scores)
    fake_targets = torch.zeros_like(fake_scores)

    loss_real = _bce_logits(real_scores, real_targets)
    loss_fake = _bce_logits(fake_scores, fake_targets)

    return loss_real + loss_fake


def gan_generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Generator loss:
        - wants fake_scores to be classified as 1 (fool D)
    """
    targets = torch.ones_like(fake_scores)
    return _bce_logits(fake_scores, targets)
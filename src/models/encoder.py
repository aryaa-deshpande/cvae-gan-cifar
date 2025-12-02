import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()

        # Convolutional feature extractor for 32x32 CIFAR images
        self.conv_layers = nn.Sequential(
            # 3 x 32 x 32 -> 32 x 16 x 16
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 32 x 16 x 16 -> 64 x 8 x 8
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64 x 8 x 8 -> 128 x 4 x 4
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.flatten_dim = 128 * 4 * 4  # after 3 downsamples

        # Latent mean and log-variance
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x, labels=None):
        """
        x: [batch, 3, 32, 32]
        returns:
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
        """
        out = self.conv_layers(x)          # [B, 128, 4, 4]
        out = out.view(out.size(0), -1)    # [B, 2048]

        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar
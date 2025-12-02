import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_classes=100, embed_dim=256):
        super().__init__()

        # CNN feature extractor
        self.conv = nn.Sequential(
            # 3 x 32 x 32 -> 64 x 16 x 16
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 x 16 x 16 -> 128 x 8 x 8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 x 8 x 8 -> 256 x 4 x 4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.feature_dim = 256 * 4 * 4

        # Map flattened features -> feature vector
        self.fc = nn.Linear(self.feature_dim, embed_dim)

        # Classification head (real/fake)
        self.real_head = nn.Linear(embed_dim, 1)

        # Label embedding for projection
        self.label_emb = nn.Embedding(num_classes, embed_dim)

    def forward(self, x, labels):
        """
        x: [B, 3, 32, 32]
        labels: [B]
        returns: [B, 1] score
        """

        # CNN feature extractor
        features = self.conv(x)                # [B, 256, 4, 4]
        features = features.view(features.size(0), -1)  # [B, 256*4*4]

        # Embed to feature dimension
        features = self.fc(features)           # [B, embed_dim]

        # Real/fake score
        realism_score = self.real_head(features).squeeze(1)  # [B]

        # Projection score (label alignment)
        label_vec = self.label_emb(labels)     # [B, embed_dim]
        proj_score = torch.sum(features * label_vec, dim=1)  # [B]

        # Final score = realism + projection
        return realism_score + proj_score
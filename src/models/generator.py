import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=128, num_classes=100, embed_dim=50):
        super().__init__()

        # Label embedding: turns class id into a vector
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        # Total input to the first dense layer
        input_dim = latent_dim + embed_dim

        # First: map (z + label_emb) to a feature map 256 x 4 x 4
        self.fc = nn.Linear(input_dim, 256 * 4 * 4)

        # Main upsampling pipeline
        self.net = nn.Sequential(
            # 256 x 4 x 4 → 128 x 8 x 8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 128 x 8 x 8 → 64 x 16 x 16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64 x 16 x 16 → 3 x 32 x 32
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),   # output in [-1, 1]
        )

    def forward(self, z, labels):
        # Embed labels
        label_vec = self.label_emb(labels)    # [B, embed_dim]

        # Combine latent + label embedding
        x = torch.cat([z, label_vec], dim=1)  # [B, latent_dim+embed_dim]

        # Dense layer to seed the feature map
        x = self.fc(x)                        # [B, 256*4*4]

        # Reshape to 256 x 4 x 4
        x = x.view(x.size(0), 256, 4, 4)

        # Upsample to 32x32
        x = self.net(x)

        return x
import torch
from models.encoder import Encoder

def main():
    x = torch.randn(4, 3, 32, 32)  # fake CIFAR batch
    model = Encoder(latent_dim=128)
    mu, logvar = model(x)

    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)

if __name__ == "__main__":
    main()
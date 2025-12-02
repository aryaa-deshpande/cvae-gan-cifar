from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_cifar100_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    validation_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders for CIFAR-100.

    - Downloads CIFAR-100 if not present
    - Applies standard normalization
    - Splits the original train set into train/val subsets
    """
    # Standard CIFAR-100 normalization values (approximate)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )

    # Full training dataset
    full_train = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    num_train = len(full_train)
    indices = list(range(num_train))
    split = int(validation_split * num_train)

    if shuffle:
        # reproducible split
        generator = torch.Generator().manual_seed(seed)
        permuted = torch.randperm(num_train, generator=generator).tolist()
        indices = permuted

    val_indices = indices[:split]
    train_indices = indices[split:]

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(full_train, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
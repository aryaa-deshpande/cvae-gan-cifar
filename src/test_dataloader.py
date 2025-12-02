from dataset import get_cifar100_dataloaders

def main():
    train_loader, val_loader = get_cifar100_dataloaders(batch_size=32)

    images, labels = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    # Optional: show class of first item
    print("First label:", labels[0].item())

if __name__ == "__main__":
    main()
"""
Dataset loading.
Supports CIFAR-10 and CelebA.
"""
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloader(
    dataset_name: str = "celeba",
    data_root: str = "./data_cache",
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    Build a training DataLoader.

    Args:
        dataset_name: "cifar10" or "celeba"
        data_root: where to store / find the data
        image_size: target image size
        batch_size: batch size
        num_workers: data loading workers

    Returns:
        DataLoader for training
    """
    # Output range [-1, 1] to match the DDPM paper
    if dataset_name.lower() == "celeba":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),
        ])
    else:
        # CIFAR-10
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),
        ])

    if dataset_name.lower() == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transform,
        )
        print(f"Loaded dataset: CIFAR-10")
    elif dataset_name.lower() == "celeba":
        dataset = datasets.CelebA(
            root=data_root,
            split='train',
            download=True,
            transform=transform,
        )
        print(f"Loaded dataset: CelebA")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Num batches: {len(dataloader)}")
    print(f"  - Image size: {image_size}x{image_size}")

    return dataloader

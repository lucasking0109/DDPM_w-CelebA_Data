"""
資料集載入模組
支援 CIFAR-10 和 CelebA 資料集
共用 GAN model 的 data_cache
"""
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloader(
    dataset_name: str = "celeba",
    data_root: str = "/Users/lucasking/Desktop/GAN model/data_cache",
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    取得資料載入器

    Args:
        dataset_name: 資料集名稱 ("cifar10" 或 "celeba")
        data_root: 資料儲存路徑 (預設共用 GAN 的 data_cache)
        image_size: 目標圖片大小
        batch_size: 批次大小
        num_workers: 資料載入的工作執行緒數

    Returns:
        DataLoader: 訓練資料載入器
    """
    # 定義圖片轉換
    # 輸出範圍 [-1, 1]，與 DDPM 論文一致
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

    # 載入資料集
    if dataset_name.lower() == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transform,
        )
        print(f"載入資料集: CIFAR-10")
    elif dataset_name.lower() == "celeba":
        dataset = datasets.CelebA(
            root=data_root,
            split='train',
            download=True,  # 自動下載
            transform=transform,
        )
        print(f"載入資料集: CelebA")
    else:
        raise ValueError(f"不支援的資料集: {dataset_name}")

    # 建立 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"  - 總樣本數: {len(dataset)}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 批次數量: {len(dataloader)}")
    print(f"  - 圖片大小: {image_size}x{image_size}")

    return dataloader

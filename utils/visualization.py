"""
視覺化工具
生成樣本圖片和損失曲線
"""
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from pathlib import Path


def save_samples(images, filepath, nrow=8):
    """
    儲存生成的樣本圖片

    Args:
        images: 生成的圖片張量 (batch, channels, height, width)，範圍 [-1, 1]
        filepath: 儲存路徑
        nrow: 每行顯示的圖片數
    """
    # 將範圍從 [-1, 1] 轉換為 [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)

    # 製作網格
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)

    # 轉換為 numpy
    grid = grid.cpu().permute(1, 2, 0).numpy()

    # 儲存圖片
    plt.figure(figsize=(12, 12))
    plt.imshow(grid)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"樣本圖片已儲存: {filepath}")


def plot_losses(losses, filepath):
    """
    繪製訓練損失曲線

    Args:
        losses: 損失值列表
        filepath: 儲存路徑
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Loss", color="blue", alpha=0.7)

    # 加入移動平均線
    if len(losses) > 100:
        window = min(100, len(losses) // 10)
        moving_avg = []
        for i in range(len(losses)):
            start = max(0, i - window)
            moving_avg.append(sum(losses[start:i+1]) / (i - start + 1))
        plt.plot(moving_avg, label=f"Moving Avg ({window})", color="red", linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE)")
    plt.title("DDPM Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"損失曲線已儲存: {filepath}")


def visualize_diffusion_process(diffusion, model, x_0, save_dir):
    """
    視覺化擴散過程 (前向加噪和反向去噪)

    Args:
        diffusion: GaussianDiffusion 實例
        model: U-Net 模型
        x_0: 原始圖片 (1, channels, height, width)
        save_dir: 儲存目錄
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    x_0 = x_0.to(device)

    # 前向過程: 顯示不同時間步的加噪結果
    timesteps = [0, 100, 250, 500, 750, 999]
    forward_images = [x_0]

    for t in timesteps[1:]:
        t_tensor = torch.tensor([t], device=device)
        x_t, _ = diffusion.q_sample(x_0, t_tensor)
        forward_images.append(x_t)

    forward_grid = torch.cat(forward_images, dim=0)
    save_samples(forward_grid, save_dir / "forward_process.png", nrow=len(timesteps))

    # 反向過程: 從噪聲生成
    model.eval()
    with torch.no_grad():
        generated, intermediates = diffusion.p_sample_loop(
            model,
            shape=(1, x_0.shape[1], x_0.shape[2], x_0.shape[3]),
            return_intermediates=True
        )

    # 儲存中間結果
    if intermediates:
        reverse_images = intermediates + [generated]
        reverse_grid = torch.cat(reverse_images, dim=0)
        save_samples(reverse_grid, save_dir / "reverse_process.png", nrow=len(reverse_images))

    print(f"擴散過程視覺化已儲存到: {save_dir}")

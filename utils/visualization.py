"""
Visualization helpers: sample grids and loss curves.
"""
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from pathlib import Path


def save_samples(images, filepath, nrow=8):
    """
    Save a grid of generated images.

    Args:
        images: tensor (batch, channels, height, width) in range [-1, 1]
        filepath: save path
        nrow: images per row
    """
    # Map from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)

    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    grid = grid.cpu().permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 12))
    plt.imshow(grid)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Samples saved: {filepath}")


def plot_losses(losses, filepath):
    """
    Plot the training loss curve.

    Args:
        losses: list of loss values
        filepath: save path
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Loss", color="blue", alpha=0.7)

    # Moving average overlay
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

    print(f"Loss curve saved: {filepath}")


def visualize_diffusion_process(diffusion, model, x_0, save_dir):
    """
    Visualize the forward (noising) and reverse (denoising) processes.

    Args:
        diffusion: GaussianDiffusion instance
        model: U-Net model
        x_0: original image (1, channels, height, width)
        save_dir: output directory
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    x_0 = x_0.to(device)

    # Forward: show noised images at several timesteps
    timesteps = [0, 100, 250, 500, 750, 999]
    forward_images = [x_0]

    for t in timesteps[1:]:
        t_tensor = torch.tensor([t], device=device)
        x_t, _ = diffusion.q_sample(x_0, t_tensor)
        forward_images.append(x_t)

    forward_grid = torch.cat(forward_images, dim=0)
    save_samples(forward_grid, save_dir / "forward_process.png", nrow=len(timesteps))

    # Reverse: generate from noise
    model.eval()
    with torch.no_grad():
        generated, intermediates = diffusion.p_sample_loop(
            model,
            shape=(1, x_0.shape[1], x_0.shape[2], x_0.shape[3]),
            return_intermediates=True
        )

    if intermediates:
        reverse_images = intermediates + [generated]
        reverse_grid = torch.cat(reverse_images, dim=0)
        save_samples(reverse_grid, save_dir / "reverse_process.png", nrow=len(reverse_images))

    print(f"Diffusion visualization saved to: {save_dir}")

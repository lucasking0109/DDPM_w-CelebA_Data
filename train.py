"""
DDPM training script.
"""
import argparse
import copy
import torch
from tqdm import tqdm

from config import Config
from models import UNet, GaussianDiffusion
from data import get_dataloader
from utils import save_checkpoint, load_checkpoint, save_samples, plot_losses


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM")
    parser.add_argument("--epochs", type=int, default=None, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="batch size")
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="resume from checkpoint")
    return parser.parse_args()


def update_ema(ema_model, model, decay=0.9999):
    """Update EMA model parameters in place"""
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def train():
    args = parse_args()

    Config.init()

    # CLI overrides
    if args.epochs is not None:
        Config.EPOCHS = args.epochs
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        Config.LEARNING_RATE = args.lr

    device = Config.DEVICE

    torch.manual_seed(Config.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(Config.SEED)

    print("\nLoading dataset...")
    dataloader = get_dataloader(
        dataset_name=Config.DATASET,
        data_root=str(Config.DATA_CACHE_DIR),
        image_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    print("\nBuilding model...")
    model = UNet(
        in_channels=Config.CHANNELS,
        out_channels=Config.CHANNELS,
        model_channels=Config.MODEL_CHANNELS,
        channel_mult=Config.CHANNEL_MULT,
        attention_resolutions=Config.ATTENTION_RESOLUTIONS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        dropout=Config.DROPOUT,
        image_size=Config.IMAGE_SIZE,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # EMA model
    if Config.USE_EMA:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
    else:
        ema_model = None

    # Diffusion process
    diffusion = GaussianDiffusion(
        timesteps=Config.TIMESTEPS,
        beta_start=Config.BETA_START,
        beta_end=Config.BETA_END,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(
            args.resume,
            model,
            optimizer,
            ema_model,
            device,
        )
        start_epoch += 1

    all_losses = []
    best_loss = float("inf")

    # Fixed noise for reproducible sample images
    fixed_noise = torch.randn(
        Config.NUM_SAMPLES,
        Config.CHANNELS,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
        device=device,
    )

    print(f"\nStarting training...")
    print(f"  - Epochs: {Config.EPOCHS}")
    print(f"  - Batch size: {Config.BATCH_SIZE}")
    print(f"  - Learning rate: {Config.LEARNING_RATE}")
    print(f"  - Timesteps: {Config.TIMESTEPS}")
    print(f"  - EMA: {'on' if Config.USE_EMA else 'off'}")
    print()

    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        epoch_losses = []

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{Config.EPOCHS}",
            leave=True,
        )

        for batch_idx, batch in enumerate(pbar):
            # Drop labels if present
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)
            batch_size = images.shape[0]

            # Random timestep per sample
            t = torch.randint(0, Config.TIMESTEPS, (batch_size,), device=device)

            loss = diffusion.p_losses(model, images, t)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if Config.USE_EMA:
                update_ema(ema_model, model, Config.EMA_DECAY)

            loss_value = loss.item()
            epoch_losses.append(loss_value)
            all_losses.append(loss_value)

            pbar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "avg": f"{sum(epoch_losses)/len(epoch_losses):.4f}",
            })

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} done - avg loss: {avg_loss:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                Config.CHECKPOINT_DIR / "best_model.pth",
                ema_model,
            )

        # Periodic checkpoint
        if (epoch + 1) % Config.SAVE_INTERVAL == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                Config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1:04d}.pth",
                ema_model,
            )

        # Periodic sample generation
        if (epoch + 1) % Config.SAMPLE_INTERVAL == 0:
            print("Generating samples...")
            sample_model = ema_model if Config.USE_EMA else model
            sample_model.eval()

            with torch.no_grad():
                samples = diffusion.sample(
                    sample_model,
                    num_samples=Config.NUM_SAMPLES,
                    image_size=Config.IMAGE_SIZE,
                    channels=Config.CHANNELS,
                )

            save_samples(
                samples,
                Config.SAMPLE_DIR / f"epoch_{epoch+1:04d}.png",
            )

            plot_losses(
                all_losses,
                Config.OUTPUT_DIR / "training_loss.png",
            )

    print("\nTraining complete!")

    # Final model
    save_checkpoint(
        model, optimizer, Config.EPOCHS - 1, avg_loss,
        Config.CHECKPOINT_DIR / "final_model.pth",
        ema_model,
    )

    print("Generating final samples...")
    sample_model = ema_model if Config.USE_EMA else model
    sample_model.eval()

    with torch.no_grad():
        samples = diffusion.sample(
            sample_model,
            num_samples=Config.NUM_SAMPLES,
            image_size=Config.IMAGE_SIZE,
            channels=Config.CHANNELS,
        )

    save_samples(samples, Config.SAMPLE_DIR / "final_samples.png")
    plot_losses(all_losses, Config.OUTPUT_DIR / "training_loss.png")

    print(f"\nResults saved to: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    train()

"""
Save and load model checkpoints.
"""
import torch
from pathlib import Path


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    filepath,
    ema_model=None,
):
    """
    Save a model checkpoint.

    Args:
        model: U-Net model
        optimizer: optimizer
        epoch: current epoch
        loss: current loss
        filepath: save path
        ema_model: optional EMA model
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }

    if ema_model is not None:
        checkpoint["ema_model_state_dict"] = ema_model.state_dict()

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath,
    model,
    optimizer=None,
    ema_model=None,
    device="cpu",
):
    """
    Load a model checkpoint.

    Args:
        filepath: checkpoint path
        model: U-Net model
        optimizer: optional optimizer
        ema_model: optional EMA model
        device: device to load onto

    Returns:
        epoch, loss
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if ema_model is not None and "ema_model_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", 0.0)

    print(f"Checkpoint loaded: {filepath}")
    print(f"  - Epoch: {epoch}")
    print(f"  - Loss: {loss:.6f}")

    return epoch, loss

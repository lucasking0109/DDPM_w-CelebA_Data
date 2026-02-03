"""
模型檢查點儲存與載入
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
    儲存模型檢查點

    Args:
        model: U-Net 模型
        optimizer: 優化器
        epoch: 當前 epoch
        loss: 當前損失
        filepath: 儲存路徑
        ema_model: EMA 模型 (可選)
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
    print(f"檢查點已儲存: {filepath}")


def load_checkpoint(
    filepath,
    model,
    optimizer=None,
    ema_model=None,
    device="cpu",
):
    """
    載入模型檢查點

    Args:
        filepath: 檢查點路徑
        model: U-Net 模型
        optimizer: 優化器 (可選)
        ema_model: EMA 模型 (可選)
        device: 載入設備

    Returns:
        epoch: 訓練到的 epoch
        loss: 最後的損失值
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if ema_model is not None and "ema_model_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", 0.0)

    print(f"檢查點已載入: {filepath}")
    print(f"  - Epoch: {epoch}")
    print(f"  - Loss: {loss:.6f}")

    return epoch, loss

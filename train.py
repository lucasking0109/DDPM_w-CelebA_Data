"""
DDPM 訓練腳本
基於原始 DDPM 論文的實現
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
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description="訓練 DDPM 模型")
    parser.add_argument("--epochs", type=int, default=None, help="訓練 epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="批次大小")
    parser.add_argument("--lr", type=float, default=None, help="學習率")
    parser.add_argument("--resume", type=str, default=None, help="從檢查點繼續訓練")
    return parser.parse_args()


def update_ema(ema_model, model, decay=0.9999):
    """更新 EMA 模型的參數"""
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def train():
    """主訓練函數"""
    args = parse_args()

    # 初始化配置
    Config.init()

    # 覆蓋命令列參數
    if args.epochs is not None:
        Config.EPOCHS = args.epochs
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        Config.LEARNING_RATE = args.lr

    device = Config.DEVICE

    # 設定隨機種子
    torch.manual_seed(Config.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(Config.SEED)

    # 載入資料
    print("\n載入資料集...")
    dataloader = get_dataloader(
        dataset_name=Config.DATASET,
        data_root=str(Config.DATA_CACHE_DIR),
        image_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 建立模型
    print("\n建立模型...")
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

    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")

    # 建立 EMA 模型
    if Config.USE_EMA:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
    else:
        ema_model = None

    # 建立擴散過程
    diffusion = GaussianDiffusion(
        timesteps=Config.TIMESTEPS,
        beta_start=Config.BETA_START,
        beta_end=Config.BETA_END,
        device=device,
    )

    # 優化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
    )

    # 載入檢查點
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(
            args.resume,
            model,
            optimizer,
            ema_model,
            device,
        )
        start_epoch += 1  # 從下一個 epoch 開始

    # 訓練記錄
    all_losses = []
    best_loss = float("inf")

    # 用於生成樣本的固定噪聲
    fixed_noise = torch.randn(
        Config.NUM_SAMPLES,
        Config.CHANNELS,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
        device=device,
    )

    print(f"\n開始訓練...")
    print(f"  - Epochs: {Config.EPOCHS}")
    print(f"  - Batch size: {Config.BATCH_SIZE}")
    print(f"  - Learning rate: {Config.LEARNING_RATE}")
    print(f"  - Timesteps: {Config.TIMESTEPS}")
    print(f"  - EMA: {'啟用' if Config.USE_EMA else '停用'}")
    print()

    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        epoch_losses = []

        # 進度條
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{Config.EPOCHS}",
            leave=True,
        )

        for batch_idx, batch in enumerate(pbar):
            # 取得圖片 (忽略標籤)
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)
            batch_size = images.shape[0]

            # 隨機選擇時間步
            t = torch.randint(0, Config.TIMESTEPS, (batch_size,), device=device)

            # 計算損失
            loss = diffusion.p_losses(model, images, t)

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪 (防止爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # 更新 EMA
            if Config.USE_EMA:
                update_ema(ema_model, model, Config.EMA_DECAY)

            # 記錄損失
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            all_losses.append(loss_value)

            # 更新進度條
            pbar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "avg": f"{sum(epoch_losses)/len(epoch_losses):.4f}",
            })

        # Epoch 結束
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} 完成 - 平均損失: {avg_loss:.6f}")

        # 儲存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                Config.CHECKPOINT_DIR / "best_model.pth",
                ema_model,
            )

        # 定期儲存檢查點
        if (epoch + 1) % Config.SAVE_INTERVAL == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                Config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1:04d}.pth",
                ema_model,
            )

        # 定期生成樣本
        if (epoch + 1) % Config.SAMPLE_INTERVAL == 0:
            print("生成樣本圖片...")
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

            # 繪製損失曲線
            plot_losses(
                all_losses,
                Config.OUTPUT_DIR / "training_loss.png",
            )

    # 訓練結束
    print("\n訓練完成！")

    # 儲存最終模型
    save_checkpoint(
        model, optimizer, Config.EPOCHS - 1, avg_loss,
        Config.CHECKPOINT_DIR / "final_model.pth",
        ema_model,
    )

    # 生成最終樣本
    print("生成最終樣本...")
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

    print(f"\n結果已儲存到: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    train()

"""
DDPM 採樣腳本
從訓練好的模型生成圖片
"""
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

from config import Config
from models import UNet, GaussianDiffusion
from utils import load_checkpoint, save_samples


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description="從 DDPM 模型生成圖片")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="模型檢查點路徑 (預設使用最佳模型)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="生成圖片數量 (預設 64)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="輸出檔案路徑",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        default=True,
        help="使用 EMA 模型 (預設啟用)",
    )
    parser.add_argument(
        "--show-process",
        action="store_true",
        help="顯示生成過程的中間結果",
    )
    return parser.parse_args()


def sample():
    """生成樣本"""
    args = parse_args()

    # 初始化配置
    Config.init()
    device = Config.DEVICE

    # 決定檢查點路徑
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # 優先使用最佳模型，否則使用最終模型
        best_path = Config.CHECKPOINT_DIR / "best_model.pth"
        final_path = Config.CHECKPOINT_DIR / "final_model.pth"

        if best_path.exists():
            checkpoint_path = best_path
        elif final_path.exists():
            checkpoint_path = final_path
        else:
            print("錯誤: 找不到模型檢查點！")
            print(f"請確認以下路徑存在:")
            print(f"  - {best_path}")
            print(f"  - {final_path}")
            return

    print(f"使用檢查點: {checkpoint_path}")

    # 建立模型
    print("\n建立模型...")
    model = UNet(
        in_channels=Config.CHANNELS,
        out_channels=Config.CHANNELS,
        model_channels=Config.MODEL_CHANNELS,
        channel_mult=Config.CHANNEL_MULT,
        attention_resolutions=Config.ATTENTION_RESOLUTIONS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        dropout=0.0,  # 採樣時關閉 dropout
        image_size=Config.IMAGE_SIZE,
    ).to(device)

    # 載入檢查點
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if args.use_ema and "ema_model_state_dict" in checkpoint:
        print("使用 EMA 模型權重")
        model.load_state_dict(checkpoint["ema_model_state_dict"])
    else:
        print("使用原始模型權重")
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # 建立擴散過程
    diffusion = GaussianDiffusion(
        timesteps=Config.TIMESTEPS,
        beta_start=Config.BETA_START,
        beta_end=Config.BETA_END,
        device=device,
    )

    print(f"\n開始生成 {args.num_samples} 張圖片...")
    print(f"  - 圖片大小: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    print(f"  - 時間步: {Config.TIMESTEPS}")
    print()

    # 生成圖片
    with torch.no_grad():
        if args.show_process:
            # 顯示生成過程
            samples, intermediates = diffusion.p_sample_loop(
                model,
                shape=(args.num_samples, Config.CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                return_intermediates=True,
            )

            # 儲存中間結果
            for i, img in enumerate(intermediates):
                step = (len(intermediates) - i - 1) * 100
                save_samples(
                    img,
                    Config.SAMPLE_DIR / f"process_step_{step:04d}.png",
                    nrow=int(args.num_samples ** 0.5),
                )
        else:
            # 直接生成最終結果
            # 使用 tqdm 顯示進度
            batch_size = args.num_samples
            x = torch.randn(
                batch_size,
                Config.CHANNELS,
                Config.IMAGE_SIZE,
                Config.IMAGE_SIZE,
                device=device,
            )

            for i in tqdm(reversed(range(Config.TIMESTEPS)), desc="採樣中", total=Config.TIMESTEPS):
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                x = diffusion.p_sample(model, x, t, i)

            samples = x

    # 決定輸出路徑
    if args.output:
        output_path = Path(args.output)
    else:
        # 找一個不存在的檔名
        idx = 1
        while True:
            output_path = Config.SAMPLE_DIR / f"generated_{idx:03d}.png"
            if not output_path.exists():
                break
            idx += 1

    # 儲存結果
    save_samples(samples, output_path, nrow=int(args.num_samples ** 0.5))

    print(f"\n生成完成！")
    print(f"結果已儲存到: {output_path}")


if __name__ == "__main__":
    sample()

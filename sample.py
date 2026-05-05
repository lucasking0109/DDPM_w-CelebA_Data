"""
DDPM sampling script.
Generate images from a trained checkpoint.
"""
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

from config import Config
from models import UNet, GaussianDiffusion
from utils import load_checkpoint, save_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from a DDPM model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint path (defaults to best_model.pth)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="number of images to generate (default 64)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output file path",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        default=True,
        help="use EMA weights (default on)",
    )
    parser.add_argument(
        "--show-process",
        action="store_true",
        help="save intermediate steps of the generation process",
    )
    return parser.parse_args()


def sample():
    args = parse_args()

    Config.init()
    device = Config.DEVICE

    # Resolve checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # Prefer best, fall back to final
        best_path = Config.CHECKPOINT_DIR / "best_model.pth"
        final_path = Config.CHECKPOINT_DIR / "final_model.pth"

        if best_path.exists():
            checkpoint_path = best_path
        elif final_path.exists():
            checkpoint_path = final_path
        else:
            print("Error: no checkpoint found.")
            print(f"Looked for:")
            print(f"  - {best_path}")
            print(f"  - {final_path}")
            return

    print(f"Using checkpoint: {checkpoint_path}")

    print("\nBuilding model...")
    model = UNet(
        in_channels=Config.CHANNELS,
        out_channels=Config.CHANNELS,
        model_channels=Config.MODEL_CHANNELS,
        channel_mult=Config.CHANNEL_MULT,
        attention_resolutions=Config.ATTENTION_RESOLUTIONS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        dropout=0.0,  # disable dropout at sampling time
        image_size=Config.IMAGE_SIZE,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if args.use_ema and "ema_model_state_dict" in checkpoint:
        print("Using EMA weights")
        model.load_state_dict(checkpoint["ema_model_state_dict"])
    else:
        print("Using raw model weights")
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=Config.TIMESTEPS,
        beta_start=Config.BETA_START,
        beta_end=Config.BETA_END,
        device=device,
    )

    print(f"\nGenerating {args.num_samples} images...")
    print(f"  - Image size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    print(f"  - Timesteps: {Config.TIMESTEPS}")
    print()

    with torch.no_grad():
        if args.show_process:
            samples, intermediates = diffusion.p_sample_loop(
                model,
                shape=(args.num_samples, Config.CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                return_intermediates=True,
            )

            for i, img in enumerate(intermediates):
                step = (len(intermediates) - i - 1) * 100
                save_samples(
                    img,
                    Config.SAMPLE_DIR / f"process_step_{step:04d}.png",
                    nrow=int(args.num_samples ** 0.5),
                )
        else:
            batch_size = args.num_samples
            x = torch.randn(
                batch_size,
                Config.CHANNELS,
                Config.IMAGE_SIZE,
                Config.IMAGE_SIZE,
                device=device,
            )

            for i in tqdm(reversed(range(Config.TIMESTEPS)), desc="Sampling", total=Config.TIMESTEPS):
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                x = diffusion.p_sample(model, x, t, i)

            samples = x

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Find a free filename
        idx = 1
        while True:
            output_path = Config.SAMPLE_DIR / f"generated_{idx:03d}.png"
            if not output_path.exists():
                break
            idx += 1

    save_samples(samples, output_path, nrow=int(args.num_samples ** 0.5))

    print(f"\nDone!")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    sample()

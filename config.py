"""
DDPM training configuration.
Hyperparameters follow the original DDPM paper.
"""
import os
from pathlib import Path
import torch


class Config:
    """DDPM training config"""

    # ==================== Paths ====================
    PROJECT_ROOT = Path(__file__).parent
    # Data cache dir; override with DDPM_DATA_DIR env var
    DATA_CACHE_DIR = Path(os.environ.get("DDPM_DATA_DIR", PROJECT_ROOT / "data_cache"))
    OUTPUT_DIR = PROJECT_ROOT / "outputs_ddpm"
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    SAMPLE_DIR = OUTPUT_DIR / "samples"

    # ==================== Dataset ====================
    DATASET = "celeba"  # "cifar10" or "celeba"
    IMAGE_SIZE = 64     # original DDPM uses 64x64 or 256x256
    CHANNELS = 3        # RGB

    # ==================== Diffusion ====================
    # Original DDPM uses 1000 steps
    TIMESTEPS = 1000

    # Linear beta schedule
    BETA_START = 1e-4
    BETA_END = 0.02

    # ==================== Model ====================
    MODEL_CHANNELS = 128
    CHANNEL_MULT = (1, 2, 2, 2)
    ATTENTION_RESOLUTIONS = (16,)  # add attention at 16x16 resolution
    NUM_RES_BLOCKS = 2
    DROPOUT = 0.1

    # ==================== Training ====================
    BATCH_SIZE = 64     # paper uses 128; 64 fits typical GPUs
    EPOCHS = 100
    LEARNING_RATE = 2e-4

    # EMA (Exponential Moving Average)
    EMA_DECAY = 0.9999
    USE_EMA = True

    # ==================== Saving ====================
    SAVE_INTERVAL = 1        # save checkpoint every N epochs
    SAMPLE_INTERVAL = 5      # generate samples every N epochs
    NUM_SAMPLES = 64         # number of samples (8x8 grid)

    # ==================== Misc ====================
    NUM_WORKERS = 4
    SEED = 42

    @classmethod
    def get_device(cls):
        """Pick the best available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    DEVICE = None  # set in init()

    @classmethod
    def init(cls):
        """Initialize config and create output dirs"""
        cls.DEVICE = cls.get_device()

        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

        # Shrink batch size on CPU
        if cls.DEVICE.type == "cpu":
            cls.BATCH_SIZE = min(cls.BATCH_SIZE, 16)
            cls.NUM_WORKERS = 0
            print("Warning: training on CPU, batch size reduced")

        print(f"DDPM config initialized")
        print(f"  - Device: {cls.DEVICE}")
        print(f"  - Dataset: {cls.DATASET}")
        print(f"  - Image size: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"  - Timesteps: {cls.TIMESTEPS}")
        print(f"  - Batch size: {cls.BATCH_SIZE}")
        print(f"  - Output dir: {cls.OUTPUT_DIR}")

    @classmethod
    def get_beta_schedule(cls):
        """Return the beta schedule"""
        return torch.linspace(cls.BETA_START, cls.BETA_END, cls.TIMESTEPS)


if __name__ == "__main__":
    Config.init()

"""
DDPM 配置檔案
基於原始 DDPM 論文的超參數設定
"""
import os
from pathlib import Path
import torch


class Config:
    """DDPM 訓練配置"""

    # ==================== 路徑設定 ====================
    PROJECT_ROOT = Path(__file__).parent
    # 資料快取目錄；可用環境變數 DDPM_DATA_DIR 覆蓋
    DATA_CACHE_DIR = Path(os.environ.get("DDPM_DATA_DIR", PROJECT_ROOT / "data_cache"))
    OUTPUT_DIR = PROJECT_ROOT / "outputs_ddpm"
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    SAMPLE_DIR = OUTPUT_DIR / "samples"

    # ==================== 資料集設定 ====================
    DATASET = "celeba"  # "cifar10" 或 "celeba"
    IMAGE_SIZE = 64     # DDPM 原始論文使用 64x64 或 256x256
    CHANNELS = 3        # RGB

    # ==================== Diffusion 設定 ====================
    # 原始 DDPM 論文使用 1000 步
    TIMESTEPS = 1000

    # Beta schedule (線性)
    BETA_START = 1e-4
    BETA_END = 0.02

    # ==================== 模型設定 ====================
    # U-Net 通道數
    MODEL_CHANNELS = 128
    CHANNEL_MULT = (1, 2, 2, 2)  # 各層的通道倍數
    ATTENTION_RESOLUTIONS = (16,)  # 在 16x16 解析度加入 attention
    NUM_RES_BLOCKS = 2  # 每個解析度的殘差塊數量
    DROPOUT = 0.1

    # ==================== 訓練設定 ====================
    BATCH_SIZE = 64     # 原始論文用 128，但 64 更適合一般 GPU
    EPOCHS = 100
    LEARNING_RATE = 2e-4  # 原始論文使用 2e-4

    # EMA (Exponential Moving Average)
    EMA_DECAY = 0.9999
    USE_EMA = True

    # ==================== 儲存設定 ====================
    SAVE_INTERVAL = 1        # 每 N epochs 儲存檢查點 (改為每 epoch 都存)
    SAMPLE_INTERVAL = 5      # 每 N epochs 生成樣本
    NUM_SAMPLES = 64         # 生成樣本數量 (8x8 網格)

    # ==================== 其他 ====================
    NUM_WORKERS = 4
    SEED = 42

    @classmethod
    def get_device(cls):
        """自動選擇最佳設備"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    DEVICE = None  # 會在 init() 中設定

    @classmethod
    def init(cls):
        """初始化配置，建立必要目錄"""
        # 設定設備
        cls.DEVICE = cls.get_device()

        # 建立輸出目錄
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

        # 如果是 CPU，調整批次大小
        if cls.DEVICE.type == "cpu":
            cls.BATCH_SIZE = min(cls.BATCH_SIZE, 16)
            cls.NUM_WORKERS = 0
            print("警告: 使用 CPU 訓練，已調整批次大小")

        print(f"DDPM 配置初始化完成")
        print(f"  - 設備: {cls.DEVICE}")
        print(f"  - 資料集: {cls.DATASET}")
        print(f"  - 圖片大小: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"  - 時間步: {cls.TIMESTEPS}")
        print(f"  - 批次大小: {cls.BATCH_SIZE}")
        print(f"  - 輸出目錄: {cls.OUTPUT_DIR}")

    @classmethod
    def get_beta_schedule(cls):
        """取得 beta schedule"""
        return torch.linspace(cls.BETA_START, cls.BETA_END, cls.TIMESTEPS)


if __name__ == "__main__":
    Config.init()

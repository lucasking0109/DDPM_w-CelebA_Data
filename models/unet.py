"""
U-Net 模型
用於預測 DDPM 中的噪聲
包含時間嵌入、殘差塊和自注意力機制
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """
    正弦位置編碼 (Sinusoidal Position Embedding)
    將時間步 t 編碼為向量

    使用與 Transformer 相同的公式:
    PE(t, 2i) = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    殘差塊 (Residual Block)
    包含兩個卷積層和時間嵌入

    結構:
        x -> Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> + x
                    ^
                    |
               time_emb -> Linear -> SiLU -> Linear
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # 時間嵌入映射
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        # 如果輸入輸出通道不同，需要 1x1 卷積做殘差連接
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_emb):
        """
        Args:
            x: 輸入特徵圖 (batch, in_channels, height, width)
            time_emb: 時間嵌入 (batch, time_emb_dim)

        Returns:
            輸出特徵圖 (batch, out_channels, height, width)
        """
        residual = self.residual_conv(x)

        # 第一個卷積
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # 加入時間嵌入
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]  # 廣播到空間維度

        # 第二個卷積
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)

        return h + residual


class AttentionBlock(nn.Module):
    """
    自注意力塊 (Self-Attention Block)
    在低解析度特徵圖上使用
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)

    def forward(self, x):
        batch, channels, height, width = x.shape
        residual = x

        # 正規化
        x = self.norm(x)

        # 重塑為序列 (batch, seq_len, channels)
        x = x.view(batch, channels, height * width).permute(0, 2, 1)

        # 自注意力
        x, _ = self.attention(x, x, x)

        # 重塑回原來的形狀
        x = x.permute(0, 2, 1).view(batch, channels, height, width)

        return x + residual


class DownBlock(nn.Module):
    """下採樣塊"""

    def __init__(self, in_channels, out_channels, time_emb_dim, has_attention=False, dropout=0.1):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, time_emb_dim, dropout)
        self.attention = AttentionBlock(out_channels) if has_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        x = self.attention(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """上採樣塊"""

    def __init__(self, in_channels, out_channels, time_emb_dim, has_attention=False, dropout=0.1):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        # 輸入是 in_channels (上採樣後) + out_channels (skip connection)
        self.res_block = ResidualBlock(in_channels + out_channels, out_channels, time_emb_dim, dropout)
        self.attention = AttentionBlock(out_channels) if has_attention else nn.Identity()

    def forward(self, x, skip, time_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x, time_emb)
        x = self.attention(x)
        return x


class UNet(nn.Module):
    """
    U-Net 模型用於 DDPM

    結構:
        64x64 -> 32x32 -> 16x16 -> 8x8 (中間層) -> 16x16 -> 32x32 -> 64x64
                                    ^
                                    |
                              時間嵌入注入

    每個解析度包含殘差塊和可選的自注意力
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(16,),
        num_res_blocks=2,
        dropout=0.1,
        image_size=64,
    ):
        """
        Args:
            in_channels: 輸入通道數 (RGB=3)
            out_channels: 輸出通道數 (預測噪聲，也是 3)
            model_channels: 基礎通道數
            channel_mult: 各層通道倍數
            attention_resolutions: 在哪些解析度加入注意力
            num_res_blocks: 每個解析度的殘差塊數
            dropout: Dropout 比例
            image_size: 輸入圖片大小
        """
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.image_size = image_size

        # 時間嵌入
        time_emb_dim = model_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 初始卷積
        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # 下採樣路徑
        self.down_blocks = nn.ModuleList()
        channels = [model_channels]
        current_channels = model_channels
        current_resolution = image_size

        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            has_attention = current_resolution in attention_resolutions

            self.down_blocks.append(
                DownBlock(current_channels, out_ch, time_emb_dim, has_attention, dropout)
            )

            current_channels = out_ch
            channels.append(current_channels)
            current_resolution //= 2

        # 中間層
        self.mid_block1 = ResidualBlock(current_channels, current_channels, time_emb_dim, dropout)
        self.mid_attention = AttentionBlock(current_channels)
        self.mid_block2 = ResidualBlock(current_channels, current_channels, time_emb_dim, dropout)

        # 上採樣路徑
        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            # 上採樣時需要的輸入是 current_channels
            # skip connection 的通道數是 channels 中倒數對應的
            skip_ch = channels.pop()
            has_attention = current_resolution in attention_resolutions

            self.up_blocks.append(
                UpBlock(current_channels, out_ch, time_emb_dim, has_attention, dropout)
            )

            current_channels = out_ch
            current_resolution *= 2

        # 輸出層
        self.final_norm = nn.GroupNorm(8, model_channels)
        self.final_conv = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        Args:
            x: 輸入圖片 (batch, in_channels, height, width)
            t: 時間步 (batch,)

        Returns:
            預測的噪聲 (batch, out_channels, height, width)
        """
        # 時間嵌入
        time_emb = self.time_embedding(t.float())

        # 初始卷積
        x = self.init_conv(x)

        # 下採樣，保存 skip connections
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, time_emb)
            skips.append(skip)

        # 中間層
        x = self.mid_block1(x, time_emb)
        x = self.mid_attention(x)
        x = self.mid_block2(x, time_emb)

        # 上採樣，使用 skip connections
        for up_block in self.up_blocks:
            skip = skips.pop()
            x = up_block(x, skip, time_emb)

        # 輸出
        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    # 測試 UNet
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=64,
        channel_mult=(1, 2, 2),
        image_size=64,
    ).to(device)

    # 測試輸入
    x = torch.randn(2, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)

    # 前向傳播
    out = model(x, t)
    print(f"輸入形狀: {x.shape}")
    print(f"輸出形狀: {out.shape}")
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")

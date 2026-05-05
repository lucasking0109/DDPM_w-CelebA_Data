"""
U-Net for predicting noise in DDPM.
Includes time embeddings, residual blocks, and self-attention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal time embedding (same form as the Transformer paper).
        PE(t, 2i)   = sin(t / 10000^(2i/d))
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
    Residual block with two convs and time-embedding injection.

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

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        # 1x1 conv on the skip path when channel counts differ
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_emb):
        """
        Args:
            x: feature map (batch, in_channels, height, width)
            time_emb: time embedding (batch, time_emb_dim)

        Returns:
            feature map (batch, out_channels, height, width)
        """
        residual = self.residual_conv(x)

        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # Inject time embedding (broadcast over spatial dims)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)

        return h + residual


class AttentionBlock(nn.Module):
    """
    Self-attention block, applied at low spatial resolutions.
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)

    def forward(self, x):
        batch, channels, height, width = x.shape
        residual = x

        x = self.norm(x)

        # Reshape to (batch, seq_len, channels)
        x = x.view(batch, channels, height * width).permute(0, 2, 1)

        x, _ = self.attention(x, x, x)

        # Reshape back
        x = x.permute(0, 2, 1).view(batch, channels, height, width)

        return x + residual


class DownBlock(nn.Module):
    """Downsampling block"""

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
    """Upsampling block"""

    def __init__(self, in_channels, out_channels, time_emb_dim, has_attention=False, dropout=0.1):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        # Input is in_channels (after upsample) + out_channels (skip connection)
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
    U-Net for DDPM.

        64x64 -> 32x32 -> 16x16 -> 8x8 (mid) -> 16x16 -> 32x32 -> 64x64
                                    ^
                                    |
                              time embedding

    Each resolution has residual blocks plus optional self-attention.
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
            in_channels: input channels (RGB=3)
            out_channels: output channels (predicted noise, also 3)
            model_channels: base channel count
            channel_mult: channel multiplier at each level
            attention_resolutions: resolutions that get self-attention
            num_res_blocks: residual blocks per resolution
            dropout: dropout rate
            image_size: input image size
        """
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.image_size = image_size

        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # Down path
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

        # Middle
        self.mid_block1 = ResidualBlock(current_channels, current_channels, time_emb_dim, dropout)
        self.mid_attention = AttentionBlock(current_channels)
        self.mid_block2 = ResidualBlock(current_channels, current_channels, time_emb_dim, dropout)

        # Up path
        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            skip_ch = channels.pop()
            has_attention = current_resolution in attention_resolutions

            self.up_blocks.append(
                UpBlock(current_channels, out_ch, time_emb_dim, has_attention, dropout)
            )

            current_channels = out_ch
            current_resolution *= 2

        # Output
        self.final_norm = nn.GroupNorm(8, model_channels)
        self.final_conv = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        Args:
            x: input image (batch, in_channels, height, width)
            t: timestep (batch,)

        Returns:
            predicted noise (batch, out_channels, height, width)
        """
        time_emb = self.time_embedding(t.float())

        x = self.init_conv(x)

        # Down path, save skip connections
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, time_emb)
            skips.append(skip)

        # Middle
        x = self.mid_block1(x, time_emb)
        x = self.mid_attention(x)
        x = self.mid_block2(x, time_emb)

        # Up path with skip connections
        for up_block in self.up_blocks:
            skip = skips.pop()
            x = up_block(x, skip, time_emb)

        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    # Smoke test
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=64,
        channel_mult=(1, 2, 2),
        image_size=64,
    ).to(device)

    x = torch.randn(2, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)

    out = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

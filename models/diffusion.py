"""
Gaussian Diffusion 過程
實現 DDPM 的前向加噪和反向去噪過程
"""
import torch
import torch.nn.functional as F


class GaussianDiffusion:
    """
    DDPM 的擴散過程

    前向過程 (加噪):
        q(x_t | x_0) = N(x_t; √ᾱ_t * x_0, (1-ᾱ_t) * I)

    反向過程 (去噪):
        p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² * I)
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        """
        Args:
            timesteps: 總時間步數 T
            beta_start: β_1 (第一步的噪聲強度)
            beta_end: β_T (最後一步的噪聲強度)
            device: 運算設備
        """
        self.timesteps = timesteps
        self.device = device

        # 定義 beta schedule (線性)
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)

        # 預計算需要的值
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 前向過程需要的值
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 反向過程需要的值
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # 計算後驗分布的方差
        # σ²_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, tensor, t, x_shape):
        """
        從預計算的張量中提取對應時間步的值，並調整形狀以便廣播

        Args:
            tensor: 預計算的 1D 張量 (timesteps,)
            t: 時間步索引 (batch_size,)
            x_shape: 目標形狀 (batch, channels, height, width)

        Returns:
            形狀為 (batch, 1, 1, 1) 的張量，可以與 x 做廣播
        """
        batch_size = t.shape[0]
        out = tensor.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_0, t, noise=None):
        """
        前向過程: 對原始圖片加噪到時間步 t

        q(x_t | x_0) = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε

        Args:
            x_0: 原始圖片 (batch, channels, height, width)，範圍 [-1, 1]
            t: 時間步 (batch_size,)
            noise: 可選的預設噪聲

        Returns:
            x_t: 加噪後的圖片
            noise: 加入的噪聲 (用於計算損失)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        # x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    def p_losses(self, model, x_0, t):
        """
        計算訓練損失 (預測噪聲)

        Args:
            model: U-Net 模型，輸入 (x_t, t)，輸出預測的噪聲
            x_0: 原始圖片
            t: 時間步

        Returns:
            loss: MSE 損失
        """
        # 產生隨機噪聲
        noise = torch.randn_like(x_0)

        # 加噪到時間步 t
        x_t, _ = self.q_sample(x_0, t, noise=noise)

        # 預測噪聲
        predicted_noise = model(x_t, t)

        # 計算 MSE 損失
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def p_sample(self, model, x_t, t, t_index):
        """
        反向過程: 單步去噪 (從 x_t 到 x_{t-1})

        Args:
            model: U-Net 模型
            x_t: 當前加噪圖片
            t: 時間步張量 (batch_size,)
            t_index: 當前時間步索引 (整數)

        Returns:
            x_{t-1}: 去噪一步後的圖片
        """
        # 提取需要的係數
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)

        # 預測噪聲
        predicted_noise = model(x_t, t)

        # 計算均值
        # μ_θ(x_t, t) = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t))
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            # 最後一步不加噪聲
            return model_mean
        else:
            # 加入噪聲
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, return_intermediates=False):
        """
        完整的反向採樣過程: 從純噪聲生成圖片

        Args:
            model: U-Net 模型
            shape: 輸出形狀 (batch, channels, height, width)
            return_intermediates: 是否返回中間結果

        Returns:
            生成的圖片，範圍 [-1, 1]
        """
        device = self.device
        batch_size = shape[0]

        # 從純噪聲開始
        x = torch.randn(shape, device=device)

        intermediates = []

        # 反向迭代 T 步
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, i)

            if return_intermediates and i % 100 == 0:
                intermediates.append(x.clone())

        if return_intermediates:
            return x, intermediates
        return x

    @torch.no_grad()
    def sample(self, model, num_samples, image_size, channels=3):
        """
        便捷的採樣函數

        Args:
            model: U-Net 模型
            num_samples: 生成數量
            image_size: 圖片大小
            channels: 通道數

        Returns:
            生成的圖片 (num_samples, channels, image_size, image_size)
        """
        return self.p_sample_loop(
            model,
            shape=(num_samples, channels, image_size, image_size)
        )

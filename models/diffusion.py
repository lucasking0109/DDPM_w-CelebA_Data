"""
Gaussian Diffusion process.
Implements the forward (noising) and reverse (denoising) steps of DDPM.
"""
import torch
import torch.nn.functional as F


class GaussianDiffusion:
    """
    DDPM diffusion process.

    Forward (noising):
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    Reverse (denoising):
        p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        """
        Args:
            timesteps: total diffusion steps T
            beta_start: beta_1
            beta_end: beta_T
            device: torch device
        """
        self.timesteps = timesteps
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)

        # Precompute commonly used quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # For the forward pass
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For the reverse pass
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance: sigma^2_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, tensor, t, x_shape):
        """
        Gather values at timesteps t and reshape for broadcasting against x.

        Args:
            tensor: 1D tensor of shape (timesteps,)
            t: timestep indices (batch_size,)
            x_shape: target shape (batch, channels, height, width)

        Returns:
            tensor of shape (batch, 1, 1, 1)
        """
        batch_size = t.shape[0]
        out = tensor.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_0, t, noise=None):
        """
        Forward step: add noise to x_0 up to timestep t.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x_0: clean image (batch, channels, height, width) in [-1, 1]
            t: timesteps (batch_size,)
            noise: optional preset noise

        Returns:
            x_t, noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    def p_losses(self, model, x_0, t):
        """
        Training loss: predict the added noise.

        Args:
            model: U-Net taking (x_t, t) and outputting predicted noise
            x_0: clean image
            t: timestep

        Returns:
            MSE loss
        """
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise=noise)
        predicted_noise = model(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, model, x_t, t, t_index):
        """
        Single reverse step: x_t -> x_{t-1}.

        Args:
            model: U-Net
            x_t: current noisy image
            t: timestep tensor (batch_size,)
            t_index: integer timestep index

        Returns:
            x_{t-1}
        """
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)

        predicted_noise = model(x_t, t)

        # mu_theta(x_t, t) = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1 - alpha_bar_t) * eps_theta(x_t, t))
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            # No noise on the final step
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, return_intermediates=False):
        """
        Full reverse process: generate an image from pure noise.

        Args:
            model: U-Net
            shape: output shape (batch, channels, height, width)
            return_intermediates: also return intermediate snapshots

        Returns:
            generated image in [-1, 1]
        """
        device = self.device
        batch_size = shape[0]

        x = torch.randn(shape, device=device)
        intermediates = []

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
        Convenience wrapper around p_sample_loop.
        """
        return self.p_sample_loop(
            model,
            shape=(num_samples, channels, image_size, image_size)
        )

"""The noise recipe (notebook 01 builds this step by step; this is the same code, reused by 02 and 03)."""
import torch


class NoiseSchedule:
    """How much noise gets added at each of the T steps of the forward process."""

    def __init__(self, steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device="cpu"):
        self.steps = steps
        # beta_t: the small amount of fresh noise mixed in at step t
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1.0 - self.betas
        # alpha_bar_t: how much of the original image survives after t steps
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """Jump straight to step t: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise."""
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alpha_bars[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise, noise

    def to(self, device):
        self.betas, self.alphas, self.alpha_bars = (v.to(device) for v in (self.betas, self.alphas, self.alpha_bars))
        return self

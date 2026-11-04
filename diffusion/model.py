"""A small U-Net that looks at a noisy image and guesses the noise in it.

You don't need to understand every line here to follow the notebooks. The idea:
  - the image is squeezed down (to see the big picture) and expanded back up (to place details),
  - at every layer the network is told WHICH step t it is at, and WHICH digit was asked for,
  - the output has the same shape as the input: one noise guess per pixel.
"""
import math

import torch
from torch import nn


def step_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Turns the step number t into a list of numbers (sines and cosines at different speeds)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    angles = t.float()[:, None] * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, c_out)  # injects "which step / which digit" into this layer
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(nn.functional.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """~0.65M parameters (the "dials"). Input/output: 1 x 32 x 32 images (MNIST digits padded from 28 to 32)."""

    NO_LABEL = 10  # label id meaning "no digit requested" (used to learn unguided generation too)

    def __init__(self, base: int = 32, emb_dim: int = 128, num_labels: int = 10):
        super().__init__()
        self.emb_dim = emb_dim
        self.step_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        # one learned embedding (a list of numbers) per digit, plus one for "no label"
        self.label_emb = nn.Embedding(num_labels + 1, emb_dim)

        c1, c2, c3 = base, base * 2, base * 2
        self.inp = nn.Conv2d(1, c1, 3, padding=1)
        self.down1 = ResBlock(c1, c1, emb_dim)            # 32x32
        self.down2 = ResBlock(c1, c2, emb_dim)            # 16x16
        self.down3 = ResBlock(c2, c3, emb_dim)            # 8x8
        self.mid = ResBlock(c3, c3, emb_dim)
        self.up3 = ResBlock(c3 + c3, c2, emb_dim)         # 8x8
        self.up2 = ResBlock(c2 + c2, c1, emb_dim)         # 16x16
        self.up1 = ResBlock(c1 + c1, c1, emb_dim)         # 32x32
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 1, 3, padding=1))
        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t, labels):
        emb = self.step_mlp(step_embedding(t, self.emb_dim)) + self.label_emb(labels)
        h1 = self.down1(self.inp(x), emb)
        h2 = self.down2(self.pool(h1), emb)
        h3 = self.down3(self.pool(h2), emb)
        m = self.mid(h3, emb)
        u3 = self.up3(torch.cat([m, h3], 1), emb)
        u2 = self.up2(torch.cat([self.upsample(u3), h2], 1), emb)
        u1 = self.up1(torch.cat([self.upsample(u2), h1], 1), emb)
        return self.out(u1)  # the noise guess, same shape as x

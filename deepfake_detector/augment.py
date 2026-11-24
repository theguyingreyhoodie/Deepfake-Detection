"""Training-time image augmentations.

Each training crop is randomly perturbed so the classifier learns what manipulation looks like rather
than incidental properties of one dataset (its exact framing, sharpness, colour or compression level).
The first five operations and their strengths are GenD's defaults. The last two (re-scaling and noise)
are additions for real-world uploads, which are often re-encoded at low resolution by social media.

Everything runs on uint8 CHW tensors in the DataLoader workers; normalisation happens on the GPU.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T


class RandomRescale:
    """Shrink then re-enlarge the crop, simulating a low-resolution source video."""

    def __init__(self, p: float = 0.2, min_scale: float = 0.35, max_scale: float = 0.9):
        self.p, self.min_scale, self.max_scale = p, min_scale, max_scale

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return img
        h, w = img.shape[-2:]
        s = random.uniform(self.min_scale, self.max_scale)
        small = F.interpolate(img[None].float(), size=(max(8, int(h * s)), max(8, int(w * s))),
                              mode="area")
        mode = random.choice(["bilinear", "bicubic"])
        big = F.interpolate(small, size=(h, w), mode=mode, align_corners=False)
        return big[0].round().clamp(0, 255).to(torch.uint8)


class RandomNoise:
    """Mild Gaussian sensor noise."""

    def __init__(self, p: float = 0.05, max_sigma: float = 6.0):
        self.p, self.max_sigma = p, max_sigma

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return img
        sigma = random.uniform(1.0, self.max_sigma)
        return (img.float() + torch.randn_like(img, dtype=torch.float32) * sigma).round().clamp(0, 255).to(torch.uint8)


def train_transform() -> T.Compose:
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomApply([T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.1),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        RandomRescale(p=0.2),
        RandomNoise(p=0.05),
        T.JPEG(quality=(40, 100)),
    ])

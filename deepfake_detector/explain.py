"""Grad-CAM heatmaps: which parts of a face crop pushed the model towards "fake".

The image is cut into 16 x 16 patches of 14 x 14 pixels; the transformer keeps one feature vector per
patch. We take the patch features entering the LAST transformer block (that block is where the CLS
token reads every patch, so these features still influence the decision), and the gradient of the
"fake minus real" logit with respect to them. Weighting each feature channel by its average gradient
and summing gives one relevance value per patch; upsampled to 224 x 224 this is the heatmap.
It is a qualitative aid for humans, not a precise localisation of the manipulated pixels.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _encoder_layers(model) -> torch.nn.ModuleList:
    enc = model.encoder
    inner = getattr(enc, "vision_model", enc)  # transformers 4.x wraps the tower in .vision_model
    return inner.encoder.layers


def gradcam(model, pixels: torch.Tensor) -> np.ndarray:
    """``pixels``: (B, 3, 224, 224) CLIP-normalised. Returns (B, 224, 224) heatmaps scaled to [0, 1]."""
    store = {}

    def hook(_module, _inputs, output):
        act = output[0] if isinstance(output, tuple) else output
        act.retain_grad()
        store["act"] = act

    handle = _encoder_layers(model)[-2].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits, _ = model(pixels)
            (logits[:, 1] - logits[:, 0]).sum().backward()
        act, grad = store["act"].detach(), store["act"].grad.detach()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    weights = grad[:, 1:, :].mean(dim=1, keepdim=True)  # drop the CLS token, average over patches
    cam = F.relu((act[:, 1:, :] * weights).sum(-1)).float()  # (B, 256)
    side = int(round(cam.shape[1] ** 0.5))
    cam = cam.reshape(-1, 1, side, side)
    cam = F.interpolate(cam, size=pixels.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam - cam.flatten(1).min(1)[0][:, None, None]
    cam = cam / cam.flatten(1).max(1)[0].clamp(min=1e-8)[:, None, None]
    return cam.cpu().numpy()


def overlay(crop_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] heatmap (jet colour map) over an RGB crop."""
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return (crop_rgb * (1 - alpha) + colored * alpha).astype(np.uint8)

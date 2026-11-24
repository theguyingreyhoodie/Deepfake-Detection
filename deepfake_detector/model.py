"""The classifier: a frozen CLIP ViT-L/14 image encoder whose LayerNorms are fine-tuned.

Why this design (GenD, Yermakov et al., WACV 2026):

* CLIP's image encoder was pre-trained on 400 million image/text pairs, so it already describes faces,
  lighting, texture and compression in rich detail. We keep all of that knowledge frozen.
* We train only the LayerNorm scale and shift parameters (about 0.1 M of the encoder's 303 M weights)
  plus a 2-way linear layer. Changing so few numbers re-weights CLIP's existing features towards
  "is this face manipulated?" without letting the network memorise the quirks of one dataset.
  The paper shows this generalises to unseen deepfake methods much better than full fine-tuning.
* The 1024-d CLS feature is L2-normalised (put on the unit sphere) before the linear layer. Training
  adds two losses on that sphere: *alignment* pulls same-class features together, and *uniformity*
  spreads all features out, which keeps the representation from collapsing onto shortcuts.

Because only the LayerNorms and the head change, a trained model is a tiny file (well under 1 MB):
the big frozen encoder is downloaded once from Hugging Face.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE = "openai/clip-vit-large-patch14"
IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# Hugging Face names of every LayerNorm inside the CLIP vision tower ("layrnorm" is HF's own spelling).
LAYERNORM_KEYS = ("pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm")


def is_layernorm(name: str) -> bool:
    return any(key in name for key in LAYERNORM_KEYS)


def canonical(name: str) -> str:
    """Parameter name without "vision_model." (transformers 4.x has that level, 5.x does not)."""
    return name.replace("vision_model.", "")


class DeepfakeClassifier(nn.Module):
    """Face crop (3 x 224 x 224, CLIP-normalised) -> logits for [real, fake] and the unit-length feature."""

    def __init__(self, backbone: str = BACKBONE, pretrained: bool = True):
        super().__init__()
        from transformers import CLIPVisionConfig, CLIPVisionModel

        if pretrained:
            self.encoder = CLIPVisionModel.from_pretrained(backbone, attn_implementation="sdpa")
        else:
            config = CLIPVisionConfig.from_pretrained(backbone)
            config._attn_implementation = "sdpa"
            self.encoder = CLIPVisionModel(config)
        self.backbone_name = backbone
        dim = self.encoder.config.hidden_size
        self.head = nn.Linear(dim, 2)
        self.set_trainable()

    def set_trainable(self) -> None:
        """Freeze everything except LayerNorm parameters and the linear head."""
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith("head.") or is_layernorm(name))

    def features(self, pixels: torch.Tensor) -> torch.Tensor:
        """Unit-length 1024-d CLS feature (CLIP's pooled output after its final LayerNorm)."""
        return F.normalize(self.encoder(pixel_values=pixels).pooler_output, dim=-1)

    def forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.features(pixels)
        return self.head(z.to(self.head.weight.dtype)), z

    @torch.no_grad()
    def fake_probability(self, pixels: torch.Tensor) -> torch.Tensor:
        logits, _ = self(pixels)
        return logits.float().softmax(-1)[:, 1]

    # ---- saving and loading only what was trained -------------------------------------------------
    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Only the trained tensors, under version-independent names."""
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {canonical(k): v.detach().cpu().float() for k, v in self.state_dict().items() if k in names}

    def _load_canonical(self, state: dict[str, torch.Tensor]) -> tuple[set[str], set[str]]:
        """Load tensors given under canonical names; returns (loaded names, unexpected names)."""
        actual = {canonical(k): k for k in self.state_dict()}
        own = self.state_dict()
        loaded, unexpected = set(), set()
        for key, value in state.items():
            target = actual.get(canonical(key))
            if target is None:
                unexpected.add(key)
                continue
            own[target].copy_(value.to(own[target].dtype))
            loaded.add(target)
        return loaded, unexpected

    def save_trainable(self, path: str | Path, **extra) -> None:
        from safetensors.torch import save_file

        meta = {"backbone": self.backbone_name, **{k: str(v) for k, v in extra.items()}}
        save_file(self.trainable_state_dict(), str(path), metadata=meta)

    def load_trainable(self, path: str | Path) -> None:
        from safetensors.torch import load_file

        loaded, unexpected = self._load_canonical(load_file(str(path)))
        missing = {n for n, p in self.named_parameters() if p.requires_grad} - loaded
        if unexpected or missing:
            raise ValueError(f"Checkpoint mismatch: unexpected={sorted(unexpected)[:5]}, missing={sorted(missing)[:5]}")

    def load_gend_checkpoint(self, path: str | Path) -> None:
        """Load the authors' released GenD weights (``yermandy/GenD_CLIP_L_14``) for baseline comparison."""
        from safetensors.torch import load_file

        mapped = {}
        for key, value in load_file(str(path)).items():
            if key.startswith("feature_extractor.vision_model."):
                mapped["encoder." + key[len("feature_extractor.vision_model."):]] = value
            elif key.startswith("model.linear."):
                mapped["head." + key[len("model.linear."):]] = value
        loaded, _ = self._load_canonical(mapped)
        missing = set(self.state_dict()) - loaded
        if missing:
            raise ValueError(f"GenD checkpoint is missing {len(missing)} tensors, e.g. {sorted(missing)[:3]}")


def normalize(images_uint8: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) uint8 RGB -> float, CLIP mean/std normalised."""
    mean = torch.tensor(CLIP_MEAN, device=images_uint8.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=images_uint8.device).view(1, 3, 1, 1)
    return (images_uint8.float() / 255.0 - mean) / std

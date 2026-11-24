"""Alignment and uniformity losses on the unit sphere (Wang & Isola, ICML 2020), as used by GenD.

Both work on L2-normalised features ``z`` (every row has length 1), where the squared distance between
two features is ``||a - b||^2 = 2 - 2 * a.b``, so everything can be computed from one similarity matrix.

* alignment:  mean squared distance between pairs with the SAME label (pull each class together).
* uniformity: log of the mean Gaussian similarity over ALL pairs (push everything apart, so the
  features keep using the whole sphere instead of collapsing onto a few directions).
"""

from __future__ import annotations

import torch


def alignment_loss(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    z = z.float()
    sq_dist = (2 - 2 * z @ z.T).clamp(min=0)
    same = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    if not same.any():
        return z.new_zeros(())
    return sq_dist[same].mean()


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    z = z.float()
    n = z.shape[0]
    if n < 2:
        return z.new_zeros(())
    sq_dist = (2 - 2 * z @ z.T).clamp(min=0)
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=z.device), diagonal=1)
    return torch.exp(-t * sq_dist[upper]).mean().clamp(min=1e-6).log()

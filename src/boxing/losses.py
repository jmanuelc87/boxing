"""Loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCrossEntropy(nn.Module):
    """Cross entropy scaled by ``(1 - p_t) ** gamma`` (Lin et al., 2017).

    Confidently correct examples contribute little, so the gradient concentrates on hard and
    rare punches. ``gamma=0`` gives plain weighted cross entropy.

    Class ``weight`` is applied per target and normalised by the sum of weights, as in
    ``nn.CrossEntropyLoss``. ``label_smoothing`` is applied to the base cross entropy, and ``p_t``
    comes from the unsmoothed logits, so the focusing term sees the model's real confidence.
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_plain = F.cross_entropy(logits, target, reduction="none")
        p_t = torch.exp(-ce_plain)
        ce = F.cross_entropy(logits, target, reduction="none", label_smoothing=self.label_smoothing)
        focal = (1.0 - p_t).clamp(min=0.0) ** self.gamma * ce
        if self.weight is None:
            return focal.mean()
        w = self.weight[target]
        return (w * focal).sum() / w.sum()

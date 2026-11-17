"""Keypoint punch classifier.

`PunchPairClassifier` is a two-stream GRU over attacker and opponent pose keypoints
(see `boxing.pair_dataset` for the input layout and normalisation).
"""

from __future__ import annotations

import torch
from torch import nn

from boxing.pair_dataset import (
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    NUM_JOINTS,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    VALID_CONF,
)

# Per boxer and frame: (x, y, score) per joint, per-joint velocity, wrist-to-target vectors,
# the hip-centre offset to the other boxer, and a role flag.
_REL_DIM = 2 * 3 * 2 + 2
PUNCH_FEATURE_DIM = NUM_JOINTS * 3 + NUM_JOINTS * 2 + _REL_DIM + 1


def _midpoint(xy: torch.Tensor, score: torch.Tensor, i: int, j: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Midpoint of joints i and j, plus whether both were present. ``xy``: (B, T, J, 2)."""
    valid = (score[..., i] > VALID_CONF) & (score[..., j] > VALID_CONF)
    return (xy[..., i, :] + xy[..., j, :]) / 2, valid


def stream_features(own: torch.Tensor, other: torch.Tensor, role: float) -> torch.Tensor:
    """Per-frame features of one boxer seen against the other. Both ``(B, T, 17, 3)``.

    Returns ``(B, T, PUNCH_FEATURE_DIM)``. Anything involving a missing joint is zeroed,
    so a frame without the opponent contributes no relative signal.
    """
    xy, score = own[..., :2], own[..., 2]
    oxy, oscore = other[..., :2], other[..., 2]

    # Velocity, only where the joint is present on both frames.
    vel = xy[:, 1:] - xy[:, :-1]
    both = ((score[:, 1:] > 0) & (score[:, :-1] > 0)).unsqueeze(-1)
    vel = torch.cat([torch.zeros_like(vel[:, :1]), vel * both], dim=1)

    # Where each wrist is relative to the other boxer's head, shoulders and hips.
    wrists = xy[..., [L_WRIST, R_WRIST], :]  # (B, T, 2, 2)
    wrist_ok = score[..., [L_WRIST, R_WRIST]] > VALID_CONF  # (B, T, 2)
    shoulders, shoulders_ok = _midpoint(oxy, oscore, L_SHOULDER, R_SHOULDER)
    hips, hips_ok = _midpoint(oxy, oscore, L_HIP, R_HIP)
    targets = torch.stack([oxy[..., NOSE, :], shoulders, hips], dim=-2)  # (B, T, 3, 2)
    target_ok = torch.stack([oscore[..., NOSE] > VALID_CONF, shoulders_ok, hips_ok], dim=-1)  # (B, T, 3)
    vec = targets.unsqueeze(2) - wrists.unsqueeze(3)  # (B, T, 2, 3, 2)
    ok = (wrist_ok.unsqueeze(-1) & target_ok.unsqueeze(2)).unsqueeze(-1)
    rel = (vec * ok).flatten(2)

    own_hips, own_hips_ok = _midpoint(xy, score, L_HIP, R_HIP)
    gap = (hips - own_hips) * (own_hips_ok & hips_ok).unsqueeze(-1)

    role_flag = torch.full_like(score[..., :1], role)
    return torch.cat([own.flatten(2), vel.flatten(2), rel, gap, role_flag], dim=-1)


class PunchPairClassifier(nn.Module):
    """Classify a punch from a window of attacker and opponent keypoints.

    Input ``x``: ``(B, 2, T, 17, 3)``, row 0 the attacker and row 1 the opponent, normalised by
    `boxing.pair_dataset.normalize_window`. Output: ``(B, num_classes)`` logits.

    A single bidirectional GRU encodes each boxer, run once per role with shared weights. The two
    embeddings are fused by an MLP. Each stream also receives the wrist positions relative to the
    other boxer's head, shoulders and hips, so the model can tell a punch aimed at the opponent
    from a movement that is not.
    """

    def __init__(self, num_classes: int = 8, hidden: int = 128, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(PUNCH_FEATURE_DIM), nn.Linear(PUNCH_FEATURE_DIM, hidden), nn.GELU())
        self.gru = nn.GRU(
            hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        # mean and max pooling over time, each 2 * hidden wide
        self.embed = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU())
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes),
        )

    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.proj(feats))
        return self.embed(torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attacker, opponent = x[:, 0], x[:, 1]
        feats = torch.cat(
            [stream_features(attacker, opponent, 1.0), stream_features(opponent, attacker, 0.0)], dim=0
        )
        att_emb, opp_emb = self.encode(feats).chunk(2, dim=0)
        return self.head(torch.cat([att_emb, opp_emb], dim=-1))

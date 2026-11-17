"""Dataset for the paired keypoint windows written by ``boxing.pairing``.

Each item is ``(x, y)`` with ``x`` of shape ``(2, T, 17, 3)``: row 0 is the
attacker, row 1 the opponent, and the last axis is ``(x, y, score)``.

Coordinates are normalised per window so the model sees body-relative motion
rather than camera position or zoom:

* origin: the attacker's mean hip centre over the window,
* scale: the attacker's mean torso length (shoulder centre to hip centre).

Both boxers share the same transform, so their relative position is kept.
Joints with score 0 (people missing from a frame) stay at (0, 0, 0).

    ds = PunchPairDataset(Path("data/processed/punch-pairs"), "train", augment=True)
    x, y = ds[0]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_JOINTS = 17

# COCO keypoint indices.
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

# Left/right swap for horizontal mirroring.
FLIP_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# A joint counts as visible above this score.
VALID_CONF = 0.1
# Floor for the torso scale in pixels, so a degenerate pose cannot blow up the coordinates.
MIN_TORSO_PX = 20.0
DEFAULT_TORSO_PX = 100.0


def load_names(root: Path) -> list[str]:
    return json.loads((root / "labels.json").read_text(encoding="utf-8"))["names"]


def normalize_window(w: np.ndarray) -> np.ndarray:
    """Pixel window ``(2, T, 17, 3)`` -> body-relative window of the same shape."""
    att = w[0]
    score = att[..., 2]

    hips_ok = (score[:, L_HIP] > VALID_CONF) & (score[:, R_HIP] > VALID_CONF)
    hip_mid = (att[:, L_HIP, :2] + att[:, R_HIP, :2]) / 2
    if hips_ok.any():
        center = hip_mid[hips_ok].mean(axis=0)
    else:
        visible = score > VALID_CONF
        center = att[..., :2][visible].mean(axis=0) if visible.any() else np.zeros(2, dtype=np.float32)

    torso_ok = hips_ok & (score[:, L_SHOULDER] > VALID_CONF) & (score[:, R_SHOULDER] > VALID_CONF)
    if torso_ok.any():
        shoulder_mid = (att[:, L_SHOULDER, :2] + att[:, R_SHOULDER, :2]) / 2
        torso = float(np.linalg.norm(shoulder_mid[torso_ok] - hip_mid[torso_ok], axis=-1).mean())
    else:
        torso = DEFAULT_TORSO_PX
    scale = max(torso, MIN_TORSO_PX)

    out = w.astype(np.float32, copy=True)
    out[..., :2] = (out[..., :2] - center) / scale
    out[..., :2][w[..., 2] <= 0] = 0.0
    return out


def augment_window(x: np.ndarray, rng: np.random.Generator, flip: bool = True) -> np.ndarray:
    """Random mirror, zoom, jitter and joint dropout on a normalised window."""
    x = x.copy()
    if flip and rng.random() < 0.5:
        # Mirroring swaps the hands, so a left-handed jab from an orthodox boxer
        # becomes a right-handed jab from a southpaw: the class is unchanged.
        x = x[..., FLIP_PERM, :]
        x[..., 0] *= -1
    x[..., :2] *= rng.uniform(0.9, 1.1)
    present = x[..., 2:3] > 0
    x[..., :2] += rng.normal(0.0, 0.01, size=x[..., :2].shape).astype(np.float32) * present
    drop = rng.random(x.shape[:-1]) < 0.05
    x[drop] = 0.0
    return x


class PunchPairDataset(Dataset):
    """One item per window: ``(x[2, T, 17, 3] float32, y int64)``.

    ``quality="all"`` keeps every window that has an attacker; ``"good"`` keeps only
    windows with the attacker and opponent on every frame.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        quality: str = "all",
        augment: bool = False,
        flip: bool = True,
        seed: int = 0,
    ):
        if quality not in ("all", "good"):
            raise ValueError(f"quality must be 'all' or 'good', got {quality!r}")
        root = Path(root)
        data = np.load(root / f"{split}.npz")
        meta = [json.loads(line) for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(meta) != len(data["y"]):
            raise ValueError(f"{split}: {len(meta)} metadata rows but {len(data['y'])} windows")

        wanted = {"good"} if quality == "good" else {"good", "partial"}
        keep = np.array([m["quality"] in wanted for m in meta], dtype=bool)
        self.meta = [m for m, k in zip(meta, keep) if k]
        # Position of each kept window in the split's full .npz order, to align side features.
        self.indices = np.flatnonzero(keep)
        self.X = np.stack([normalize_window(w) for w in data["X"][keep]]) if keep.any() else np.zeros((0, 2, 0, NUM_JOINTS, 3), np.float32)
        self.y = data["y"][keep].astype(np.int64)
        self.augment = augment
        self.flip = flip
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[i]
        if self.augment:
            x = augment_window(x, self.rng, self.flip)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(self.y[i])

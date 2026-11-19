"""RGB clips plus keypoints for EPAM-Net, built from the windows written by ``boxing.pairing``.

Each item is ``(images, keypoints, label)``:

    images     (T, 3, 224, 224) float32, ImageNet-normalised RGB
    keypoints  (2, T, 17, 3)    float32, (x, y, score) with x and y in [0, 1] of the same crop
    label      int64

Row 0 of ``keypoints`` is the first boxer and row 1 the second (by DeepSORT track id). The crop is the square bounding box
of all confident keypoints of both boxers over the window, padded, so the gloves stay large in
the 224x224 frame. Images and keypoints share one crop, which is what the pose heatmaps need to
line up with the RGB features.

Training augmentation: random crop scale and shift, horizontal mirror (with the left and right
joints swapped, so the class is unchanged), and brightness and contrast jitter.

    ds = EPAMDataset(Path("data/processed/punch-pairs"), Path("data/processed/punch-seq"), "train", augment=True)
    images, keypoints, label = ds[0]
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_CROP_PX = 64

# COCO-17 left/right joint swap, used when mirroring a window.
FLIP_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
# A joint counts as visible above this score.
VALID_CONF = 0.1


def crop_square(win: np.ndarray, width: int, height: int, margin: float = 0.25) -> tuple[int, int, int]:
    """Square crop ``(x0, y0, side)`` around all confident keypoints of a window ``(2, T, 17, 3)``."""
    pts = win[..., :2][win[..., 2] > VALID_CONF]
    if len(pts) == 0:
        side = min(width, height)
        return (width - side) // 2, (height - side) // 2, side
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = min(max(x2 - x1, y2 - y1) * (1 + 2 * margin), width, height)
    side = max(side, MIN_CROP_PX)
    side = min(side, width, height)
    x0 = int(np.clip(cx - side / 2, 0, width - side))
    y0 = int(np.clip(cy - side / 2, 0, height - side))
    return x0, y0, int(side)


def jitter_crop(
    x0: int, y0: int, side: int, width: int, height: int, rng: np.random.Generator
) -> tuple[int, int, int]:
    """Randomly rescale (0.9-1.15x) and shift (up to 5% of the side) a crop, kept inside the image."""
    new_side = int(min(max(side * rng.uniform(0.9, 1.15), 32), width, height))
    cx = x0 + side / 2 + rng.uniform(-0.05, 0.05) * side
    cy = y0 + side / 2 + rng.uniform(-0.05, 0.05) * side
    nx0 = int(np.clip(cx - new_side / 2, 0, width - new_side))
    ny0 = int(np.clip(cy - new_side / 2, 0, height - new_side))
    return nx0, ny0, new_side


def keypoints_in_crop(win: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    """Pixel keypoints to crop coordinates in [0, 1]; missing joints stay (0, 0, 0)."""
    kp = win.astype(np.float32, copy=True)
    kp[..., 0] = (kp[..., 0] - x0) / side
    kp[..., 1] = (kp[..., 1] - y0) / side
    kp[kp[..., 2] <= 0] = 0.0
    return kp


def mirror(images: np.ndarray, kp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip frames left-right and mirror the keypoints, swapping left and right joints."""
    flipped = kp[..., FLIP_PERM, :].copy()
    flipped[..., 0] = 1.0 - flipped[..., 0]
    flipped[flipped[..., 2] <= 0] = 0.0
    return images[:, :, ::-1], flipped


class EPAMDataset(Dataset):
    """One item per paired window: RGB frames and keypoints in one shared crop.

    ``quality="all"`` keeps every window with at least one boxer tracked; ``"good"`` keeps only
    windows with both boxers on every frame. ``limit`` keeps that many windows, spread
    evenly across the split (used by the smoke-test mode).
    """

    def __init__(
        self,
        pairs: Path,
        src: Path,
        split: str,
        quality: str = "all",
        augment: bool = False,
        limit: int | None = None,
    ):
        if quality not in ("all", "good"):
            raise ValueError(f"quality must be 'all' or 'good', got {quality!r}")
        self.src = Path(src)
        self.augment = augment
        data = np.load(Path(pairs) / f"{split}.npz")
        self.X, self.labels = data["X"], data["y"].astype(np.int64)
        meta = _read_jsonl(Path(pairs) / f"{split}.jsonl")
        self.records = _read_jsonl(self.src / f"{split}.jsonl")
        if not (len(meta) == len(self.records) == len(self.X)):
            raise ValueError(f"{split}: {len(self.records)} clips, {len(meta)} metadata rows, {len(self.X)} windows")
        for m, r in zip(meta, self.records):
            if m["task"] != r["task"] or m["start"] != r["start"]:
                raise ValueError(f"{split}: pairing metadata and clip records are out of order")

        wanted = {"good"} if quality == "good" else {"good", "partial"}
        keep = np.flatnonzero([m["quality"] in wanted for m in meta])
        if len(keep) == 0 and len(meta) > 0:
            counts = dict(Counter(m["quality"] for m in meta))
            raise ValueError(
                f"{split}: none of the {len(meta)} paired windows has quality in {sorted(wanted)} (found {counts}). "
                "If they are all 'no_boxers', the keypoints are missing or empty: re-run boxing.keypoints, then boxing.pairing."
            )
        if limit is not None and limit < len(keep):
            keep = keep[np.linspace(0, len(keep) - 1, limit).astype(int)]
        self.indices = keep
        self.y = self.labels[keep]
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def _generator(self) -> np.random.Generator:
        # Created lazily so every DataLoader worker draws a different stream, set by torch's seed.
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
        return self._rng

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = self.indices[i]
        rec, win = self.records[k], self.X[k]
        width, height = rec["width"], rec["height"]
        x0, y0, side = crop_square(win, width, height)
        rng = self._generator() if self.augment else None
        if rng is not None:
            x0, y0, side = jitter_crop(x0, y0, side, width, height, rng)

        frames = []
        for f in rec["frames"]:
            img = cv2.imread(str(self.src / f["image"]))
            if img is None:
                raise FileNotFoundError(self.src / f["image"])
            crop = img[y0 : y0 + side, x0 : x0 + side]
            frames.append(cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA))
        images = np.stack(frames)[..., ::-1]  # BGR -> RGB, (T, H, W, 3)
        kp = keypoints_in_crop(win, x0, y0, side)

        if rng is not None:
            if rng.random() < 0.5:
                images, kp = mirror(images, kp)
            images = images.astype(np.float32) / 255.0
            images = (images - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.1, 0.1)
            images = np.clip(images, 0.0, 1.0)
        else:
            images = images.astype(np.float32) / 255.0
        images = ((images - MEAN) / STD).transpose(0, 3, 1, 2)

        return (
            torch.from_numpy(np.ascontiguousarray(images, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(kp)),
            torch.tensor(self.labels[k]),
        )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

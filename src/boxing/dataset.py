"""PyTorch loading for the clip dataset written by ``boxing.preprocess``.

Each item is one clip: ``seq_len`` frames and one detection target per frame.

    ds = PunchSequenceDataset(Path("data/processed/punch-seq"), "train")
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_clips)
    for x, targets in dl:                      # x: (B, T, 3, S, S)
        x, targets = flatten_clips(x, targets)  # x: (B*T, 3, S, S)
        loss = model(pixel_values=x, labels=targets).loss

``flatten_clips`` lets the single-image ``BoxingDetector`` train on clips
unchanged: every frame becomes its own image, in clip order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# ImageNet statistics, matching the timm backbones' pretraining.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

Target = dict[str, torch.Tensor]


def load_labels(root: Path) -> dict:
    return json.loads((root / "labels.json").read_text(encoding="utf-8"))


def load_id2label(root: Path) -> dict[int, str]:
    """Class id -> name from ``labels.json`` (JSON keys are strings)."""
    return {int(k): v for k, v in load_labels(root)["id2label"].items()}


def frame_target(frame: dict) -> Target:
    """Detection target for one frame; an empty frame gives shapes (0,) and (0, 4)."""
    return {
        "class_labels": torch.tensor(frame["class_labels"], dtype=torch.long),
        "boxes": torch.tensor(frame["boxes"], dtype=torch.float32).reshape(-1, 4),
    }


class PunchSequenceDataset(Dataset):
    """One item per clip: ``(x[T, 3, S, S], [target] * T)``.

    Frames are resized to a ``image_size`` square and normalised with the
    ImageNet mean/std. Boxes are normalised ``(cx, cy, w, h)``, so the resize
    does not change them. No horizontal flip: it would swap left/right hand and
    so the class.
    """

    def __init__(self, root: Path, split: str, image_size: int = 640):
        self.root = Path(root)
        self.image_size = image_size
        labels = load_labels(self.root)
        if "seq_len" not in labels:
            raise ValueError(
                f"{self.root} is a per-frame dataset (no seq_len in labels.json); "
                "rebuild it with the current boxing.preprocess"
            )
        self.seq_len = int(labels["seq_len"])
        lines = (self.root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        self.records = [json.loads(line) for line in lines if line]

    def __len__(self) -> int:
        return len(self.records)

    def load_frame(self, image: str) -> torch.Tensor:
        img = Image.open(self.root / image).convert("RGB").resize((self.image_size, self.image_size))
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
        return (x - MEAN) / STD

    def __getitem__(self, i: int) -> tuple[torch.Tensor, list[Target]]:
        frames = self.records[i]["frames"]
        x = torch.stack([self.load_frame(f["image"]) for f in frames])
        return x, [frame_target(f) for f in frames]


def collate_clips(
    batch: list[tuple[torch.Tensor, list[Target]]],
) -> tuple[torch.Tensor, list[list[Target]]]:
    """Stack clips into ``x[B, T, 3, S, S]`` and keep targets as ``B`` lists of ``T`` dicts."""
    xs, ts = zip(*batch)
    return torch.stack(xs), list(ts)


def flatten_clips(x: torch.Tensor, targets: list[list[Target]]) -> tuple[torch.Tensor, list[Target]]:
    """Turn ``x[B, T, ...]`` into ``x[B*T, ...]`` and the targets into one flat list, in the same order."""
    b, t = x.shape[:2]
    if len(targets) != b or any(len(clip) != t for clip in targets):
        raise ValueError(f"targets do not match a ({b}, {t}) clip batch")
    return x.reshape(b * t, *x.shape[2:]), [frame for clip in targets for frame in clip]

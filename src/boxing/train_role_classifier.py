"""Baseline role classifier trained on hand labels exported from Label Studio.

    uv run python -m boxing.train_role_classifier EXPORT.json CROPS_DIR [--keypoints DIR]
        [--holdout TASK] [--out DIR]

EXPORT.json is the JSON export of the labeled project. CROPS_DIR is the output of
``boxing.person_crops`` (it holds the crop images). KEYPOINTS is the directory of
per-frame keypoint JSONs (``data/processed/punch-keypoints/images``), used to locate each
person's torso so the colour features match the colour rules.

Each labeled crop is described by a pretrained Model embedding plus the four torso colour
fractions (red, blue, white, black). A linear head is trained on the standardised features.
The head is also trained on embeddings alone so the effect of the colour features is visible.

The held-out task is a whole task folder, so the score reflects a new camera/session rather
than near-duplicate frames. The colour rules are scored on the same crops for comparison.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from boxing.color_roles import colour_fractions, torso_roi
from boxing.person_crops import padded_box

ROLES = ["boxer_red", "boxer_blue", "referee", "other"]
CROP_PAD = 0.1  # must match the --pad used by boxing.person_crops (its default)
COLOUR_KEYS = ("red", "blue", "white", "black")


def load_labels(export_path: Path, crops_dir: Path) -> list[dict]:
    """One item per labeled crop: its path, task, human label, colour-rule prediction and source."""
    items = []
    for task in json.loads(export_path.read_text(encoding="utf-8")):
        annotations = [
            a for a in task.get("annotations", []) if not a.get("was_cancelled")
        ]
        if not annotations:
            continue
        result = annotations[-1]["result"]
        if not result:
            continue
        label = result[0]["value"]["choices"][0]

        data = task["data"]
        crop = (
            crops_dir / data["task"] / f"{data['frame']}_{int(data['person']):02d}.jpg"
        )
        if not crop.exists():
            continue

        # In the export, ``predictions`` holds only IDs. The colour-rule result is stored on
        # the annotation's ``prediction`` object (present when the task was pre-labelled).
        rule = None
        pre = annotations[-1].get("prediction") or {}
        if pre.get("result"):
            rule = pre["result"][0]["value"]["choices"][0]
        items.append(
            {
                "path": crop,
                "task": data["task"],
                "frame": data["frame"],
                "person": int(data["person"]),
                "label": label,
                "rule": rule,
            }
        )
    return items


def colour_features(item: dict, keypoints_dir: Path) -> list[float]:
    """Torso colour fractions measured inside the crop, using the same torso region as the rules."""
    rec = json.loads(
        (keypoints_dir / item["task"] / f"{item['frame']}.json").read_text(
            encoding="utf-8"
        )
    )
    frame_h, frame_w = rec["height"], rec["width"]
    person = rec["people"][item["person"]]

    # Torso in full-frame pixels, then shifted into the crop's coordinate system.
    x1, y1, x2, y2 = torso_roi(person, frame_h, frame_w)
    ox, oy, _, _ = padded_box(person["box"], CROP_PAD, frame_w, frame_h)
    crop = cv2.imread(str(item["path"]))
    ch, cw = crop.shape[:2]
    cx1, cy1 = max(0, x1 - ox), max(0, y1 - oy)
    cx2, cy2 = min(cw, x2 - ox), min(ch, y2 - oy)
    if cx2 <= cx1 or cy2 <= cy1:  # torso outside the crop: no colour evidence
        return [0.0] * len(COLOUR_KEYS)
    fractions = colour_fractions(crop[cy1:cy2, cx1:cx2])
    return [fractions[k] for k in COLOUR_KEYS]


def embed(paths: list[Path], device: torch.device) -> torch.Tensor:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    backbone = models.resnet18(weights=weights)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)
    preprocess = weights.transforms()

    feats = []
    with torch.no_grad():
        for start in range(0, len(paths), 64):
            batch = torch.stack(
                [
                    preprocess(Image.open(p).convert("RGB"))
                    for p in paths[start : start + 64]
                ]
            ).to(device)
            feats.append(backbone(batch).cpu())
    return torch.cat(feats)


def standardise(x: torch.Tensor, train_idx: list[int]) -> torch.Tensor:
    """Scale each feature using statistics from the training rows only."""
    mean = x[train_idx].mean(dim=0)
    std = x[train_idx].std(dim=0).clamp_min(1e-6)
    return (x - mean) / std


def train_head(x: torch.Tensor, y: torch.Tensor, epochs: int = 300) -> nn.Linear:
    head = nn.Linear(x.shape[1], len(ROLES))
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(head(x), y)
        loss.backward()
        opt.step()
    return head


def evaluate(
    x: torch.Tensor, y: torch.Tensor, train_idx: list[int], test_idx: list[int]
) -> list[int]:
    x = standardise(x, train_idx)
    head = train_head(x[train_idx], y[train_idx])
    with torch.no_grad():
        return head(x[test_idx]).argmax(dim=1).tolist()


class FineTuner(nn.Module):
    """ResNet-101 backbone (trainable) with the colour features concatenated before the head.

    torchvision has no ResNet-108; ResNet-101 is the nearest available depth.
    """

    def __init__(self, n_colour: int):
        super().__init__()
        weights = models.ResNet101_Weights.IMAGENET1K_V2
        self.backbone = models.resnet101(weights=weights)
        self.backbone.fc = nn.Identity()  # 2048-d pooled features
        self.head = nn.Linear(2048 + n_colour, len(ROLES))

    def forward(self, images: torch.Tensor, colour: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.backbone(images), colour], dim=1))


def preprocess_images(paths: list[Path]) -> torch.Tensor:
    """Resize and normalise every crop once, so fine-tuning doesn't re-decode JPEGs each epoch."""
    preprocess = models.ResNet101_Weights.IMAGENET1K_V2.transforms()
    return torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths])


def finetune_and_evaluate(
    images: torch.Tensor,
    colour: torch.Tensor,
    y: torch.Tensor,
    train_idx: list[int],
    test_idx: list[int],
    device: torch.device,
    epochs: int = 8,
    batch_size: int = 32,
) -> tuple[list[int], dict]:
    """Fine-tune the backbone; return test predictions and the state needed to reload the model."""
    mean = colour[train_idx].mean(dim=0)
    std = colour[train_idx].std(dim=0).clamp_min(1e-6)
    colour = (colour - mean) / std

    model = FineTuner(colour.shape[1]).to(device)
    backbone_params = list(model.backbone.parameters())
    opt = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": 1e-4},
            {"params": model.head.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    # Inverse-frequency weights so the small "other" class is not ignored.
    counts = torch.bincount(y[train_idx], minlength=len(ROLES)).float().clamp_min(1)
    weights = (counts.sum() / (len(ROLES) * counts)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_idx))
        total = 0.0
        for start in range(0, len(order), batch_size):
            idx = [train_idx[j] for j in order[start : start + batch_size]]
            batch = images[idx].to(device)
            if (
                torch.rand(1).item() < 0.5
            ):  # horizontal flip only: keeps the colour cues intact
                batch = torch.flip(batch, dims=[3])
            opt.zero_grad()
            loss = loss_fn(model(batch, colour[idx].to(device)), y[idx].to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"  epoch {epoch + 1}/{epochs} loss {total / len(train_idx):.3f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(test_idx), 64):
            idx = test_idx[start : start + 64]
            logits = model(images[idx].to(device), colour[idx].to(device))
            preds += logits.argmax(dim=1).cpu().tolist()
    state = {
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "colour_mean": mean,
        "colour_std": std,
    }
    return preds, state


def set_seed(seed: int) -> None:
    """Seed every source of randomness used here (head init, batch order, flips)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_metrics(truth: list[str], pred: list[str]) -> dict:
    """Accuracy, per-class precision/recall/F1 and a confusion matrix (rows = true, cols = predicted)."""
    index = {r: k for k, r in enumerate(ROLES)}
    matrix = [[0] * len(ROLES) for _ in ROLES]
    for t, p in zip(truth, pred):
        matrix[index[t]][index[p]] += 1

    per_class = {}
    for k, role in enumerate(ROLES):
        tp = matrix[k][k]
        support = sum(matrix[k])
        predicted = sum(row[k] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class[role] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    present = [m["f1"] for m in per_class.values() if m["support"]]
    return {
        "accuracy": sum(matrix[k][k] for k in range(len(ROLES))) / max(1, len(truth)),
        "macro_f1": sum(present) / max(1, len(present)),
        "n": len(truth),
        "confusion": matrix,
        "per_class": per_class,
    }


def report(name: str, truth: list[str], pred: list[str]) -> dict:
    m = compute_metrics(truth, pred)
    print(f"\n== {name} ==")
    print(
        f"accuracy {m['accuracy']:.1%} ({round(m['accuracy'] * m['n'])}/{m['n']})   macro-F1 {m['macro_f1']:.3f}"
    )

    width = max(len(r) for r in ROLES) + 2
    print("confusion matrix (rows = true, columns = predicted)")
    print(" " * width + "".join(f"{r:>{width}}" for r in ROLES))
    for k, role in enumerate(ROLES):
        print(f"{role:<{width}}" + "".join(f"{v:>{width}}" for v in m["confusion"][k]))

    print(f"{'':<{width}}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for role in ROLES:
        c = m["per_class"][role]
        print(
            f"{role:<{width}}{c['precision']:>10.3f}{c['recall']:>10.3f}{c['f1']:>10.3f}{c['support']:>10}"
        )
    return m


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("export", type=Path)
    parser.add_argument("crops", type=Path)
    parser.add_argument(
        "--keypoints",
        type=Path,
        default=Path("data/processed/punch-keypoints/images"),
        help="directory of <task>/<frame>.json keypoint files",
    )
    parser.add_argument(
        "--holdout",
        default=None,
        help="task folder to hold out for testing (default: the last one)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for reproducible runs (default: 0)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to save the fine-tuned model and report.json",
    )
    args = parser.parse_args()
    print(f"seed {args.seed}")

    items = load_labels(args.export, args.crops)
    tasks = sorted({i["task"] for i in items})
    holdout = args.holdout or tasks[-1]
    print(
        f"{len(items)} labeled crops across {len(tasks)} tasks; holding out {holdout}"
    )
    print("label counts:", dict(collections.Counter(i["label"] for i in items)))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    emb = embed([i["path"] for i in items], device)
    colour = torch.tensor(
        [colour_features(i, args.keypoints) for i in items], dtype=torch.float32
    )
    both = torch.cat([emb, colour], dim=1)

    train_idx = [k for k, i in enumerate(items) if i["task"] != holdout]
    test_idx = [k for k, i in enumerate(items) if i["task"] == holdout]
    if not train_idx or not test_idx:
        raise SystemExit("need labels in both the training tasks and the held-out task")

    y = torch.tensor([ROLES.index(i["label"]) for i in items])
    truth = [items[k]["label"] for k in test_idx]

    results: dict[str, dict] = {}

    # Re-seed before each stage so every stage is reproducible on its own.
    set_seed(args.seed)
    pred_idx = evaluate(emb, y, train_idx, test_idx)
    results["frozen ResNet-18 embeddings"] = report(
        "frozen ResNet-18 embeddings", truth, [ROLES[p] for p in pred_idx]
    )

    set_seed(args.seed)
    pred_idx = evaluate(both, y, train_idx, test_idx)
    results["frozen ResNet-18 + colour"] = report(
        "frozen ResNet-18 + colour", truth, [ROLES[p] for p in pred_idx]
    )

    print("\nfine-tuning ResNet-101 end to end...")
    set_seed(args.seed)
    images = preprocess_images([i["path"] for i in items])
    ft_pred, ft_state = finetune_and_evaluate(
        images, colour, y, train_idx, test_idx, device
    )
    results["fine-tuned ResNet-101 + colour"] = report(
        "fine-tuned ResNet-101 + colour", truth, [ROLES[p] for p in ft_pred]
    )

    rule_pairs = [
        (items[k]["label"], items[k]["rule"]) for k in test_idx if items[k]["rule"]
    ]
    if rule_pairs:
        results["colour rules"] = report(
            "colour rules", [t for t, _ in rule_pairs], [p for _, p in rule_pairs]
        )

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": ft_state["model"],
                "backbone": "resnet101",
                "roles": ROLES,
                "colour_keys": COLOUR_KEYS,
                "colour_mean": ft_state["colour_mean"],
                "colour_std": ft_state["colour_std"],
                "seed": args.seed,
                "holdout": holdout,
            },
            args.out / "role_model.pt",
        )
        (args.out / "report.json").write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "holdout": holdout,
                    "roles": ROLES,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"\nsaved fine-tuned model to {args.out / 'role_model.pt'} and metrics to {args.out / 'report.json'}"
        )


if __name__ == "__main__":
    main()

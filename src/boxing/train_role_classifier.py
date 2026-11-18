"""Baseline role classifier trained on hand labels exported from Label Studio.

    uv run python -m boxing.train_role_classifier EXPORT.json CROPS_DIR
        [--holdout TASK] [--out DIR]

EXPORT.json is the JSON export of the labeled project. CROPS_DIR is the output of
``boxing.person_crops`` (it holds the crop images).

Each labeled crop is described by a pretrained timm embedding. A linear head is trained on the
standardised embeddings, and the backbone is also fine-tuned end to end.

The held-out task is a whole task folder, so the score reflects a new camera/session rather
than near-duplicate frames. The colour rules are scored on the same crops for comparison.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image

from boxing.person_crops import padded_box

log = logging.getLogger("boxing.train_role_classifier")

ROLES = ["boxer_red", "boxer_blue", "referee", "other"]


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


def embed(paths: list[Path], device: torch.device) -> torch.Tensor:
    backbone, _ = build_backbone("resnet18")
    backbone.eval().to(device)
    preprocess = backbone_transform("resnet18")

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


BACKBONES = (
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
)


def build_backbone(name: str) -> tuple[nn.Module, int]:
    """Pretrained timm backbone with the classifier removed; returns (model, pooled feature dim)."""
    model = timm.create_model(name, pretrained=True, num_classes=0)  # pooled features
    return model, model.num_features


def backbone_transform(name: str):
    """Preprocessing that matches the pretrained weights of a timm backbone."""
    model = timm.create_model(name, pretrained=True, num_classes=0)
    return timm.data.create_transform(**timm.data.resolve_data_config(model=model))


class FineTuner(nn.Module):
    """Trainable backbone (selected by name) with a linear role head on its pooled features."""

    def __init__(self, backbone: str):
        super().__init__()
        self.backbone, dim = build_backbone(backbone)
        self.head = nn.Linear(dim, len(ROLES))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))


class FocalCrossEntropy(nn.Module):
    """Cross-entropy scaled by (1 - p_t) ** gamma so easy examples contribute less.

    ``weight`` is the per-class weight (as in ``nn.CrossEntropyLoss``); ``gamma=0`` reduces
    to plain weighted cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = torch.log_softmax(logits, dim=1)
        log_pt = log_p.gather(1, target[:, None]).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def preprocess_images(paths: list[Path], backbone: str) -> torch.Tensor:
    """Resize and normalise every crop once, so fine-tuning doesn't re-decode JPEGs each epoch."""
    preprocess = backbone_transform(backbone)
    return torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths])


def finetune_and_evaluate(
    images: torch.Tensor,
    y: torch.Tensor,
    train_idx: list[int],
    test_idx: list[int],
    device: torch.device,
    backbone: str = "resnet101",
    epochs: int = 8,
    batch_size: int = 32,
    gamma: float = 2.0,
) -> tuple[list[int], dict]:
    """Fine-tune the backbone; return test predictions and the state needed to reload the model."""
    model = FineTuner(backbone).to(device)
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
    loss_fn = FocalCrossEntropy(gamma=gamma, weight=weights)

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_idx))
        total = 0.0
        for start in range(0, len(order), batch_size):
            idx = [train_idx[j] for j in order[start : start + batch_size]]
            batch = images[idx].to(device)
            if (
                torch.rand(1).item() < 0.5
            ):  # horizontal flip only
                batch = torch.flip(batch, dims=[3])
            opt.zero_grad()
            loss = loss_fn(model(batch), y[idx].to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"  epoch {epoch + 1}/{epochs} loss {total / len(train_idx):.3f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(test_idx), 64):
            idx = test_idx[start : start + 64]
            logits = model(images[idx].to(device))
            preds += logits.argmax(dim=1).cpu().tolist()
    state = {"model": {k: v.cpu() for k, v in model.state_dict().items()}}
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


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint saved by ``main``, on any PyTorch version.

    PyTorch 2.6 made ``torch.load`` refuse anything but tensors and plain containers
    (``weights_only=True``). Checkpoints that hold other objects, such as numpy scalars or paths
    from an older version of this script, need the old full unpickling. That can run code from
    the file, so it is only a fallback for checkpoints you trained yourself; PyTorch before 1.13
    has no ``weights_only`` argument and always loads the old way.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 1.13: no weights_only argument
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        log.warning("%s needs full unpickling (weights_only=False); only load checkpoints you trust", path)
        return torch.load(path, map_location="cpu", weights_only=False)


class RoleClassifier:
    """The fine-tuned role model saved by ``main`` (``role_model.pt``), used to classify live people.

    Crops are taken with the same padding ``boxing.person_crops`` used for the training crops.
    """

    def __init__(self, path: Path, device: torch.device | None = None, pad: float = 0.1):
        ckpt = load_checkpoint(path)
        if ckpt["roles"] != ROLES:
            raise ValueError(f"{path} was trained for roles {ckpt['roles']}, expected {ROLES}")
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = FineTuner(ckpt["backbone"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(self.device)
        self.transform = backbone_transform(ckpt["backbone"])
        self.pad = pad

    @torch.no_grad()
    def classify(self, image_bgr: np.ndarray, boxes: list[list[float]]) -> list[str]:
        """Role for each person box ``[x1, y1, x2, y2]``. Empty crops are labelled ``other``."""
        roles = ["other"] * len(boxes)
        height, width = image_bgr.shape[:2]
        tensors, slots = [], []
        for k, box in enumerate(boxes):
            x1, y1, x2, y2 = padded_box(box, self.pad, width, height)
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(Image.fromarray(rgb)))
            slots.append(k)
        if tensors:
            logits = self.model(torch.stack(tensors).to(self.device))
            for k, idx in zip(slots, logits.argmax(dim=1).tolist()):
                roles[k] = ROLES[idx]
        return roles


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("export", type=Path)
    parser.add_argument("crops", type=Path)
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
        "--epochs",
        type=int,
        default=8,
        help="fine-tuning epochs (default: 8)",
    )
    parser.add_argument(
        "--backbone",
        choices=BACKBONES,
        default="resnet101",
        help="timm backbone to fine-tune (default: resnet101)",
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

    ft_name = f"fine-tuned {args.backbone}"
    print(f"\nfine-tuning {args.backbone} end to end...")
    set_seed(args.seed)
    images = preprocess_images([i["path"] for i in items], args.backbone)
    ft_pred, ft_state = finetune_and_evaluate(
        images, y, train_idx, test_idx, device,
        backbone=args.backbone, epochs=args.epochs,
    )
    results[ft_name] = report(ft_name, truth, [ROLES[p] for p in ft_pred])

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
                "backbone": args.backbone,
                "roles": ROLES,
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

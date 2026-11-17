"""Train the keypoint punch classifier on the windows written by ``boxing.pairing``.

    uv run python -m boxing.train_punch_classifier
        [--data data/processed/punch-pairs] [--out runs/punch-classifier]

Train and validation are separate fights, so the score reflects unseen fights. The checkpoint
with the best validation macro F1 is kept, since accuracy alone is dominated by the common
"Jab a la cabeza" class.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from boxing.model import PunchPairClassifier
from boxing.pair_dataset import PunchPairDataset, load_names


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def class_metrics(cm: np.ndarray) -> dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(dim=-1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int) -> dict:
    y, pred = predict(model, loader, device)
    cm = confusion_matrix(y, pred, n_classes)
    m = class_metrics(cm)
    return {"cm": cm, "accuracy": float((y == pred).mean()), "macro_f1": float(m["f1"].mean()), **m}


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


def print_report(result: dict, names: list[str]) -> None:
    print(f"accuracy {result['accuracy']:.3f}  macro F1 {result['macro_f1']:.3f}")
    print(f"{'class':<22}{'prec':>7}{'rec':>7}{'f1':>7}{'n':>6}")
    for i, name in enumerate(names):
        print(f"{name:<22}{result['precision'][i]:>7.3f}{result['recall'][i]:>7.3f}{result['f1'][i]:>7.3f}{int(result['support'][i]):>6}")
    print("confusion matrix (rows = true, columns = predicted):")
    for i, row in enumerate(result["cm"]):
        print(f"{i:>2} " + " ".join(f"{v:>4}" for v in row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/processed/punch-pairs"))
    parser.add_argument("--out", type=Path, default=Path("runs/punch-classifier"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce", help="ce: weighted cross entropy (default); focal: FocalCrossEntropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="focusing parameter for --loss focal")
    parser.add_argument("--patience", type=int, default=15, help="stop after this many epochs without a macro-F1 gain")
    parser.add_argument("--quality", choices=["all", "good"], default="all", help="which paired windows to use")
    parser.add_argument("--no-flip", action="store_true", help="disable mirror augmentation")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    names = load_names(args.data)
    n_classes = len(names)
    train_ds = PunchPairDataset(args.data, "train", args.quality, augment=True, flip=not args.no_flip, seed=args.seed)
    val_ds = PunchPairDataset(args.data, "val", args.quality)
    print(f"device {device}  train {len(train_ds)}  val {len(val_ds)}  quality={args.quality}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    # Inverse-sqrt class weights: the rare classes count more, without letting them dominate.
    counts = np.bincount(train_ds.y, minlength=n_classes).clip(min=1)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    if args.loss == "focal":
        criterion = FocalCrossEntropy(weight=class_weight, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)

    model = PunchPairClassifier(n_classes, args.hidden, args.layers, args.dropout).to(device)
    print(f"parameters {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = 3
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: (e + 1) / warmup if e < warmup else 0.5 * (1 + np.cos(np.pi * (e - warmup) / max(1, args.epochs - warmup))),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(y)
            seen += len(y)
        schedule.step()

        val = evaluate(model, val_loader, device, n_classes)
        history.append({"epoch": epoch, "loss": total / seen, "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"]})
        marker = ""
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, marker = val["macro_f1"], epoch, " *"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {"num_classes": n_classes, "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout},
                    "names": names,
                },
                args.out / "best.pt",
            )
        print(f"epoch {epoch:>3}  loss {total / seen:.4f}  val acc {val['accuracy']:.3f}  macro F1 {val['macro_f1']:.3f}{marker}")
        if epoch - best_epoch >= args.patience:
            print(f"no macro-F1 gain for {args.patience} epochs, stopping")
            break

    # Report the best checkpoint, not the last epoch.
    best = torch.load(args.out / "best.pt", map_location=device)
    model.load_state_dict(best["state_dict"])
    final = evaluate(model, val_loader, device, n_classes)
    print(f"\nbest epoch {best_epoch}")
    print_report(final, names)
    (args.out / "metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "accuracy": final["accuracy"],
                "macro_f1": final["macro_f1"],
                "per_class": {
                    n: {k: float(final[k][i]) for k in ("precision", "recall", "f1", "support")} for i, n in enumerate(names)
                },
                "confusion_matrix": final["cm"].tolist(),
                "history": history,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

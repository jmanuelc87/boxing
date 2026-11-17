"""Train EPAM-Net on RGB clips and keypoints from the windows written by ``boxing.pairing``.

    uv run python -m boxing.train_punch_classifier [--epochs 30] [--batch-size 8] [--lr 0.01]
    uv run python -m boxing.train_punch_classifier --test      # smoke test, see below

Both streams are trained at once with a classification loss on each head (the attended RGB head
and the pose head), summed. Predictions fuse the two streams' softmax scores. The optimiser is SGD
with momentum and a cosine schedule with warm-up, as in the reference implementation.

Train and validation are separate fights, so the score reflects unseen fights. The checkpoint with
the best validation macro F1 of the fused prediction is kept, since accuracy alone is dominated by
the common "Jab a la cabeza" class.

The best model is saved with ``save_pretrained`` (``best/``: ``model.safetensors``, ``config.json``,
the model code and a model card) and, unless ``--no-push``, uploaded to the Hugging Face Hub repo
``--hub-repo`` each time the validation score improves. A resumable checkpoint (weights, optimiser,
scheduler, history) is uploaded to ``checkpoints/last`` after every epoch. Both need ``HF_TOKEN``
(read from ``.env``). A repo that does not exist yet is created private unless ``--hub-public``.

``--test`` runs every stage of the real pipeline on a handful of windows: data loading with
augmentation, forward and backward passes, the scheduler, validation, the checkpoint, reloading it,
and the metrics file. It never pushes or logs to wandb. It uses 2 short epochs, a different output folder, and the numbers it prints
mean nothing. Use it to check a change in seconds before starting a long run.

The model is large for a laptop. Measured here, a step of 4 clips took 1.6 s on the CPU and 3.4 s on
MPS, so the default device is CUDA if present and the CPU otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from huggingface_hub import HfApi, ModelCard, ModelCardData
from torch.utils.data import DataLoader
from tqdm import tqdm

from boxing.epam_dataset import EPAMDataset
from boxing.losses import FocalCrossEntropy
from boxing.model import EPAMNet, EPAMNetConfig
from boxing.pair_dataset import load_names

TEST_EPOCHS = 2
TEST_BATCH = 4
TEST_TRAIN_WINDOWS = 24
TEST_VAL_WINDOWS = 16
TEST_WORKERS = 2
HUB_REPO = "jmanuelc87/boxing-punch-epam-net"
MODEL_OVERRIDES: dict = {}  # extra EPAMNetConfig fields; the tests shrink the network with this


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def score(y: np.ndarray, pred: np.ndarray, n_classes: int) -> dict:
    cm = confusion_matrix(y, pred, n_classes)
    m = class_metrics(cm)
    return {"cm": cm, "accuracy": float((y == pred).mean()), "macro_f1": float(m["f1"].mean()), **m}


def print_report(result: dict, names: list[str]) -> None:
    print(f"accuracy {result['accuracy']:.3f}  macro F1 {result['macro_f1']:.3f}")
    print(f"{'class':<22}{'prec':>7}{'rec':>7}{'f1':>7}{'n':>6}")
    for i, name in enumerate(names):
        print(f"{name:<22}{result['precision'][i]:>7.3f}{result['recall'][i]:>7.3f}{result['f1'][i]:>7.3f}{int(result['support'][i]):>6}")
    print("confusion matrix (rows = true, columns = predicted):")
    for i, row in enumerate(result["cm"]):
        print(f"{i:>2} " + " ".join(f"{v:>4}" for v in row))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int, rgb_weight: float) -> dict:
    """Scores for the fused prediction and for each stream alone."""
    model.eval()
    ys, fused, rgb, pose = [], [], [], []
    for images, keypoints, y in loader:
        out = model(images.to(device), keypoints.to(device))
        fused.append(out.probabilities(rgb_weight).argmax(-1).cpu().numpy())
        rgb.append(out.rgb_logits.argmax(-1).cpu().numpy())
        pose.append(out.pose_logits.argmax(-1).cpu().numpy())
        ys.append(y.numpy())
    y = np.concatenate(ys)
    return {
        "fused": score(y, np.concatenate(fused), n_classes),
        "rgb": score(y, np.concatenate(rgb), n_classes),
        "pose": score(y, np.concatenate(pose), n_classes),
    }


def lr_factor(step: int, total_steps: int, warmup_steps: int, floor: float = 0.01) -> float:
    """Linear warm-up, then cosine decay to ``floor`` of the base learning rate."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def model_card(names: list[str], args: argparse.Namespace, best_epoch: int, result: dict, n_params: int) -> str:
    """README.md for the Hub repo: what the model is, how to load it, and the validation scores."""
    fused = result["fused"]
    rows = "\n".join(
        f"| {n} | {fused['precision'][i]:.3f} | {fused['recall'][i]:.3f} | {fused['f1'][i]:.3f} | {int(fused['support'][i])} |"
        for i, n in enumerate(names)
    )
    front = ModelCardData(
        library_name="transformers",
        pipeline_tag="video-classification",
        tags=["boxing", "action-recognition", "pose", "two-stream", "epam-net"],
        metrics=["accuracy", "f1"],
    ).to_yaml()
    return f"""---
{front}
---

# Boxing punch classifier (EPAM-Net)

Classifies a 16-frame window of a boxing round into one of {len(names)} punch classes. It is a
reimplementation in plain PyTorch of EPAM-Net (Nady et al., Neurocomputing 2025,
[arXiv:2408.05421](https://arxiv.org/abs/2408.05421)): an RGB stream and a pose stream, with the pose
features guiding the RGB stream's attention. The two streams' softmax scores are fused at inference.
{n_params / 1e6:.2f}M parameters.

## Input

* `images`: `(B, 16, 3, 224, 224)`, one square crop around both boxers, ImageNet-normalised RGB.
* `keypoints`: `(B, 2, 16, 17, 3)`, COCO joints `(x, y, score)` of the attacker and the opponent, with
  `x` and `y` in `[0, 1]` of the same crop.

## Usage

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("{args.hub_repo}", trust_remote_code=True).eval()
with torch.no_grad():
    out = model(images, keypoints)
probs = out.probabilities(model.config.rgb_weight)   # fused scores, (B, {len(names)})
label = model.config.id2label[probs.argmax(-1)[0].item()]
```

The model code is in `model.py` in this repo, so `trust_remote_code=True` runs that file. Read it
first if you do not trust the source.

## Validation results

The best epoch ({best_epoch}) by fused macro F1 on the validation split. Train and validation are
separate fights, so these scores are on fights the model did not see. Accuracy alone is dominated by
the most common class, so the macro F1 is the number to watch.

| | accuracy | macro F1 |
|---|---|---|
| fused (RGB + pose) | {fused['accuracy']:.3f} | {fused['macro_f1']:.3f} |
| RGB stream alone | {result['rgb']['accuracy']:.3f} | {result['rgb']['macro_f1']:.3f} |
| pose stream alone | {result['pose']['accuracy']:.3f} | {result['pose']['macro_f1']:.3f} |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
{rows}

## Training

SGD (momentum {args.momentum}, weight decay {args.weight_decay}), learning rate {args.lr} with a
{args.warmup_epochs:g}-epoch warm-up then cosine decay, batch size {args.batch_size},
{args.loss} loss (label smoothing {args.label_smoothing}), inverse-square-root class weights,
seed {args.seed}. Windows quality filter: `{args.quality}`.

## Limitations

Trained on a small set of annotated fights, so it may not transfer to other cameras, rings or
styles. The pose input comes from a separate keypoint detector, and its errors carry through. The
validation split was used to pick the checkpoint, so the scores above are slightly optimistic.
"""


def push_folder(api: HfApi | None, repo: str, folder: Path, path_in_repo: str, message: str) -> None:
    """Upload a folder to the Hub. A network failure is reported, not fatal: training goes on."""
    if api is None:
        return
    try:
        api.upload_folder(repo_id=repo, folder_path=str(folder), path_in_repo=path_in_repo, commit_message=message)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: upload of {folder} to {repo}:{path_in_repo} failed ({exc})")


def train_epoch(
    model, loader, criterion, optimizer, scheduler, device, epoch: int, epochs: int, log_every: int = 0
) -> float:
    """Train one epoch. With ``log_every`` > 0 and a live wandb run, push the loss and LR every that many steps."""
    model.train()
    total, seen = 0.0, 0
    bar = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False)
    for images, keypoints, y in bar:
        images, keypoints, y = images.to(device), keypoints.to(device), y.to(device)
        out = model(images, keypoints)
        loss = criterion(out.rgb_logits, y) + criterion(out.pose_logits, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(y)
        seen += len(y)
        bar.set_postfix(loss=f"{total / seen:.3f}")
        if log_every and wandb.run is not None and scheduler.last_epoch % log_every == 0:
            wandb.log(
                {"step/loss": loss.item(), "step/lr": optimizer.param_groups[0]["lr"], "epoch": epoch},
                step=scheduler.last_epoch,
            )
    return total / seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, default=Path("data/processed/punch-pairs"), help="output of boxing.pairing")
    parser.add_argument("--src", type=Path, default=Path("data/processed/punch-seq"), help="clips and frames from boxing.preprocess")
    parser.add_argument("--out", type=Path, default=None, help="default: runs/epam-net, or runs/epam-test with --test")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce", help="ce: weighted cross entropy; focal: FocalCrossEntropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--rgb-weight", type=float, default=0.5, help="weight of the RGB stream when fusing scores")
    parser.add_argument("--patience", type=int, default=10, help="stop after this many epochs without a macro-F1 gain")
    parser.add_argument("--quality", choices=["all", "good"], default="all", help="which paired windows to use")
    parser.add_argument("--workers", type=int, default=4, help="data loading processes")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available, else cpu)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test", action="store_true", help="smoke test: the whole pipeline on a few windows")
    parser.add_argument("--wandb-project", default="boxing", help="Weights & Biases project (needs WANDB_API_KEY)")
    parser.add_argument("--wandb-name", default=None, help="run name (default: wandb picks one)")
    parser.add_argument("--log-every", type=int, default=100, help="push loss and LR to wandb every N steps")
    parser.add_argument("--no-wandb", action="store_true", help="do not log to Weights & Biases")
    parser.add_argument("--hub-repo", default=HUB_REPO, help="Hugging Face repo that receives the best model and checkpoints")
    parser.add_argument("--hub-public", action="store_true", help="create the repo public if it does not exist (default: private)")
    parser.add_argument("--no-push", action="store_true", help="do not upload to the Hugging Face Hub")
    args = parser.parse_args()
    load_dotenv()  # HF_TOKEN and WANDB_API_KEY from .env

    if args.test:
        args.epochs, args.batch_size = TEST_EPOCHS, TEST_BATCH
        args.workers = min(args.workers, TEST_WORKERS)
        args.patience = args.epochs
        args.no_wandb = args.no_push = True  # smoke-test numbers are meaningless; keep them out of wandb and the Hub
    out_dir = args.out or Path("runs/epam-test" if args.test else "runs/epam-net")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    names = load_names(args.pairs)
    n_classes = len(names)
    train_ds = EPAMDataset(args.pairs, args.src, "train", args.quality, augment=True, limit=TEST_TRAIN_WINDOWS if args.test else None)
    val_ds = EPAMDataset(args.pairs, args.src, "val", args.quality, limit=TEST_VAL_WINDOWS if args.test else None)
    mode = "TEST MODE (a few windows, metrics are meaningless)" if args.test else "full run"
    print(f"{mode}\ndevice {device}  train {len(train_ds)}  val {len(val_ds)}  quality={args.quality}  out={out_dir}")

    loader_kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) >= args.batch_size, **loader_kw
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, **loader_kw)

    # Inverse-sqrt class weights from the whole training split: rare classes count more without dominating.
    counts = np.bincount(np.load(args.pairs / "train.npz")["y"], minlength=n_classes).clip(min=1)
    weights = 1.0 / np.sqrt(counts)
    class_weight = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    if args.loss == "focal":
        criterion = FocalCrossEntropy(weight=class_weight, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)

    config = EPAMNetConfig(
        num_classes=n_classes,
        rgb_weight=args.rgb_weight,
        id2label=dict(enumerate(names)),
        label2id={n: i for i, n in enumerate(names)},
        **MODEL_OVERRIDES,
    )
    model = EPAMNet(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters {n_params / 1e6:.2f}M")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_epochs * steps_per_epoch))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total_steps, warmup_steps))

    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir, ckpt_dir = out_dir / "best", out_dir / "checkpoint"
    api = None
    if not args.no_push:
        api = HfApi()
        api.create_repo(args.hub_repo, private=not args.hub_public, exist_ok=True)
        print(f"pushing to https://huggingface.co/{args.hub_repo}")
    run = None
    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            job_type="train",
        )
    best_f1, best_epoch, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, args.epochs, args.log_every
        )
        val = evaluate(model, val_loader, device, n_classes, args.rgb_weight)
        f1 = val["fused"]["macro_f1"]
        row = {
            "epoch": epoch,
            "loss": loss,
            "lr": optimizer.param_groups[0]["lr"],
            "val_acc": val["fused"]["accuracy"],
            "val_macro_f1": f1,
            "val_macro_f1_rgb": val["rgb"]["macro_f1"],
            "val_macro_f1_pose": val["pose"]["macro_f1"],
            "seconds": time.time() - started,
        }
        history.append(row)
        if run is not None:
            wandb.log(
                {
                    "epoch/train_loss": loss,
                    "epoch/lr": row["lr"],
                    "epoch/val_acc": row["val_acc"],
                    "epoch/val_macro_f1": f1,
                    "epoch/val_macro_f1_rgb": row["val_macro_f1_rgb"],
                    "epoch/val_macro_f1_pose": row["val_macro_f1_pose"],
                    "epoch/seconds": row["seconds"],
                    "epoch": epoch,
                },
                step=scheduler.last_epoch,
            )
        marker = ""
        if f1 > best_f1:
            best_f1, best_epoch, marker = f1, epoch, " *"
            model.save_pretrained(best_dir)
            push_folder(api, args.hub_repo, best_dir, ".", f"Best model: epoch {epoch}, val macro F1 {f1:.3f}")
        print(
            f"epoch {epoch:>3}  loss {loss:.4f}  val acc {row['val_acc']:.3f}  macro F1 {f1:.3f} "
            f"(rgb {row['val_macro_f1_rgb']:.3f}, pose {row['val_macro_f1_pose']:.3f})  {row['seconds']:.0f}s{marker}"
        )
        model.save_pretrained(ckpt_dir)
        torch.save(
            {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "history": history,
            },
            ckpt_dir / "training_state.pt",
        )
        push_folder(api, args.hub_repo, ckpt_dir, "checkpoints/last", f"Checkpoint: epoch {epoch}")
        if epoch - best_epoch >= args.patience:
            print(f"no macro-F1 gain for {args.patience} epochs, stopping")
            break

    # Report the best checkpoint, reloaded from disk into a fresh model, not the last epoch.
    reloaded = EPAMNet.from_pretrained(best_dir).to(device)
    final = evaluate(reloaded, val_loader, device, n_classes, reloaded.config.rgb_weight)
    if args.test and not math.isclose(final["fused"]["macro_f1"], best_f1, abs_tol=1e-6):
        raise RuntimeError(f"reloaded checkpoint scores {final['fused']['macro_f1']:.4f}, expected {best_f1:.4f}")
    print(f"\nbest epoch {best_epoch}" + ("  (checkpoint reload verified)" if args.test else ""))
    print("fused prediction:")
    print_report(final["fused"], names)
    print(f"RGB stream alone: macro F1 {final['rgb']['macro_f1']:.3f}   pose stream alone: macro F1 {final['pose']['macro_f1']:.3f}")

    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "test_mode": args.test,
                "best_epoch": best_epoch,
                "accuracy": final["fused"]["accuracy"],
                "macro_f1": final["fused"]["macro_f1"],
                "macro_f1_rgb": final["rgb"]["macro_f1"],
                "macro_f1_pose": final["pose"]["macro_f1"],
                "per_class": {
                    n: {k: float(final["fused"][k][i]) for k in ("precision", "recall", "f1", "support")}
                    for i, n in enumerate(names)
                },
                "confusion_matrix": final["fused"]["cm"].tolist(),
                "history": history,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if run is not None:
        run.summary.update(
            {
                "best_epoch": best_epoch,
                "final/accuracy": final["fused"]["accuracy"],
                "final/macro_f1": final["fused"]["macro_f1"],
                "final/macro_f1_rgb": final["rgb"]["macro_f1"],
                "final/macro_f1_pose": final["pose"]["macro_f1"],
            }
        )
        run.finish()
    # The model card is written last, from the reloaded checkpoint's final scores, then pushed with the metrics.
    (best_dir / "README.md").write_text(model_card(names, args, best_epoch, final, n_params), encoding="utf-8")
    print(f"wrote model card {best_dir / 'README.md'}")
    if api is not None:
        push_folder(api, args.hub_repo, best_dir, ".", f"Model card and final scores: best epoch {best_epoch}")
        try:
            api.upload_file(
                repo_id=args.hub_repo, path_or_fileobj=str(out_dir / "metrics.json"), path_in_repo="metrics.json",
                commit_message="Final metrics",
            )
            print(f"pushed https://huggingface.co/{args.hub_repo}")
        except Exception as exc:  # noqa: BLE001
            print(f"warning: upload of metrics.json failed ({exc})")
    print(f"saved {best_dir} and {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

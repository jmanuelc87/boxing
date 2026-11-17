import json
import sys

import cv2
import numpy as np
import pytest
import torch

from boxing import train_punch_classifier as train
from boxing.epam_dataset import (
    IMAGE_SIZE,
    EPAMDataset,
    crop_square,
    jitter_crop,
    keypoints_in_crop,
    mirror,
)
from boxing.model import EPAMNet
from boxing.pair_dataset import FLIP_PERM

W, H, T = 128, 96, 16


@pytest.fixture
def data(tmp_path):
    """Tiny pairs + clips folders with real JPEGs: train has 6 windows (one without an attacker), val 4."""
    pairs, src = tmp_path / "pairs", tmp_path / "seq"
    pairs.mkdir()
    rng = np.random.default_rng(0)
    for split, n in (("train", 6), ("val", 4)):
        X = np.zeros((n, 2, T, 17, 3), dtype=np.float32)
        X[..., 0] = rng.uniform(40, 90, X.shape[:-1])
        X[..., 1] = rng.uniform(25, 70, X.shape[:-1])
        X[..., 2] = 0.9
        y = (np.arange(n) % 8).astype(np.int64)
        np.savez(pairs / f"{split}.npz", X=X, y=y)
        meta, records = [], []
        for i in range(n):
            task, start = f"task_{i % 2}", i * 100
            frames = []
            for t in range(T):
                rel = f"images/{task}/{start + t:06d}.jpg"
                (src / rel).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(src / rel), rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
                frames.append({"image": rel, "class_labels": [], "boxes": []})
            quality = "no_attacker" if (split == "train" and i == 1) else "good"
            meta.append({"task": task, "track": i, "label": int(y[i]), "start": start, "quality": quality})
            records.append({"task": task, "track": i, "label": int(y[i]), "start": start, "width": W, "height": H, "frames": frames})
        for folder, rows in ((pairs, meta), (src, records)):
            (folder / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (pairs / "labels.json").write_text(json.dumps({"names": [f"class{i}" for i in range(8)]}))
    return pairs, src


def test_item_shapes_and_keypoints_stay_inside_the_crop(data):
    pairs, src = data
    ds = EPAMDataset(pairs, src, "train")
    images, kp, label = ds[0]
    assert images.shape == (T, 3, IMAGE_SIZE, IMAGE_SIZE) and images.dtype == torch.float32
    assert kp.shape == (2, T, 17, 3)
    assert label.dtype == torch.int64
    assert 0.0 <= kp[..., :2].min() and kp[..., :2].max() <= 1.0


def test_windows_without_an_attacker_are_dropped_and_limit_spreads_evenly(data):
    pairs, src = data
    assert len(EPAMDataset(pairs, src, "train")) == 5
    assert len(EPAMDataset(pairs, src, "train", quality="good")) == 5
    limited = EPAMDataset(pairs, src, "train", limit=3)
    assert len(limited) == 3 and limited.indices[0] == 0 and limited.indices[-1] == 5


def test_misordered_metadata_is_rejected(data):
    pairs, src = data
    rows = (src / "val.jsonl").read_text().splitlines()
    (src / "val.jsonl").write_text("\n".join(reversed(rows)) + "\n")
    with pytest.raises(ValueError):
        EPAMDataset(pairs, src, "val")


def test_crop_square_without_keypoints_is_centred_and_jitter_stays_inside():
    assert crop_square(np.zeros((2, T, 17, 3), np.float32), W, H) == ((W - H) // 2, 0, H)
    rng = np.random.default_rng(0)
    for _ in range(200):
        x0, y0, side = jitter_crop(20, 10, 60, W, H, rng)
        assert 0 <= x0 and x0 + side <= W and 0 <= y0 and y0 + side <= H


def test_mirror_swaps_hands_flips_x_and_is_an_involution():
    rng = np.random.default_rng(1)
    images = rng.integers(0, 255, (T, 8, 8, 3), dtype=np.uint8)
    kp = np.zeros((2, T, 17, 3), np.float32)
    kp[0, :, 5] = [0.2, 0.4, 0.9]  # left shoulder of the attacker
    flipped_images, flipped = mirror(images, kp)
    np.testing.assert_allclose(flipped[0, :, 6], np.tile([0.8, 0.4, 0.9], (T, 1)))  # right shoulder, x mirrored
    assert not flipped[0, :, 5].any() and not flipped[1].any()  # missing joints stay at zero
    np.testing.assert_array_equal(flipped_images, images[:, :, ::-1])
    back_images, back = mirror(flipped_images, flipped)
    np.testing.assert_allclose(back, kp)
    assert [FLIP_PERM[i] for i in FLIP_PERM] == list(range(17))


def test_keypoints_in_crop_rescales_and_zeroes_missing_joints():
    win = np.zeros((2, 1, 17, 3), np.float32)
    win[0, 0, 0] = [60, 50, 0.8]
    kp = keypoints_in_crop(win, x0=40, y0=30, side=40)
    np.testing.assert_allclose(kp[0, 0, 0], [0.5, 0.5, 0.8])
    assert not kp[1].any()


def test_augmented_item_has_the_same_shapes(data):
    pairs, src = data
    torch.manual_seed(0)
    images, kp, _ = EPAMDataset(pairs, src, "train", augment=True)[0]
    assert images.shape == (T, 3, IMAGE_SIZE, IMAGE_SIZE) and kp.shape == (2, T, 17, 3)


def test_training_smoke_test_runs_every_stage(data, tmp_path, monkeypatch):
    pairs, src = data
    small = dict(
        rgb_blocks=(1, 1, 1, 1), rgb_channels=(8, 16, 24, 32), pose_blocks=(1, 1, 1), pose_channels=(8, 16, 24)
    )
    monkeypatch.setattr(train, "EPAMNet", lambda num_classes: EPAMNet(num_classes=num_classes, **small))
    out = tmp_path / "run"
    monkeypatch.setattr(
        sys, "argv",
        ["train", "--test", "--pairs", str(pairs), "--src", str(src), "--out", str(out), "--workers", "0"],
    )
    train.main()
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["test_mode"] is True and len(metrics["history"]) == train.TEST_EPOCHS
    assert (out / "best.pt").exists()
    assert {"loss", "lr", "val_macro_f1", "val_macro_f1_rgb", "val_macro_f1_pose"} <= set(metrics["history"][0])


def test_lr_factor_warms_up_then_decays():
    f = lambda s: train.lr_factor(s, total_steps=100, warmup_steps=10)
    assert f(0) < f(9) <= 1.0
    assert f(10) == pytest.approx(1.0)
    assert f(99) < f(50) < f(10)
    assert f(99) >= 0.01

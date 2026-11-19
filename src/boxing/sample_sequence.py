"""Draw one random EPAM-Net sample per class into one image-sequence GIF.

Each frame is a 4-panel strip (player 1, player 2, player 1 with keypoints, player 2 with keypoints)
with a black footer showing the frame counter and class label. Every player is cropped around their
own keypoints over the window. The frames of all samples are concatenated into one GIF; no PNG frames are written.

    uv run python -m boxing.sample_sequence --split train --seed 0 --out media/sample.gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from boxing.epam_dataset import IMAGE_SIZE, VALID_CONF, EPAMDataset, crop_square, keypoints_in_crop

# COCO-17 limbs.
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
COLORS = [(60, 60, 255), (255, 140, 40)]  # BGR: player 1 red, player 2 blue


def draw_pose(img: np.ndarray, kp: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Draw one boxer's ``(17, 3)`` keypoints (x, y in [0, 1] of ``img``) on a copy of ``img``."""
    out = img.copy()
    pts = (kp[:, :2] * img.shape[0]).astype(int)
    ok = kp[:, 2] > VALID_CONF
    for a, b in SKELETON:
        if ok[a] and ok[b]:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
    for p, v in zip(pts, ok):
        if v:
            cv2.circle(out, tuple(p), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, tuple(p), 4, color, 1, cv2.LINE_AA)
    return out


def label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_sample(ds: EPAMDataset, k: int, cls: str, src: Path) -> list[Image.Image]:
    """Return the RGB frame strips for window ``k`` of ``ds``."""
    rec, win = ds.records[k], ds.X[k]  # win: (2, T, 17, 3) in pixels

    # One crop per player, fixed over the window so the motion is visible.
    crops = [crop_square(win[p : p + 1], rec["width"], rec["height"]) for p in range(2)]
    kps = [keypoints_in_crop(win[p], *crops[p]) for p in range(2)]  # (T, 17, 3)

    frames = []
    for t, f in enumerate(rec["frames"]):
        img = cv2.imread(str(src / f["image"]))
        if img is None:
            raise FileNotFoundError(src / f["image"])
        plain = []
        for p in range(2):
            x0, y0, side = crops[p]
            plain.append(cv2.resize(img[y0 : y0 + side, x0 : x0 + side], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA))
        posed = [draw_pose(plain[p], kps[p][t], COLORS[p]) for p in range(2)]
        panels = [label(plain[0].copy(), "player 1"), label(plain[1].copy(), "player 2"),
                  label(posed[0], "player 1 + keypoints"), label(posed[1], "player 2 + keypoints")]
        strip = np.hstack(panels)
        footer = np.zeros((28, strip.shape[1], 3), dtype=np.uint8)  # black background for the caption
        cv2.putText(footer, f"frame {t + 1}/{len(rec['frames'])}  {cls}", (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        strip = np.vstack([strip, footer])
        frames.append(Image.fromarray(strip[..., ::-1]))
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=Path("data/processed/punch-pairs"))
    ap.add_argument("--src", type=Path, default=Path("data/processed/punch-seq"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--quality", default="good", choices=["all", "good"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("media/sample.gif"), help="output GIF path")
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()

    ds = EPAMDataset(args.pairs, args.src, args.split, args.quality)
    names = json.loads((args.pairs / "labels.json").read_text())["names"]
    rng = np.random.default_rng(args.seed)
    item_labels = np.array([ds.labels[k] for k in ds.indices])

    # One random dataset item per class, in label order.
    picks = []
    for c, name in enumerate(names):
        pool = np.flatnonzero(item_labels == c)
        if len(pool) == 0:
            print(f"no windows for class {name}, skipping")
            continue
        picks.append(int(rng.choice(pool)))
    if not picks:
        raise ValueError("no windows found in the split")

    frames: list[Image.Image] = []
    for i in picks:
        k = int(ds.indices[i])
        rec = ds.records[k]
        cls = names[int(ds.labels[k])]
        print(f"sample {i} (window {k}) task={rec['task']} start={rec['start']} label={cls}")
        frames.extend(render_sample(ds, k, cls, args.src))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=1000 // args.fps, loop=0)
    print(f"wrote {len(frames)} frames from {len(picks)} samples to {args.out}")


if __name__ == "__main__":
    main()

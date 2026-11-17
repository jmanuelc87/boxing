"""Pair each punch sequence with its opponent using the pose keypoints.

Reads the punch-seq clips (``train.jsonl`` / ``val.jsonl`` written by
``boxing.preprocess``) and the per-frame pose JSONs (written by
``boxing.keypoints``). For every 16-frame window it finds the attacking boxer
and the opponent in each frame and writes one array per split:

    X[N, 2, T, 17, 3]   row 0 = attacker, row 1 = opponent
                        last axis = (x, y, score) in source pixels
                        missing people are zero-filled with score 0

Attacker selection: on labeled frames the punch box is used. Its center is
matched to the person whose box contains it, with ties broken by distance to
the previous attacker position. Unlabeled frames follow the nearest person to
the attacker's last known center, forward and backward from the first labeled
frame. The opponent is the other person closest to the attacker.

Outputs in ``--dst``:
    train.npz, val.npz    X (float32) and y (int64, class id)
    train.jsonl, val.jsonl  one metadata row per window, aligned with X
    labels.json           copied from ``--src``

Usage:
    python -m boxing.pairing --src data/processed/punch-seq \\
        --keypoints data/processed/punch-keypoints-boxers/images \\
        --dst data/processed/punch-pairs
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

log = logging.getLogger("boxing.pairing")

NUM_KEYPOINTS = 17
SPLITS = ("train", "val")


def load_people(kp_root: Path, image: str) -> list[dict]:
    """People in one frame's keypoint JSON; empty if the file is missing."""
    rel = Path(image).relative_to("images").with_suffix(".json")
    path = kp_root / rel
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    people = []
    for p in data.get("people", []):
        kps = np.array(
            [[k["x"], k["y"], k["score"]] for k in p["keypoints"]], dtype=np.float32
        ).reshape(-1, 3)
        if kps.shape[0] != NUM_KEYPOINTS:
            continue
        box = np.asarray(p["box"], dtype=np.float32)
        people.append({"box": box, "center": _center(box), "kps": kps})
    return people


def _center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)


def _glove_px(frame: dict, width: int, height: int) -> np.ndarray | None:
    """Center of the labeled punch box in pixels, or None for unlabeled frames."""
    if not frame["class_labels"] or not frame["boxes"]:
        return None
    cx, cy, _, _ = frame["boxes"][0]
    return np.array([cx * width, cy * height], dtype=np.float32)


def _inside(box: np.ndarray, pt: np.ndarray) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def pick_attacker(
    people: list[dict], glove: np.ndarray | None, ref: np.ndarray | None
) -> tuple[int | None, str]:
    """Index of the attacker in one frame and how it was chosen."""
    if not people:
        return None, "no_people"
    if glove is not None:
        inside = [i for i, p in enumerate(people) if _inside(p["box"], glove)]
        if len(inside) == 1:
            return inside[0], "glove_in_box"
        if len(inside) > 1:
            pool, how = inside, "glove_multi"
        else:
            pool, how = list(range(len(people))), "glove_nearest"
        anchor = ref if ref is not None else glove
        return min(pool, key=lambda i: np.linalg.norm(people[i]["center"] - anchor)), how
    if ref is not None:
        return min(range(len(people)), key=lambda i: np.linalg.norm(people[i]["center"] - ref)), "tracked"
    return None, "unanchored"


def pick_opponent(people: list[dict], attacker: int | None) -> int | None:
    if attacker is None:
        return None
    others = [i for i in range(len(people)) if i != attacker]
    if not others:
        return None
    ref = people[attacker]["center"]
    return min(others, key=lambda i: np.linalg.norm(people[i]["center"] - ref))


def pair_record(rec: dict, kp_root: Path) -> tuple[np.ndarray, dict]:
    """Build the (2, T, 17, 3) window and its metadata for one clip."""
    frames = rec["frames"]
    T = len(frames)
    W, H = rec["width"], rec["height"]
    people = [load_people(kp_root, f["image"]) for f in frames]
    gloves = [_glove_px(f, W, H) for f in frames]

    attacker: list[int | None] = [None] * T
    how = ["none"] * T

    labeled = [t for t in range(T) if gloves[t] is not None]
    if labeled:
        anchor = labeled[0]
        # Seed the anchor frame, then walk outwards in both directions.
        attacker[anchor], how[anchor] = pick_attacker(people[anchor], gloves[anchor], None)
        for order in (range(anchor + 1, T), range(anchor - 1, -1, -1)):
            ref = None
            a = attacker[anchor]
            if a is not None:
                ref = people[anchor][a]["center"]
            for t in order:
                idx, reason = pick_attacker(people[t], gloves[t], ref)
                attacker[t], how[t] = idx, reason
                if idx is not None:
                    ref = people[t][idx]["center"]

    window = np.zeros((2, T, NUM_KEYPOINTS, 3), dtype=np.float32)
    opponent_missing = 0
    for t in range(T):
        a = attacker[t]
        if a is None:
            continue
        window[0, t] = people[t][a]["kps"]
        o = pick_opponent(people[t], a)
        if o is None:
            opponent_missing += 1
        else:
            window[1, t] = people[t][o]["kps"]

    n_attacker = sum(a is not None for a in attacker)
    if n_attacker == 0:
        quality = "no_attacker"
    elif n_attacker == T and opponent_missing == 0:
        quality = "good"
    else:
        quality = "partial"

    meta = {
        "task": rec["task"],
        "track": rec["track"],
        "label": rec["label"],
        "start": rec["start"],
        "width": W,
        "height": H,
        "n_labeled": len(labeled),
        "attacker_frames": n_attacker,
        "opponent_missing_frames": opponent_missing,
        "quality": quality,
        "attacker_method": dict(Counter(how)),
    }
    return window, meta


def process_split(src: Path, kp_root: Path, split: str, dst: Path) -> Counter:
    records = [json.loads(line) for line in (src / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    T = len(records[0]["frames"]) if records else 0
    X = np.zeros((len(records), 2, T, NUM_KEYPOINTS, 3), dtype=np.float32)
    y = np.zeros(len(records), dtype=np.int64)
    metas = []
    quality = Counter()
    for i, rec in enumerate(records):
        window, meta = pair_record(rec, kp_root)
        X[i] = window
        y[i] = rec["label"]
        metas.append(meta)
        quality[meta["quality"]] += 1

    np.savez_compressed(dst / f"{split}.npz", X=X, y=y)
    with (dst / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
        for m in metas:
            fh.write(json.dumps(m) + "\n")
    log.info("%s: %d windows %s", split, len(records), dict(quality))
    return quality


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m boxing.pairing", description="Pair each punch sequence with its opponent using pose keypoints."
    )
    parser.add_argument("--src", type=Path, default=Path("data/processed/punch-seq"), help="punch-seq directory with train/val jsonl and labels.json")
    parser.add_argument("--keypoints", type=Path, default=Path("data/processed/punch-keypoints-boxers/images"), help="root of the per-frame keypoint JSONs")
    parser.add_argument("--dst", type=Path, default=Path("data/processed/punch-pairs"), help="output directory (created if missing)")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    args = parser.parse_args()

    if not args.src.is_dir():
        parser.error(f"--src is not a directory: {args.src}")
    if not args.keypoints.is_dir():
        parser.error(f"--keypoints is not a directory: {args.keypoints}")

    args.dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.src / "labels.json", args.dst / "labels.json")
    for split in args.splits:
        process_split(args.src, args.keypoints, split, args.dst)


if __name__ == "__main__":
    main()

"""Pair each punch sequence with its opponent, tracking the boxers with DeepSORT and Re-ID.

Reads the punch-seq clips (``train.jsonl`` / ``val.jsonl`` written by
``boxing.preprocess``), the frames they point to, and the per-frame pose JSONs (written by
``boxing.keypoints``). For every 16-frame window it finds the attacking boxer and the opponent
in each frame and writes one array per split:

    X[N, 2, T, 17, 3]   row 0 = attacker, row 1 = opponent
                        last axis = (x, y, score) in source pixels
                        missing people are zero-filled with score 0

Identity comes from a tracker, not from "the nearest person to where the attacker was". Each
person box is cropped from the frame and embedded by a Re-ID network; DeepSORT (Kalman motion
model + appearance matching) gives every person a track id that survives clinches, where two
boxers overlap and nearest-centre matching swaps them.

    1. Track every person through the window (forward, one DeepSORT per clip).
    2. The attacker is the person whose box contains the labeled punch box. Their track id is
       followed through the window; the opponent's track id likewise.
    3. The punch box seeds the attacker on the anchor frame and is then ignored while the track
       exists. Re-binding to the punch box on every labelled frame (``--glove-rebind``) caused
       most identity swaps in validation, so it is off by default.
    4. If a track is lost, the nearest person to the last known position is used as a fallback.

Outputs in ``--dst`` are the same as before (consumers of ``punch-pairs`` need no change):
    train.npz, val.npz    X (float32) and y (int64, class id)
    train.jsonl, val.jsonl  one metadata row per window, aligned with X
    labels.json           copied from ``--src``

The metadata gains ``attacker_tracks`` (distinct track ids the attacker was assigned; 1 means a
stable identity) and ``attacker_method`` now counts ``track`` and ``nearest`` (fallback) frames.

Usage:
    python -m boxing.pairing --src data/processed/punch-seq \\
        --keypoints data/processed/punch-keypoints-boxers/images \\
        --dst data/processed/punch-pairs [--embedder mobilenet] [--workers N]

``--workers N`` pairs clips in N processes (``0`` = every core); each process loads its own
embedder, so on a GPU keep N small. The default ``torchreid`` embedder (OSNet, a person Re-ID
model) needs ``torchreid`` installed; ``--embedder mobilenet`` uses the generic embedder that
ships with ``deep-sort-realtime`` and needs nothing extra.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm

log = logging.getLogger("boxing.pairing")

NUM_KEYPOINTS = 17
SPLITS = ("train", "val")
EMBEDDERS = ("mobilenet", "torchreid")
MIN_CROP = 2  # pixels; smaller crops cannot be embedded


@dataclass(frozen=True)
class TrackConfig:
    """DeepSORT and Re-ID settings (picklable, so it can go to worker processes)."""

    embedder: str = "torchreid"
    cpu: bool = False
    max_age: int = 16  # frames a track may go unseen; a whole window, so a boxer can be hidden and come back
    n_init: int = 2  # hits before a track is confirmed
    max_cosine_distance: float = 0.3
    max_iou_distance: float = 0.7
    rebind_by_glove: bool = False  # True re-binds the track to the punch box when it disagrees; off by default (see --glove-rebind)


# --------------------------------------------------------------------------------------
# Re-ID embedder (one per process, created on first use)
# --------------------------------------------------------------------------------------

_EMBEDDER = None
_EMBEDDER_KEY: tuple | None = None


def _get_embedder(cfg: TrackConfig):
    global _EMBEDDER, _EMBEDDER_KEY
    key = (cfg.embedder, cfg.cpu)
    if _EMBEDDER is None or _EMBEDDER_KEY != key:
        from deep_sort_realtime.embedder import embedder_pytorch

        if cfg.embedder == "torchreid":
            _EMBEDDER = embedder_pytorch.TorchReID_Embedder(bgr=True, gpu=not cfg.cpu)
        else:
            _EMBEDDER = embedder_pytorch.MobileNetv2_Embedder(half=True, bgr=True, gpu=not cfg.cpu)
        _EMBEDDER_KEY = key
        _frame_embeddings.cache_clear()
    return _EMBEDDER


def _init_worker(single_thread: bool) -> None:
    if single_thread:
        import torch

        torch.set_num_threads(1)


def _crop(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = int(np.clip(np.floor(box[0]), 0, max(w - MIN_CROP, 0)))
    y1 = int(np.clip(np.floor(box[1]), 0, max(h - MIN_CROP, 0)))
    x2 = int(np.clip(np.ceil(box[2]), min(x1 + MIN_CROP, w), w))
    y2 = int(np.clip(np.ceil(box[3]), min(y1 + MIN_CROP, h), h))
    return image[y1:y2, x1:x2]


@lru_cache(maxsize=1024)
def _frame_embeddings(path: str, boxes: tuple[tuple[float, float, float, float], ...], cfg: TrackConfig) -> np.ndarray:
    """Re-ID embedding of each person box in one frame. Cached: windows can share frames."""
    embedder = _get_embedder(cfg)
    image = cv2.imread(path)
    if image is None:
        # No frame to look at: a constant vector makes appearance neutral and leaves motion to decide.
        dim = len(embedder.predict([np.zeros((MIN_CROP, MIN_CROP, 3), dtype=np.uint8)])[0])
        return np.full((len(boxes), dim), 1.0 / np.sqrt(dim), dtype=np.float32)
    return np.asarray(embedder.predict([_crop(image, b) for b in boxes]), dtype=np.float32)


# --------------------------------------------------------------------------------------
# Reading the keypoint JSONs
# --------------------------------------------------------------------------------------


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
        people.append({"box": box, "center": _center(box), "kps": kps, "score": float(p.get("score", 1.0))})
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


# --------------------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------------------


def track_window(
    people: list[list[dict]], images: list[Path], cfg: TrackConfig
) -> list[dict[int, str]]:
    """DeepSORT over one window. Returns, per frame, ``{person index: track id}``.

    A person only gets an id in the frames where the tracker matched a detection to a track.
    """
    tracker = DeepSort(
        max_age=cfg.max_age,
        n_init=cfg.n_init,
        max_cosine_distance=cfg.max_cosine_distance,
        max_iou_distance=cfg.max_iou_distance,
        embedder=None,  # embeddings are computed here, so they can be cached across windows
    )
    ids: list[dict[int, str]] = []
    for frame_people, image in zip(people, images):
        usable = [i for i, p in enumerate(frame_people) if p["box"][2] > p["box"][0] and p["box"][3] > p["box"][1]]
        if not usable:
            tracker.update_tracks([], embeds=[])
            ids.append({})
            continue
        boxes = tuple(tuple(float(v) for v in frame_people[i]["box"]) for i in usable)
        embeds = _frame_embeddings(str(image), boxes, cfg)
        detections = [
            ([b[0], b[1], b[2] - b[0], b[3] - b[1]], frame_people[i]["score"], "person") for i, b in zip(usable, boxes)
        ]
        tracks = tracker.update_tracks(detections, embeds=list(embeds), others=usable)
        # Tentative tracks count too: a boxer's id must exist from the first frame they are seen.
        ids.append({t.get_det_supplementary(): t.track_id for t in tracks if t.time_since_update == 0})
    return ids


# --------------------------------------------------------------------------------------
# Choosing the attacker and the opponent
# --------------------------------------------------------------------------------------


def pick_attacker(
    people: list[dict], glove: np.ndarray | None, ref: np.ndarray | None
) -> tuple[int | None, str]:
    """Index of the attacker in one frame from the punch box and/or the last position."""
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
        return min(range(len(people)), key=lambda i: np.linalg.norm(people[i]["center"] - ref)), "nearest"
    return None, "unanchored"


def pick_opponent(people: list[dict], attacker: int | None) -> int | None:
    if attacker is None:
        return None
    others = [i for i in range(len(people)) if i != attacker]
    if not others:
        return None
    ref = people[attacker]["center"]
    return min(others, key=lambda i: np.linalg.norm(people[i]["center"] - ref))


def _step(
    people: list[dict], ids: dict[int, str], glove: np.ndarray | None, state: dict, rebind: bool = True
) -> tuple[int | None, int | None, str]:
    """Attacker and opponent of one frame. ``state`` carries their track ids and last position."""
    if not people:
        return None, None, "no_people"
    by_id = {tid: i for i, tid in ids.items()}
    tracked = by_id.get(state["attacker"])

    idx, how = pick_attacker(people, glove, state["ref"])
    if glove is None:
        if tracked is not None:
            idx, how = tracked, "track"
    elif not rebind and tracked is not None:
        # Ablation: the track keeps the attacker identity; the punch box only seeds it.
        idx, how = tracked, "track"
    elif how == "glove_multi" and tracked is not None and _inside(people[tracked]["box"], glove):
        idx, how = tracked, "track"
    elif how == "glove_nearest" and tracked is not None:
        idx, how = tracked, "track"
    # An unambiguous "glove_in_box" always wins and re-binds the track below.

    if idx is None:
        return None, None, how
    state["attacker"] = ids.get(idx)
    state["ref"] = people[idx]["center"]

    opp = by_id.get(state["opponent"])
    if opp is None or opp == idx:
        opp = pick_opponent(people, idx)
    if opp is not None:
        state["opponent"] = ids.get(opp)
    return idx, opp, how


def pair_record(rec: dict, kp_root: Path, img_root: Path, cfg: TrackConfig) -> tuple[np.ndarray, dict]:
    """Build the (2, T, 17, 3) window and its metadata for one clip."""
    frames = rec["frames"]
    T = len(frames)
    W, H = rec["width"], rec["height"]
    people = [load_people(kp_root, f["image"]) for f in frames]
    gloves = [_glove_px(f, W, H) for f in frames]
    track_ids = track_window(people, [img_root / f["image"] for f in frames], cfg)

    attacker: list[int | None] = [None] * T
    opponent: list[int | None] = [None] * T
    attacker_track: list[str | None] = [None] * T
    how = ["none"] * T

    labeled = [t for t in range(T) if gloves[t] is not None]
    if labeled:
        anchor = labeled[0]
        # Seed the anchor frame, then walk outwards in both directions with the same identities.
        seed = {"attacker": None, "opponent": None, "ref": None}
        attacker[anchor], opponent[anchor], how[anchor] = _step(
            people[anchor], track_ids[anchor], gloves[anchor], seed, cfg.rebind_by_glove
        )
        attacker_track[anchor] = seed["attacker"]
        for order in (range(anchor + 1, T), range(anchor - 1, -1, -1)):
            state = dict(seed)
            for t in order:
                attacker[t], opponent[t], how[t] = _step(people[t], track_ids[t], gloves[t], state, cfg.rebind_by_glove)
                attacker_track[t] = state["attacker"] if attacker[t] is not None else None

    window = np.zeros((2, T, NUM_KEYPOINTS, 3), dtype=np.float32)
    opponent_missing = 0
    for t in range(T):
        a = attacker[t]
        if a is None:
            continue
        window[0, t] = people[t][a]["kps"]
        o = opponent[t]
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
        "attacker_tracks": len({tid for tid in attacker_track if tid is not None}),
    }
    return window, meta


def process_split(
    src: Path, kp_root: Path, split: str, dst: Path, cfg: TrackConfig, workers: int = 1
) -> Counter:
    records = [json.loads(line) for line in (src / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    T = len(records[0]["frames"]) if records else 0
    X = np.zeros((len(records), 2, T, NUM_KEYPOINTS, 3), dtype=np.float32)
    y = np.zeros(len(records), dtype=np.int64)
    metas = []
    quality = Counter()
    n_workers = (os.cpu_count() or 1) if workers == 0 else max(1, workers)
    work = partial(pair_record, kp_root=kp_root, img_root=src, cfg=cfg)
    bar = partial(tqdm, total=len(records), desc=f"pairing {split}", unit="clip", dynamic_ncols=True)
    if n_workers > 1 and len(records) > 1:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(True,)) as pool:
            # map keeps clip order, so X, y and the metadata stay aligned
            results = list(bar(pool.map(work, records, chunksize=16)))
    else:
        results = list(bar(map(work, records)))
    for i, (rec, (window, meta)) in enumerate(zip(records, results)):
        X[i] = window
        y[i] = rec["label"]
        metas.append(meta)
        quality[meta["quality"]] += 1

    np.savez_compressed(dst / f"{split}.npz", X=X, y=y)
    with (dst / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
        for m in metas:
            fh.write(json.dumps(m) + "\n")
    stable = sum(m["attacker_tracks"] == 1 for m in metas)
    log.info("%s: %d windows %s, attacker identity stable in %d", split, len(records), dict(quality), stable)
    return quality


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m boxing.pairing",
        description="Pair each punch sequence with its opponent, tracking the boxers with DeepSORT and Re-ID.",
    )
    parser.add_argument("--src", type=Path, default=Path("data/processed/punch-seq"), help="punch-seq directory with train/val jsonl, labels.json and images/")
    parser.add_argument("--keypoints", type=Path, default=Path("data/processed/punch-keypoints-boxers/images"), help="root of the per-frame keypoint JSONs")
    parser.add_argument("--dst", type=Path, default=Path("data/processed/punch-pairs"), help="output directory (created if missing)")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--workers", type=int, default=1, help="processes pairing clips in parallel; 0 = all cores (default 1)")
    parser.add_argument("--embedder", choices=EMBEDDERS, default="torchreid", help="Re-ID appearance model (default torchreid, OSNet)")
    parser.add_argument("--cpu", action="store_true", help="run the embedder on the CPU even if an accelerator is available")
    parser.add_argument("--max-age", type=int, default=TrackConfig.max_age, help="frames a track may go unseen before it is dropped")
    parser.add_argument("--n-init", type=int, default=TrackConfig.n_init, help="hits before a track is confirmed")
    parser.add_argument("--max-cosine-distance", type=float, default=TrackConfig.max_cosine_distance, help="appearance gate: larger matches looser")
    parser.add_argument("--max-iou-distance", type=float, default=TrackConfig.max_iou_distance, help="motion gate on box overlap")
    parser.add_argument("--glove-rebind", action="store_true", help="re-bind the attacker's track to the punch box when they disagree (off by default: it caused most swaps)")
    args = parser.parse_args()

    if not args.src.is_dir():
        parser.error(f"--src is not a directory: {args.src}")
    if not args.keypoints.is_dir():
        parser.error(f"--keypoints is not a directory: {args.keypoints}")

    cfg = TrackConfig(
        embedder=args.embedder,
        cpu=args.cpu,
        max_age=args.max_age,
        n_init=args.n_init,
        max_cosine_distance=args.max_cosine_distance,
        max_iou_distance=args.max_iou_distance,
        rebind_by_glove=args.glove_rebind,
    )
    args.dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.src / "labels.json", args.dst / "labels.json")
    for split in args.splits:
        process_split(args.src, args.keypoints, split, args.dst, cfg, args.workers)


if __name__ == "__main__":
    main()

"""Pair each punch sequence with the two boxers in it, identified by DeepSORT track ids.

Reads the punch-seq clips (``train.jsonl`` / ``val.jsonl`` written by
``boxing.preprocess``), the frames they point to, and the per-frame pose JSONs (written by
``boxing.keypoints``). For every 16-frame window it finds the two boxers and writes one array
per split:

    X[N, 2, T, 17, 3]   row r = the r-th boxer of the window, ordered by DeepSORT track id
                        last axis = (x, y, score) in source pixels
                        missing people are zero-filled with score 0

Identity comes from a tracker, not from "the nearest person to where they were". Each person box
is cropped from the frame and embedded by a Re-ID network; DeepSORT (Kalman motion model +
appearance matching) gives every person a track id that survives clinches, where two boxers
overlap and nearest-centre matching swaps them.

    1. Track every person through the window (forward, one DeepSORT per clip).
    2. The two tracks seen in the most frames are the boxers (ties: the larger mean box).
    3. Row 0 is the boxer with the lower track id, row 1 the other. DeepSORT numbers tracks in the
       order it creates them, so this is the boxer seen first. The ids restart for every window:
       row 0 of one window is not the same person as row 0 of another.
    4. If a track is not matched on a frame, the nearest person to its last known position that is
       not the other boxer fills in.

No role is assigned, and the labeled punch box is not used.

Outputs in ``--dst``:
    train.npz, val.npz    X (float32) and y (int64, class id)
    train.jsonl, val.jsonl  one metadata row per window, aligned with X
    labels.json           copied from ``--src``

The metadata has ``track_ids`` (the two DeepSORT ids, row order), ``track_frames`` (frames each
row was filled), ``method`` (counts of ``track`` frames, ``nearest`` fallback frames and empty
ones) and ``quality``: ``good`` when both boxers are present on every frame, ``partial`` when
some are missing, ``no_boxers`` when none was found.

Usage:
    python -m boxing.pairing --src data/processed/punch-seq \\
        --keypoints data/processed/punch-keypoints-boxers/images \\
        --dst data/processed/punch-pairs [--workers N]

``--workers N`` pairs clips in N processes (``0`` = every core); each process loads its own
embedder, so on a GPU keep N small. The Re-ID embedder is the MobileNetV2 that ships with
``deep-sort-realtime``, so no extra dependency is needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import Counter, defaultdict
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
N_BOXERS = 2
SPLITS = ("train", "val")
MIN_CROP = 2  # pixels; smaller crops cannot be embedded


@dataclass(frozen=True)
class TrackConfig:
    """DeepSORT and Re-ID settings (picklable, so it can go to worker processes)."""

    cpu: bool = False
    max_age: int = 16  # frames a track may go unseen; a whole window, so a boxer can be hidden and come back
    n_init: int = 2  # hits before a track is confirmed
    max_cosine_distance: float = 0.3
    max_iou_distance: float = 0.7


# --------------------------------------------------------------------------------------
# Re-ID embedder (one per process, created on first use)
# --------------------------------------------------------------------------------------

_EMBEDDER = None
_EMBEDDER_KEY: bool | None = None


def _get_embedder(cfg: TrackConfig):
    global _EMBEDDER, _EMBEDDER_KEY
    key = cfg.cpu
    if _EMBEDDER is None or _EMBEDDER_KEY != key:
        from deep_sort_realtime.embedder import embedder_pytorch

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
# Choosing the two boxers
# --------------------------------------------------------------------------------------


def _id_key(track_id: str) -> tuple[int, str]:
    """Sort key putting numeric ids in numeric order ("2" before "10")."""
    return (int(track_id), track_id) if str(track_id).isdigit() else (1 << 62, str(track_id))


def pick_tracks(people: list[list[dict]], track_ids: list[dict[int, str]], n: int = N_BOXERS) -> list[str]:
    """The ``n`` tracks seen in the most frames, ordered by track id.

    Ties go to the track with the larger mean box, which is the boxer rather than a spectator.
    """
    frames: Counter = Counter()
    area: defaultdict = defaultdict(float)
    for frame_people, ids in zip(people, track_ids):
        for i, tid in ids.items():
            box = frame_people[i]["box"]
            frames[tid] += 1
            area[tid] += float((box[2] - box[0]) * (box[3] - box[1]))
    best = sorted(frames, key=lambda tid: (-frames[tid], -area[tid] / frames[tid], _id_key(tid)))[:n]
    return sorted(best, key=_id_key)


def follow_tracks(
    people: list[list[dict]], track_ids: list[dict[int, str]], tracks: list[str]
) -> tuple[list[list[int | None]], list[list[str]]]:
    """Person index of each track on each frame, and how it was found.

    Returns ``(index[r][t], how[r][t])``. ``how`` is ``track`` (matched by DeepSORT), ``nearest``
    (the track was not matched: the closest person to its last known position that is not the
    other boxer), ``no_people`` or ``missing``.
    """
    T = len(people)
    index: list[list[int | None]] = [[None] * T for _ in tracks]
    how: list[list[str]] = [["missing"] * T for _ in tracks]
    # Before a track first shows up, its first position is the best guess for where it is.
    last: dict[str, np.ndarray] = {}
    for frame_people, ids in zip(people, track_ids):
        for i, tid in ids.items():
            if tid in tracks:
                last.setdefault(tid, frame_people[i]["center"])
    for t in range(T):
        if not people[t]:
            for r in range(len(tracks)):
                how[r][t] = "no_people"
            continue
        by_id = {tid: i for i, tid in track_ids[t].items()}
        taken = {by_id[tid] for tid in tracks if tid in by_id}
        for r, tid in enumerate(tracks):
            if tid in by_id:
                index[r][t], how[r][t] = by_id[tid], "track"
            elif tid in last:
                free = [i for i in range(len(people[t])) if i not in taken]
                if free:
                    idx = min(free, key=lambda i: np.linalg.norm(people[t][i]["center"] - last[tid]))
                    taken.add(idx)
                    index[r][t], how[r][t] = idx, "nearest"
            if index[r][t] is not None:
                last[tid] = people[t][index[r][t]]["center"]
    return index, how


def pair_record(rec: dict, kp_root: Path, img_root: Path, cfg: TrackConfig) -> tuple[np.ndarray, dict]:
    """Build the (2, T, 17, 3) window and its metadata for one clip."""
    frames = rec["frames"]
    T = len(frames)
    W, H = rec["width"], rec["height"]
    people = [load_people(kp_root, f["image"]) for f in frames]
    track_ids = track_window(people, [img_root / f["image"] for f in frames], cfg)

    tracks = pick_tracks(people, track_ids)
    index, how = follow_tracks(people, track_ids, tracks)

    window = np.zeros((N_BOXERS, T, NUM_KEYPOINTS, 3), dtype=np.float32)
    for r, row in enumerate(index):
        for t, i in enumerate(row):
            if i is not None:
                window[r, t] = people[t][i]["kps"]

    filled = [sum(i is not None for i in row) for row in index]
    filled += [0] * (N_BOXERS - len(filled))  # fewer than two boxers were tracked
    if sum(filled) == 0:
        quality = "no_boxers"
    elif all(n == T for n in filled):
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
        "track_ids": tracks,
        "track_frames": filled,
        "quality": quality,
        "method": dict(Counter(h for row in how for h in row)),
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
    if records and quality["no_boxers"] == len(records):
        found = sum(1 for _ in kp_root.rglob("*.json"))
        log.warning(
            "%s: no boxer found in any of the %d windows. %d keypoint JSON files under %s: "
            "run boxing.keypoints first and check it wrote people for these frames.",
            split, len(records), found, kp_root,
        )
    tracked = sum(m["method"].get("nearest", 0) == 0 for m in metas)
    log.info("%s: %d windows %s, %d without any nearest-person fallback", split, len(records), dict(quality), tracked)
    return quality


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m boxing.pairing",
        description="Pair each punch sequence with its two boxers, identified by DeepSORT track ids and Re-ID.",
    )
    parser.add_argument("--src", type=Path, default=Path("data/processed/punch-seq"), help="punch-seq directory with train/val jsonl, labels.json and images/")
    parser.add_argument("--keypoints", type=Path, default=Path("data/processed/punch-keypoints-boxers/images"), help="root of the per-frame keypoint JSONs")
    parser.add_argument("--dst", type=Path, default=Path("data/processed/punch-pairs"), help="output directory (created if missing)")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--workers", type=int, default=1, help="processes pairing clips in parallel; 0 = all cores (default 1)")
    parser.add_argument("--cpu", action="store_true", help="run the embedder on the CPU even if an accelerator is available")
    parser.add_argument("--max-age", type=int, default=TrackConfig.max_age, help="frames a track may go unseen before it is dropped")
    parser.add_argument("--n-init", type=int, default=TrackConfig.n_init, help="hits before a track is confirmed")
    parser.add_argument("--max-cosine-distance", type=float, default=TrackConfig.max_cosine_distance, help="appearance gate: larger matches looser")
    parser.add_argument("--max-iou-distance", type=float, default=TrackConfig.max_iou_distance, help="motion gate on box overlap")
    args = parser.parse_args()

    if not args.src.is_dir():
        parser.error(f"--src is not a directory: {args.src}")
    if not args.keypoints.is_dir():
        parser.error(f"--keypoints is not a directory: {args.keypoints}")

    cfg = TrackConfig(
        cpu=args.cpu,
        max_age=args.max_age,
        n_init=args.n_init,
        max_cosine_distance=args.max_cosine_distance,
        max_iou_distance=args.max_iou_distance,
    )
    args.dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.src / "labels.json", args.dst / "labels.json")
    for split in args.splits:
        process_split(args.src, args.keypoints, split, args.dst, cfg, args.workers)


if __name__ == "__main__":
    main()

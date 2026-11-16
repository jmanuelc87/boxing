"""Convert the raw CVAT punch dataset into a normalised detection dataset.

Input (``src``): one folder per task, each with ``annotations.json`` (CVAT video
1.1 JSON) and ``data/<video>.mp4``. See README.md for the annotation format.

Output (``dst``)::

    dst/
    ├── labels.json              # {"names": [...], "id2label": {...}, "label2id": {...}}
    ├── images/<task>/<frame>.jpg
    ├── train.jsonl              # one record per annotated frame
    └── val.jsonl

Each JSONL record::

    {"image": "images/<task>/<frame>.jpg", "width": W, "height": H,
     "class_labels": [0, 3], "boxes": [[cx, cy, w, h], ...]}

Boxes are normalised to [0, 1] in ``(cx, cy, w, h)`` order, which is what
``RTDetrV2ForObjectDetection`` expects. Images are resized so the longest side
is ``--max-size`` pixels; normalised boxes are unaffected by that resize.

Track keyframes are linearly interpolated, so every frame between a track's
first keyframe and its ``outside`` keyframe gets a box. Only frames that contain
at least one box are written.

Usage::

    uv run python -m boxing.preprocess data/raw/punch data/processed/punch
    uv run python -m boxing.preprocess data/raw/punch data/processed/punch-test --test   # quick check
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

# --test mode: a small end-to-end run (both splits, real decode/resize/write).
TEST_TASKS = 3
TEST_FRAMES_PER_TASK = 20

# Class names as they appear in the raw CVAT annotations (Polish). Order is the
# class id, so the ids are fixed by this list.
RAW_LABELS: list[str] = [
    "Głowa lewą ręką",  # head, left hand
    "Głowa prawą ręką",  # head, right hand
    "Chybienie lewą ręką",  # miss, left hand
    "Chybienie prawą ręką",  # miss, right hand
    "Blok lewą ręką",  # block, left hand
    "Blok prawą ręką",  # block, right hand
    "Korpus lewą ręką",  # body, left hand
    "Korpus prawą ręką",  # body, right hand
]
RAW_LABEL2ID = {name: i for i, name in enumerate(RAW_LABELS)}

# Output class names (Spanish boxing terms), indexed by the same class id. These
# go into labels.json and are what the model sees as its label names.
#
# The raw annotations record the hand and the target, not the punch type. Left
# hand is mapped to "jab" and right hand to "cruzado" (the orthodox-stance
# lead/rear punches). Hooks and uppercuts are not separable in this dataset.
LABELS: list[str] = [
    "Jab a la cabeza",  # head, left hand
    "Cruzado a la cabeza",  # head, right hand
    "Jab fallido",  # miss, left hand
    "Cruzado fallido",  # miss, right hand
    "Bloqueo de jab",  # block, left hand
    "Bloqueo de cruzado",  # block, right hand
    "Jab al cuerpo",  # body, left hand
    "Cruzado al cuerpo",  # body, right hand
]
assert len(LABELS) == len(RAW_LABELS)


def interpolate_track(shapes: list[dict]) -> dict[int, list[float]]:
    """Return {frame: [x1, y1, x2, y2]} for every frame covered by one track.

    Keyframes are sorted by frame. A keyframe with ``outside=True`` ends the
    track and is not itself a visible box.
    """
    keys = sorted(shapes, key=lambda s: s["frame"])
    out: dict[int, list[float]] = {}
    for cur, nxt in zip(keys, keys[1:]):
        f0, f1 = cur["frame"], nxt["frame"]
        if cur.get("outside") or f1 <= f0:
            continue
        p0, p1 = cur["points"], nxt["points"]
        for f in range(f0, f1):
            t = (f - f0) / (f1 - f0)
            out[f] = [a + (b - a) * t for a, b in zip(p0, p1)]
    last = keys[-1]
    if not last.get("outside"):
        # The final keyframe is visible and nothing ends the track after it.
        out[last["frame"]] = list(last["points"])
    return out


def to_normalised_cxcywh(points: list[float], width: int, height: int) -> list[float] | None:
    """Convert pixel (x1, y1, x2, y2) to normalised (cx, cy, w, h); None if degenerate."""
    x1, y1, x2, y2 = points
    x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
    w, h = (x2 - x1) / width, (y2 - y1) / height
    if w <= 0 or h <= 0:
        return None
    return [(x1 / width + w / 2), (y1 / height + h / 2), w, h]


def collect_task(task_dir: Path) -> dict[int, list[tuple[int, list[float]]]]:
    """Return {frame: [(class_id, pixel_box), ...]} for one task."""
    data = json.loads((task_dir / "annotations.json").read_text(encoding="utf-8"))
    frames: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for track in data[0]["tracks"]:
        cls = RAW_LABEL2ID.get(track["label"])
        if cls is None:
            raise ValueError(f"{task_dir.name}: unknown label {track['label']!r}")
        for frame, box in interpolate_track(track["shapes"]).items():
            frames[frame].append((cls, box))
    return frames


def extract_frames(
    video: Path, wanted: set[int], out_dir: Path, max_size: int, desc: str | None = None
) -> dict[int, tuple[int, int]]:
    """Decode ``video`` and write the ``wanted`` frames as JPEGs.

    Frames that are not wanted are skipped with ``grab()`` (no pixel decode).
    Decoding stops after the last wanted frame. ``desc`` titles a per-video
    progress bar; ``None`` hides it.

    Returns {frame: (width, height)} of the written images, after resizing.
    """
    if not wanted:
        return {}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, tuple[int, int]] = {}
    last_needed = max(wanted)
    try:
        with tqdm(total=last_needed + 1, desc=desc, unit="frame", leave=False, disable=desc is None) as bar:
            for idx in range(last_needed + 1):
                if not cap.grab():
                    break
                bar.update(1)
                if idx not in wanted:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                h, w = frame.shape[:2]
                scale = min(1.0, max_size / max(h, w))
                if scale < 1.0:
                    frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
                path = out_dir / f"{idx:06d}.jpg"
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"failed to write {path}")
                written[idx] = (frame.shape[1], frame.shape[0])
    finally:
        cap.release()
    return written


def process_task(
    task_dir: Path,
    dst: Path,
    max_size: int,
    src_size: tuple[int, int],
    max_frames: int | None = None,
    show_progress: bool = True,
) -> list[dict]:
    """Extract frames and build JSONL records for one task.

    ``max_frames`` keeps only the first N annotated frames (used by ``--test``).
    """
    frames = collect_task(task_dir)
    if max_frames is not None:
        frames = {f: frames[f] for f in sorted(frames)[:max_frames]}
    video = next((task_dir / "data").glob("*.mp4"))
    rel_dir = Path("images") / task_dir.name
    sizes = extract_frames(video, set(frames), dst / rel_dir, max_size, task_dir.name if show_progress else None)

    records = []
    for frame in sorted(frames):
        if frame not in sizes:
            continue  # video ended early; nothing to label
        tw, th = sizes[frame]
        src_w, src_h = src_size
        boxes, classes = [], []
        for cls, pts in frames[frame]:
            nb = to_normalised_cxcywh(pts, src_w, src_h)
            if nb is not None:
                boxes.append(nb)
                classes.append(cls)
        if not boxes:
            continue
        records.append(
            {
                "image": (rel_dir / f"{frame:06d}.jpg").as_posix(),
                "width": tw,
                "height": th,
                "class_labels": classes,
                "boxes": boxes,
            }
        )
    return records


def source_size(video: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def split_tasks(tasks: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Split by task (recording session), never by frame, to avoid leakage."""
    if not 0 <= val_fraction < 1:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    shuffled = sorted(tasks)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction)) if val_fraction > 0 else 0
    return shuffled[n_val:], shuffled[:n_val]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def preprocess(
    src: Path,
    dst: Path,
    max_size: int = 640,
    val_fraction: float = 0.2,
    seed: int = 0,
    test: bool = False,
) -> None:
    """Run the full pipeline. With ``test=True``, process only a small sample.

    Test mode runs every stage (split, decode, resize, write, JSONL, labels) on
    ``TEST_TASKS`` tasks spread across the dataset and the first
    ``TEST_FRAMES_PER_TASK`` annotated frames of each, so it finishes quickly.
    """
    tasks = sorted(p for p in src.iterdir() if (p / "annotations.json").is_file())
    if not tasks:
        raise FileNotFoundError(f"no task folders with annotations.json under {src}")
    max_frames = None
    if test:
        tasks = tasks[:: max(1, len(tasks) // TEST_TASKS)][:TEST_TASKS]
        max_frames = TEST_FRAMES_PER_TASK
        val_fraction = max(val_fraction, 1 / len(tasks)) if len(tasks) > 1 else 0.0
        print(f"[test] {len(tasks)} tasks, up to {max_frames} frames each")
    dst.mkdir(parents=True, exist_ok=True)

    sizes = {t.name: source_size(next((t / "data").glob("*.mp4"))) for t in tasks}

    train_tasks, val_tasks = split_tasks(tasks, val_fraction, seed)
    for split, group in (("train", train_tasks), ("val", val_tasks)):
        records: list[dict] = []
        for task in tqdm(group, desc=split, unit="task"):
            records.extend(process_task(task, dst, max_size, sizes[task.name], max_frames))
        write_jsonl(dst / f"{split}.jsonl", records)
        print(f"[{split}] {len(records)} frames from {len(group)} tasks")

    (dst / "labels.json").write_text(
        json.dumps(
            {
                "names": LABELS,
                "id2label": {str(i): n for i, n in enumerate(LABELS)},
                "label2id": {name: i for i, name in enumerate(LABELS)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="raw dataset folder (contains task_* folders)")
    parser.add_argument("dst", type=Path, help="output folder for images and JSONL splits")
    parser.add_argument("--max-size", type=int, default=640, help="resize longest image side to this (default 640)")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="fraction of tasks held out (default 0.2)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the task split (default 0)")
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"quick end-to-end run: {TEST_TASKS} tasks, {TEST_FRAMES_PER_TASK} frames each",
    )
    args = parser.parse_args()
    preprocess(args.src, args.dst, args.max_size, args.val_fraction, args.seed, args.test)


if __name__ == "__main__":
    main()

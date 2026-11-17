"""Convert the raw CVAT punch dataset into a dataset of fixed-length frame clips.

Input (``src``): one folder per task, each with ``annotations.json`` (CVAT video
1.1 JSON) and ``data/<video>.mp4``. See README.md for the annotation format.

Output (``dst``)::

    dst/
    ├── labels.json              # {"names", "id2label", "label2id", "seq_len"}
    ├── images/<task>/<frame>.jpg
    ├── train.jsonl              # one record per clip
    └── val.jsonl

Each punch track becomes one clip of ``SEQ_LEN`` consecutive frames centred on
the track, so the clip also holds context frames before and after the punch.
Each JSONL record::

    {"task": "<task>", "track": 12, "label": 2, "start": 647,
     "width": W, "height": H,
     "frames": [{"image": "images/<task>/000647.jpg",
                 "class_labels": [], "boxes": []}, ...]}

``frames`` has exactly ``SEQ_LEN`` entries in frame order. Every frame lists the
boxes of all tracks visible on it (not only the clip's own track), so a frame
where two punches overlap carries both; a frame with no punch has empty lists.
``track`` (index in the task's annotations), ``label`` (that track's class id)
and ``start`` (first frame number) are metadata.

Boxes are normalised to [0, 1] in ``(cx, cy, w, h)`` order. Images are resized so the longest side
is ``--max-size`` pixels; normalised boxes are unaffected by that resize.

Track keyframes are linearly interpolated, so every frame between a track's
first keyframe and its ``outside`` keyframe gets a box.

Usage::

    uv run python -m boxing.preprocess data/raw/punch data/processed/punch-seq
    uv run python -m boxing.preprocess data/raw/punch data/processed/punch-seq-test --test   # quick check
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

# Frames per clip: 0.32 s at 50 fps. Punch tracks last 4 frames at the median and
# at most 19, so 16 frames hold 99% of punches with context on both sides.
SEQ_LEN = 16

# --test mode: a small end-to-end run (both splits, real decode/resize/write).
TEST_TASKS = 3
TEST_CLIPS_PER_TASK = 5

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

# One track: (class id, {frame: pixel box [x1, y1, x2, y2]}).
Track = tuple[int, dict[int, list[float]]]


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


def collect_tracks(task_dir: Path) -> list[Track]:
    """Return one (class_id, {frame: pixel_box}) per track, in annotation order.

    Tracks with no visible frame (only an ``outside`` keyframe) are kept as an
    empty dict so that the list index stays the track's index in the file.
    """
    data = json.loads((task_dir / "annotations.json").read_text(encoding="utf-8"))
    tracks: list[Track] = []
    for track in data[0]["tracks"]:
        cls = RAW_LABEL2ID.get(track["label"])
        if cls is None:
            raise ValueError(f"{task_dir.name}: unknown label {track['label']!r}")
        tracks.append((cls, interpolate_track(track["shapes"])))
    return tracks


def merge_tracks(tracks: list[Track]) -> dict[int, list[tuple[int, list[float]]]]:
    """Return {frame: [(class_id, pixel_box), ...]}: every box visible on each frame."""
    frames: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for cls, boxes in tracks:
        for frame, box in boxes.items():
            frames[frame].append((cls, box))
    return frames


def collect_task(task_dir: Path) -> dict[int, list[tuple[int, list[float]]]]:
    """Return {frame: [(class_id, pixel_box), ...]} for one task."""
    return merge_tracks(collect_tracks(task_dir))


def clip_window(first: int, last: int, seq_len: int, n_frames: int) -> int:
    """Return the start frame of a ``seq_len`` clip centred on frames [first, last].

    The clip is shifted to stay inside the video ([0, n_frames)). A track longer
    than ``seq_len`` keeps only its centre ``seq_len`` frames.
    """
    if n_frames < seq_len:
        raise ValueError(f"video has {n_frames} frames, fewer than seq_len={seq_len}")
    start = (first + last) // 2 - seq_len // 2
    return min(max(start, 0), n_frames - seq_len)


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
    video_meta: tuple[int, int, int],
    seq_len: int = SEQ_LEN,
    max_clips: int | None = None,
    show_progress: bool = True,
) -> tuple[list[dict], int]:
    """Extract clip frames and build one JSONL record per track for one task.

    ``video_meta`` is ``(width, height, n_frames)`` of the source video.
    ``max_clips`` keeps only the first N tracks (used by ``--test``).

    Returns (records, number of clips dropped because the video is shorter
    than the annotations).
    """
    src_w, src_h, n_frames = video_meta
    tracks = collect_tracks(task_dir)
    per_frame = merge_tracks(tracks)

    clips: list[tuple[int, int, int]] = []  # (track index, class id, start frame)
    dropped = 0
    for i, (cls, boxes) in enumerate(tracks):
        if not boxes:
            continue  # track has no visible frame
        if min(boxes) >= n_frames:
            dropped += 1  # annotated past the end of the video
            continue
        start = clip_window(min(boxes), max(boxes), seq_len, n_frames)
        clips.append((i, cls, start))
    if max_clips is not None:
        clips = clips[:max_clips]

    wanted = {start + k for _, _, start in clips for k in range(seq_len)}
    video = next((task_dir / "data").glob("*.mp4"))
    rel_dir = Path("images") / task_dir.name
    sizes = extract_frames(video, wanted, dst / rel_dir, max_size, task_dir.name if show_progress else None)

    records = []
    for track, cls, start in clips:
        frame_ids = range(start, start + seq_len)
        if any(f not in sizes for f in frame_ids):
            dropped += 1  # the video ended before the clip did
            continue
        frames = []
        for f in frame_ids:
            boxes, classes = [], []
            for c, pts in per_frame.get(f, []):
                nb = to_normalised_cxcywh(pts, src_w, src_h)
                if nb is not None:
                    boxes.append(nb)
                    classes.append(c)
            frames.append({"image": (rel_dir / f"{f:06d}.jpg").as_posix(), "class_labels": classes, "boxes": boxes})
        tw, th = sizes[start]
        records.append(
            {
                "task": task_dir.name,
                "track": track,
                "label": cls,
                "start": start,
                "width": tw,
                "height": th,
                "frames": frames,
            }
        )
    return records, dropped


def video_info(video: Path) -> tuple[int, int, int]:
    """Return (width, height, frame count) of ``video``."""
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {video}")
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


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
    seq_len: int = SEQ_LEN,
    show_progress: bool = True,
) -> None:
    """Run the full pipeline. With ``test=True``, process only a small sample.

    Test mode runs every stage (split, decode, resize, write, JSONL, labels) on
    ``TEST_TASKS`` tasks spread across the dataset and the first
    ``TEST_CLIPS_PER_TASK`` clips of each, so it finishes quickly.
    """
    tasks = sorted(p for p in src.iterdir() if (p / "annotations.json").is_file())
    if not tasks:
        raise FileNotFoundError(f"no task folders with annotations.json under {src}")
    max_clips = None
    if test:
        tasks = tasks[:: max(1, len(tasks) // TEST_TASKS)][:TEST_TASKS]
        max_clips = TEST_CLIPS_PER_TASK
        val_fraction = max(val_fraction, 1 / len(tasks)) if len(tasks) > 1 else 0.0
        print(f"[test] {len(tasks)} tasks, up to {max_clips} clips each")
    dst.mkdir(parents=True, exist_ok=True)

    meta = {t.name: video_info(next((t / "data").glob("*.mp4"))) for t in tasks}

    train_tasks, val_tasks = split_tasks(tasks, val_fraction, seed)
    for split, group in (("train", train_tasks), ("val", val_tasks)):
        records: list[dict] = []
        dropped = 0
        for task in tqdm(group, desc=split, unit="task", disable=not show_progress):
            recs, d = process_task(task, dst, max_size, meta[task.name], seq_len, max_clips, show_progress)
            records.extend(recs)
            dropped += d
        write_jsonl(dst / f"{split}.jsonl", records)
        note = f", {dropped} dropped (video shorter than annotations)" if dropped else ""
        print(f"[{split}] {len(records)} clips of {seq_len} frames from {len(group)} tasks{note}")

    (dst / "labels.json").write_text(
        json.dumps(
            {
                "names": LABELS,
                "id2label": {str(i): n for i, n in enumerate(LABELS)},
                "label2id": {name: i for i, name in enumerate(LABELS)},
                "seq_len": seq_len,
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
        "--seq-len",
        type=int,
        default=SEQ_LEN,
        help=f"frames per clip (default {SEQ_LEN})",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"quick end-to-end run: {TEST_TASKS} tasks, {TEST_CLIPS_PER_TASK} clips each",
    )
    args = parser.parse_args()
    preprocess(
        args.src,
        args.dst,
        args.max_size,
        args.val_fraction,
        args.seed,
        args.test,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()

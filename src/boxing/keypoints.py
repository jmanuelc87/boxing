"""Extract COCO person keypoints with the ``pictograph/pg-coco-pose`` model.

The model is an RF-DETR keypoint detector published as ONNX, so it runs through
ONNX Runtime rather than ``transformers``.

    python -m boxing.keypoints SRC DST [--threshold 0.5] [--all-people]

By default people are classified by torso colour (see ``boxing.color_roles``) and only the
red and blue boxers are kept; referees, audience and anyone else are dropped from every
frame. Pass ``--all-people`` to keep everyone. The colour rules are heuristics, so check the
output on your footage.

SRC is an image, a directory of images, a dataset root, or a video:

* image     -> DST is a ``.json`` file
* directory -> DST is a directory; one ``.json`` per image, same relative paths
* dataset   -> a root with ``labels.json`` and split files (e.g. ``data/processed/punch-seq``).
  Every frame referenced by the splits is processed; DST is a directory that mirrors
  the dataset's image paths, one ``.json`` per frame
* video     -> DST is a ``.jsonl`` file; one line per frame

Each record lists the detected people. A person has a pixel box ``[x1, y1, x2, y2]``,
a detection ``score``, and 17 keypoints ``{"name", "x", "y", "score"}`` in COCO order.

Weights are downloaded from the Hugging Face Hub on first use and cached. The model
is licensed for non-commercial use only; see the model card.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from boxing.color_roles import classify_person

BOXER_ROLES = {"boxer_red", "boxer_blue"}

REPO_ID = "pictograph/pg-coco-pose"
ONNX_FILE = "pg-coco-pose.onnx"
CONFIG_FILE = "config.json"

# ImageNet statistics, the normalisation the model was trained with.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Raw ``pred_keypoints`` layout, per query: rows 0..16 are the joints in config
# order; columns 0 and 1 are x and y normalised to the full frame; column 7 is the
# joint score.
KP_SCORE_COL = 7

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True)
class Joint:
    name: str
    x: float  # pixels in the original image
    y: float
    score: float


@dataclass(frozen=True)
class Person:
    box: list[float]  # [x1, y1, x2, y2] in pixels
    score: float
    keypoints: list[Joint] = field(default_factory=list)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def decode(
    outputs: dict[str, np.ndarray],
    names: list[str],
    width: int,
    height: int,
    threshold: float = 0.5,
) -> list[Person]:
    """Turn the raw ``pred_*`` tensors of one image into people in pixel coordinates.

    ``outputs`` holds a batch of one: ``pred_boxes`` (1, Q, 4) normalised cxcywh,
    ``pred_logits`` (1, Q, C+1) with the background as the last slot, and
    ``pred_keypoints`` (1, Q, R, 8) with R >= len(names).
    """
    boxes = outputs["pred_boxes"][0]
    logits = outputs["pred_logits"][0]
    keypoints = outputs["pred_keypoints"][0]

    # RF-DETR uses per-class sigmoid, not softmax. The last slot is background.
    scores = sigmoid(logits[:, :-1]).max(axis=1)

    people: list[Person] = []
    for q in np.flatnonzero(scores > threshold):
        cx, cy, bw, bh = boxes[q]
        box = [
            float((cx - bw / 2) * width),
            float((cy - bh / 2) * height),
            float((cx + bw / 2) * width),
            float((cy + bh / 2) * height),
        ]
        joints = [
            Joint(
                name=name,
                x=float(keypoints[q, j, 0] * width),
                y=float(keypoints[q, j, 1] * height),
                score=float(keypoints[q, j, KP_SCORE_COL]),
            )
            for j, name in enumerate(names)
        ]
        people.append(Person(box=box, score=float(scores[q]), keypoints=joints))
    return people


def keep_boxers(image_bgr: np.ndarray, people: list[Person]) -> list[Person]:
    """Drop everyone whose torso colour is not a red or blue boxer (referee, audience, ...)."""
    return [p for p in people if classify_person(image_bgr, asdict(p)) in BOXER_ROLES]


class PoseEstimator:
    def __init__(
        self,
        onnx_path: str | Path,
        config_path: str | Path,
        threshold: float = 0.5,
        providers: list[str] | None = None,
        boxers_only: bool = False,
    ):
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))["_pictograph"]
        self.input_h, self.input_w = config["input_shape"]
        self.names: list[str] = config["class_names"]
        self.threshold = threshold
        self.boxers_only = boxers_only

        available = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    @classmethod
    def from_hub(cls, threshold: float = 0.5, boxers_only: bool = False) -> PoseEstimator:
        return cls(
            hf_hub_download(REPO_ID, ONNX_FILE),
            hf_hub_download(REPO_ID, CONFIG_FILE),
            threshold=threshold,
            boxers_only=boxers_only,
        )

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """BGR image -> NCHW float32 batch of one. The frame is stretched to the input size."""
        resized = cv2.resize(image_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (rgb - MEAN) / STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def predict(self, image_bgr: np.ndarray) -> list[Person]:
        height, width = image_bgr.shape[:2]
        raw = self.session.run(None, {self.input_name: self.preprocess(image_bgr)})
        outputs = {o.name: r for o, r in zip(self.session.get_outputs(), raw)}
        people = decode(outputs, self.names, width, height, self.threshold)
        return keep_boxers(image_bgr, people) if self.boxers_only else people


def record(people: list[Person], width: int, height: int) -> dict:
    return {"width": width, "height": height, "people": [asdict(p) for p in people]}


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"cannot read image: {path}")
    return image


def run_image(estimator: PoseEstimator, src: Path, dst: Path) -> None:
    image = _read_image(src)
    if dst.is_dir():
        dst = dst / f"{src.stem}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    h, w = image.shape[:2]
    dst.write_text(json.dumps(record(estimator.predict(image), w, h), indent=2), encoding="utf-8")


def is_dataset(root: Path) -> bool:
    """A dataset root has ``labels.json`` and split files such as ``train.jsonl``."""
    return (root / "labels.json").is_file() and any(root.glob("*.jsonl"))


def dataset_images(root: Path) -> list[Path]:
    """Every frame image referenced by the split files, relative to ``root``, in first-seen order."""
    seen: dict[str, None] = {}
    for split in sorted(root.glob("*.jsonl")):
        for line in split.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            for frame in json.loads(line)["frames"]:
                seen.setdefault(frame["image"], None)
    return [Path(p) for p in seen]


def run_directory(estimator: PoseEstimator, src: Path, dst: Path) -> None:
    if is_dataset(src):
        images = [src / rel for rel in dataset_images(src)]
        missing = [p for p in images if not p.is_file()]
        if missing:
            raise SystemExit(f"{len(missing)} image(s) listed in the splits are missing, e.g. {missing[0]}")
    else:
        images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no images found under {src}")
    for path in tqdm(images, unit="img"):
        out = (dst / path.relative_to(src)).with_suffix(".json")
        run_image(estimator, path, out)


def run_video(estimator: PoseEstimator, src: Path, dst: Path) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dst.open("w", encoding="utf-8") as f, tqdm(total=total, unit="frame") as bar:
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                rec = {"frame": index, "time_s": index / fps if fps > 0 else None}
                rec.update(record(estimator.predict(frame), w, h))
                f.write(json.dumps(rec) + "\n")
                index += 1
                bar.update(1)
    finally:
        cap.release()


def run(src: Path, dst: Path, estimator: PoseEstimator) -> None:
    if src.is_dir():
        run_directory(estimator, src, dst)
    elif src.suffix.lower() in IMAGE_SUFFIXES:
        run_image(estimator, src, dst)
    elif src.suffix.lower() in VIDEO_SUFFIXES:
        run_video(estimator, src, dst)
    else:
        raise SystemExit(f"unsupported source: {src}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m boxing.keypoints", description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="image, video, or directory of images")
    parser.add_argument("dst", type=Path, help="output .json, .jsonl, or directory (see module docs)")
    parser.add_argument("--threshold", type=float, default=0.5, help="minimum person score (default: 0.5)")
    parser.add_argument(
        "--all-people",
        dest="boxers_only",
        action="store_false",
        help="keep everyone; by default only red/blue boxers (by torso colour) are kept",
    )
    args = parser.parse_args(argv)

    if not args.src.exists():
        parser.error(f"source does not exist: {args.src}")

    estimator = PoseEstimator.from_hub(threshold=args.threshold, boxers_only=args.boxers_only)
    run(args.src, args.dst, estimator)


if __name__ == "__main__":
    main()

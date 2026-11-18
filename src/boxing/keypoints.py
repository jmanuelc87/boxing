"""Extract COCO person keypoints with the ``pictograph/pg-coco-pose`` model.

The model is an RF-DETR keypoint detector published as ONNX, so it runs through
ONNX Runtime rather than ``transformers``.

    python -m boxing.keypoints SRC DST [--threshold 0.5] [--role-model PATH] [--all-people] [--color-rules] [--workers N] [--batch-size B]

By default each person is classified by the trained role model (``role_model.pt`` from
``boxing.train_role_classifier``, default ``data/processed/role-classifier/role_model.pt``) and
only the red and blue boxers are kept; referees, audience and anyone else are dropped from every
frame. Pass ``--color-rules`` to use the torso-colour heuristics (``boxing.color_roles``) instead,
or ``--all-people`` to keep everyone.

SRC is an image, a directory of images, a dataset root, or a video:

* image     -> DST is a ``.json`` file
* directory -> DST is a directory; one ``.json`` per image, same relative paths
* dataset   -> a root with ``labels.json`` and split files (e.g. ``data/processed/punch-seq``).
  Every frame referenced by the splits is processed; DST is a directory that mirrors
  the dataset's image paths, one ``.json`` per frame
* video     -> DST is a ``.jsonl`` file; one line per frame

Each record lists the detected people. A person has a pixel box ``[x1, y1, x2, y2]``,
a detection ``score``, and 17 keypoints ``{"name", "x", "y", "score"}`` in COCO order.

``--batch-size B`` (directory and dataset sources) sends B images through the model per forward pass.

``--workers N`` (directory and dataset sources only) runs N worker processes, each with its own copy of the
model; ``0`` uses every core. It pays off on the CPU; on a single GPU (MPS/CUDA) the workers share the device
and mostly add memory use, so keep N small there.

Weights are downloaded from the Hugging Face Hub on first use and cached. The model
is licensed for non-commercial use only; see the model card.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from boxing.color_roles import classify_person

BOXER_ROLES = {"boxer_red", "boxer_blue"}
DEFAULT_ROLE_MODEL = Path("data/processed/role-classifier/role_model.pt")

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
    index: int = 0,
) -> list[Person]:
    """Turn the raw ``pred_*`` tensors of image ``index`` in the batch into people in pixel coordinates.

    ``outputs`` holds a batch of B: ``pred_boxes`` (B, Q, 4) normalised cxcywh,
    ``pred_logits`` (B, Q, C+1) with the background as the last slot, and
    ``pred_keypoints`` (B, Q, R, 8) with R >= len(names).
    """
    boxes = outputs["pred_boxes"][index]
    logits = outputs["pred_logits"][index]
    keypoints = outputs["pred_keypoints"][index]

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


def keep_boxers(image_bgr: np.ndarray, people: list[Person], role_model=None) -> list[Person]:
    """Drop everyone who is not a red or blue boxer (referee, audience, ...).

    ``role_model`` is a ``boxing.train_role_classifier.RoleClassifier``; without one, the
    torso-colour rules decide.
    """
    if not people:
        return people
    if role_model is None:
        return [p for p in people if classify_person(image_bgr, asdict(p)) in BOXER_ROLES]
    roles = role_model.classify(image_bgr, [p.box for p in people])
    return [p for p, role in zip(people, roles) if role in BOXER_ROLES]


class PoseEstimator:
    def __init__(
        self,
        onnx_path: str | Path,
        config_path: str | Path,
        threshold: float = 0.5,
        providers: list[str] | None = None,
        boxers_only: bool = False,
        role_model=None,
    ):
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))["_pictograph"]
        self.input_h, self.input_w = config["input_shape"]
        self.names: list[str] = config["class_names"]
        self.threshold = threshold
        self.boxers_only = boxers_only
        self.role_model = role_model

        import onnxruntime as ort  # lazy: the YOLO backend does not need it

        available = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # A fixed batch dimension (an int, not a symbolic name or None) caps the batch size the model accepts.
        self.max_batch: int | None = inp.shape[0] if isinstance(inp.shape[0], int) else None

    @classmethod
    def from_hub(cls, threshold: float = 0.5, boxers_only: bool = False, role_model=None) -> PoseEstimator:
        return cls(
            hf_hub_download(REPO_ID, ONNX_FILE),
            hf_hub_download(REPO_ID, CONFIG_FILE),
            threshold=threshold,
            boxers_only=boxers_only,
            role_model=role_model,
        )

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """BGR image -> CHW float32. The frame is stretched to the input size."""
        resized = cv2.resize(image_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1))

    def predict_batch(self, images: list[np.ndarray]) -> list[list[Person]]:
        """One forward pass per chunk of images; images may differ in size (each is resized to the input)."""
        results: list[list[Person]] = []
        step = min(len(images), self.max_batch or len(images)) or 1
        for start in range(0, len(images), step):
            chunk = images[start : start + step]
            batch = np.stack([self.preprocess(im) for im in chunk])
            raw = self.session.run(None, {self.input_name: batch})
            outputs = {o.name: r for o, r in zip(self.session.get_outputs(), raw)}
            for i, im in enumerate(chunk):
                h, w = im.shape[:2]
                people = decode(outputs, self.names, w, h, self.threshold, index=i)
                results.append(keep_boxers(im, people, self.role_model) if self.boxers_only else people)
        return results

    def predict(self, image_bgr: np.ndarray) -> list[Person]:
        return self.predict_batch([image_bgr])[0]


COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]  # fmt: skip


class YoloPoseEstimator:
    """Ultralytics YOLO pose model. Runs on Apple's GPU with ``device="mps"``.

    Ultralytics is AGPL-3.0 licensed; see its license terms before distributing.
    """

    def __init__(
        self,
        weights: str = "yolo11n-pose.pt",
        threshold: float = 0.5,
        device: str | None = None,
        boxers_only: bool = False,
        role_model=None,
    ):
        import torch
        from ultralytics import YOLO

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = YOLO(weights)
        self.device = device
        self.threshold = threshold
        self.boxers_only = boxers_only
        self.role_model = role_model

    def predict_batch(self, images: list[np.ndarray]) -> list[list[Person]]:
        """Ultralytics letterboxes and stacks a list of frames into one forward pass."""
        results = self.model.predict(images, device=self.device, conf=self.threshold, verbose=False)
        out = []
        for im, result in zip(images, results):
            people = self._people(result)
            out.append(keep_boxers(im, people, self.role_model) if self.boxers_only else people)
        return out

    def predict(self, image_bgr: np.ndarray) -> list[Person]:
        return self.predict_batch([image_bgr])[0]

    @staticmethod
    def _people(result) -> list[Person]:
        people: list[Person] = []
        if result.boxes is None or len(result.boxes) == 0:
            return people

        xy = result.keypoints.xy.cpu().numpy()
        kp_conf = result.keypoints.conf
        kp_conf = kp_conf.cpu().numpy() if kp_conf is not None else np.ones(xy.shape[:2])
        for box, score, pts, conf in zip(
            result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), xy, kp_conf
        ):
            joints = [
                Joint(name=name, x=float(x), y=float(y), score=float(c))
                for name, (x, y), c in zip(COCO_NAMES, pts, conf)
            ]
            people.append(Person(box=[float(v) for v in box], score=float(score), keypoints=joints))
        return people


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


def run_batch(estimator: PoseEstimator | YoloPoseEstimator, jobs: list[tuple[Path, Path]]) -> None:
    """Predict ``jobs`` (source, destination) in one batched call and write one ``.json`` per image."""
    images = [_read_image(src) for src, _ in jobs]
    for (_, dst), image, people in zip(jobs, images, estimator.predict_batch(images)):
        dst.parent.mkdir(parents=True, exist_ok=True)
        h, w = image.shape[:2]
        dst.write_text(json.dumps(record(people, w, h), indent=2), encoding="utf-8")


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


_WORKER_ESTIMATOR: PoseEstimator | YoloPoseEstimator | None = None


def _init_worker(spec: dict) -> None:
    """Build this worker's own estimator once, and keep each worker to one CPU thread."""
    global _WORKER_ESTIMATOR
    cv2.setNumThreads(1)
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    _WORKER_ESTIMATOR = build_estimator(**spec)


def _worker_batch(jobs: list[tuple[Path, Path]]) -> int:
    try:
        run_batch(_WORKER_ESTIMATOR, jobs)
        return len(jobs)
    except SystemExit as exc:  # SystemExit would kill the worker instead of reporting the error
        raise RuntimeError(str(exc)) from None


def run_directory(
    estimator: PoseEstimator | YoloPoseEstimator | None,
    src: Path,
    dst: Path,
    workers: int = 1,
    spec: dict | None = None,
    batch_size: int = 1,
) -> None:
    if is_dataset(src):
        images = [src / rel for rel in dataset_images(src)]
        missing = [p for p in images if not p.is_file()]
        if missing:
            raise SystemExit(f"{len(missing)} image(s) listed in the splits are missing, e.g. {missing[0]}")
    else:
        images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no images found under {src}")
    jobs = [(path, (dst / path.relative_to(src)).with_suffix(".json")) for path in images]
    batches = [jobs[i : i + batch_size] for i in range(0, len(jobs), batch_size)]
    n_workers = (os.cpu_count() or 1) if workers == 0 else max(1, workers)
    if n_workers > 1 and spec is not None and len(jobs) > 1:
        # spawn, not fork: forking after torch/CUDA/ONNX Runtime has started threads is unsafe
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx, initializer=_init_worker, initargs=(spec,)
        ) as pool:
            with tqdm(total=len(jobs), unit="img") as bar:
                for n in pool.map(_worker_batch, batches):
                    bar.update(n)
    else:
        with tqdm(total=len(jobs), unit="img") as bar:
            for chunk in batches:
                run_batch(estimator, chunk)
                bar.update(len(chunk))


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


def build_estimator(
    backend: str = "yolo",
    weights: str = "yolo11m-pose.pt",
    threshold: float = 0.5,
    device: str | None = None,
    boxers_only: bool = True,
    role_model_path: Path | None = None,
) -> PoseEstimator | YoloPoseEstimator:
    """Build an estimator from plain, picklable settings, so a worker process can build its own."""
    role_model = None
    if boxers_only and role_model_path is not None:
        from boxing.train_role_classifier import RoleClassifier

        role_model = RoleClassifier(role_model_path)
    if backend == "yolo":
        return YoloPoseEstimator(
            weights=weights, threshold=threshold, device=device, boxers_only=boxers_only, role_model=role_model
        )
    return PoseEstimator.from_hub(threshold=threshold, boxers_only=boxers_only, role_model=role_model)


def run(
    src: Path,
    dst: Path,
    estimator: PoseEstimator | YoloPoseEstimator | None,
    workers: int = 1,
    spec: dict | None = None,
    batch_size: int = 1,
) -> None:
    if src.is_dir():
        run_directory(estimator, src, dst, workers, spec, batch_size)
    elif src.suffix.lower() in IMAGE_SUFFIXES:
        run_image(estimator, src, dst)
    elif src.suffix.lower() in VIDEO_SUFFIXES:
        if workers != 1:
            print("--workers has no effect on a video (frames are processed in order); running in one process")
        run_video(estimator, src, dst)
    else:
        raise SystemExit(f"unsupported source: {src}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m boxing.keypoints", description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="image, video, or directory of images")
    parser.add_argument("dst", type=Path, help="output .json, .jsonl, or directory (see module docs)")
    parser.add_argument("--threshold", type=float, default=0.5, help="minimum person score (default: 0.5)")
    parser.add_argument(
        "--backend",
        choices=["yolo", "rtdetr"],
        default="yolo",
        help="yolo: Ultralytics pose model, runs on MPS/CUDA (default); rtdetr: pictograph ONNX model",
    )
    parser.add_argument(
        "--weights", default="yolo11m-pose.pt", help="YOLO pose weights, used with --backend yolo"
    )
    parser.add_argument("--device", default=None, help="yolo device: mps, cuda, or cpu (default: mps if available)")
    parser.add_argument(
        "--role-model",
        type=Path,
        default=DEFAULT_ROLE_MODEL,
        help="trained role model used as the boxer filter (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker processes for directory/dataset sources, each with its own model; 0 = all cores (default 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="images per forward pass for directory/dataset sources (default 8; 1 = one by one)",
    )
    parser.add_argument(
        "--color-rules",
        action="store_true",
        help="filter by torso colour rules instead of the role model",
    )
    parser.add_argument(
        "--all-people",
        dest="boxers_only",
        action="store_false",
        help="keep everyone; by default only red/blue boxers are kept",
    )
    args = parser.parse_args(argv)

    if not args.src.exists():
        parser.error(f"source does not exist: {args.src}")

    role_model_path = None
    if args.boxers_only and not args.color_rules:
        if not args.role_model.exists():
            parser.error(f"role model not found: {args.role_model} (train one, or pass --color-rules)")
        role_model_path = args.role_model

    spec = dict(
        backend=args.backend,
        weights=args.weights,
        threshold=args.threshold,
        device=args.device,
        boxers_only=args.boxers_only,
        role_model_path=role_model_path,
    )
    parallel = args.workers != 1 and args.src.is_dir()
    # Parallel workers each build their own estimator, so the parent skips loading one.
    estimator = None if parallel else build_estimator(**spec)
    run(args.src, args.dst, estimator, args.workers, spec if parallel else None, max(1, args.batch_size))


if __name__ == "__main__":
    main()

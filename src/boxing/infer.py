"""Streaming inference for the clip-based punch detector.

The detector looks at ``seq_len`` consecutive frames at once, so a stream is
handled with a rolling buffer: each pushed frame is appended, and once the
buffer is full the model runs on it. Predictions are read from the centre frame
of the buffer, the one with context on both sides, as in training.

    from boxing.infer import PunchDetector

    det = PunchDetector.from_pretrained("outputs/boxing-detector", stride=2)
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pred = det.push(frame)            # None until a clip is ready
        if pred is not None:
            for d in pred.detections:     # d.box is (x1, y1, x2, y2) in frame pixels
                ...

``frame`` is a BGR ``numpy`` array, as OpenCV returns it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from boxing.dataset import MEAN, STD

# Defaults match the training configuration (notebooks/train.ipynb).
DEFAULT_IMAGE_SIZE = 640
DEFAULT_SEQ_LEN = 16
DEFAULT_SCORE_THRESH = 0.3


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    score: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels of the original frame


@dataclass(frozen=True)
class Prediction:
    frame_index: int  # 0-based index, in push order, of the frame the detections belong to
    detections: list[Detection] = field(default_factory=list)


class PunchDetector:
    """Rolling-window detector over a stream of BGR frames.

    ``stride`` runs the model once every ``stride`` frames once the buffer is
    full (1 = every frame). Frames between runs return ``None``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        id2label: dict[int, str],
        *,
        image_size: int = DEFAULT_IMAGE_SIZE,
        seq_len: int = DEFAULT_SEQ_LEN,
        stride: int = 1,
        score_thresh: float = DEFAULT_SCORE_THRESH,
        device: torch.device | str | None = None,
    ):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.device = torch.device(device) if device is not None else _default_device()
        self.model = model.to(self.device).eval()
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.image_size = image_size
        self.seq_len = seq_len
        self.stride = stride
        self.score_thresh = score_thresh
        self.reset()

    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs: Any) -> PunchDetector:
        """Load a checkpoint saved with ``BoxingDetector.save_pretrained``."""
        from boxing.model import BoxingDetector

        model = BoxingDetector.from_pretrained(str(path))
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        return cls(model, id2label, **kwargs)

    def reset(self) -> None:
        """Forget buffered frames, e.g. when the stream jumps or restarts."""
        self._tensors: deque[torch.Tensor] = deque(maxlen=self.seq_len)
        self._n_pushed = 0
        self._since_run = self.stride  # first run happens as soon as the buffer is full

    def push(self, frame: np.ndarray) -> Prediction | None:
        """Add one BGR frame. Returns a ``Prediction`` when the model runs, else ``None``."""
        self._tensors.append(preprocess_frame(frame, self.image_size))
        index = self._n_pushed
        self._n_pushed += 1
        if len(self._tensors) < self.seq_len:
            return None
        self._since_run += 1
        if self._since_run < self.stride:
            return None
        self._since_run = 0

        clip = torch.stack(list(self._tensors))[None].to(self.device)  # (1, T, 3, S, S)
        with torch.inference_mode():
            out = self.model(pixel_values=clip)
        centre = self.seq_len // 2
        # The window ends at `index`, so its centre is `centre` frames into a window
        # that starts `seq_len - 1` frames before `index`.
        centre_index = index - (self.seq_len - 1) + centre
        return Prediction(
            frame_index=centre_index,
            detections=decode(
                out.logits[centre].cpu(),
                out.pred_boxes[centre].cpu(),
                frame_hw=_frame_hw(frame),
                id2label=self.id2label,
                score_thresh=self.score_thresh,
            ),
        )


def preprocess_frame(frame: np.ndarray, image_size: int) -> torch.Tensor:
    """BGR frame -> normalised RGB tensor ``(3, S, S)``, the same transform as training."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
    return (x - MEAN) / STD


def decode(
    logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    *,
    frame_hw: tuple[int, int],
    id2label: dict[int, str],
    score_thresh: float = DEFAULT_SCORE_THRESH,
) -> list[Detection]:
    """Turn one frame's raw outputs into pixel-space detections.

    ``logits`` is ``(queries, classes)`` and ``pred_boxes`` is ``(queries, 4)`` in
    normalised ``(cx, cy, w, h)``. Each query is scored by sigmoid, and its best
    class is kept if the score reaches ``score_thresh``. Boxes are clipped to the frame.
    """
    h, w = frame_hw
    probs = logits.sigmoid()
    scores, classes = probs.max(dim=-1)
    out: list[Detection] = []
    for s, c, (cx, cy, bw, bh) in zip(scores.tolist(), classes.tolist(), pred_boxes.tolist()):
        if s < score_thresh:
            continue
        x1 = min(max(int((cx - bw / 2) * w), 0), w - 1)
        y1 = min(max(int((cy - bh / 2) * h), 0), h - 1)
        x2 = min(max(int((cx + bw / 2) * w), 0), w - 1)
        y2 = min(max(int((cy + bh / 2) * h), 0), h - 1)
        out.append(Detection(class_id=c, label=id2label[c], score=s, box=(x1, y1, x2, y2)))
    out.sort(key=lambda d: d.score, reverse=True)
    return out


def _frame_hw(frame: np.ndarray) -> tuple[int, int]:
    return int(frame.shape[0]), int(frame.shape[1])


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

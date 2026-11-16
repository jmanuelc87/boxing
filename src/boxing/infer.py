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

Running on a Jetson with TensorRT
---------------------------------
A TensorRT engine is tied to the GPU and TensorRT version that built it, so the
conversion is two steps: ONNX anywhere, engine on the Jetson itself.

    # 1. any machine with the checkpoint (writes model.onnx + model.labels.json)
    python -m boxing.infer export-onnx outputs/boxing-detector model.onnx

    # 2. on the Jetson (JetPack provides tensorrt; trtexec works too)
    python -m boxing.infer build-engine model.onnx model.engine --fp16

    # 3. inference; same API as above
    det = PunchDetector.from_engine("model.engine", stride=2)

The engine has a fixed input shape ``(1, seq_len, 3, S, S)``, taken from the export.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
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
        model: torch.nn.Module | TRTModel,
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
        # A TRTModel is already bound to the GPU; only torch modules need moving.
        self.model = model.to(self.device).eval() if isinstance(model, torch.nn.Module) else model
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

    @classmethod
    def from_engine(cls, path: str | Path, **kwargs: Any) -> PunchDetector:
        """Load a TensorRT engine built with ``build-engine``.

        ``seq_len`` and ``image_size`` come from the engine's input shape, and class
        names from the ``.labels.json`` next to the ONNX file / engine (``id2label``
        can be passed instead).
        """
        model = TRTModel(path)
        _, seq_len, _, size, _ = model.input_shape
        kwargs.setdefault("seq_len", seq_len)
        kwargs.setdefault("image_size", size)
        if (kwargs["seq_len"], kwargs["image_size"]) != (seq_len, size):
            raise ValueError(f"engine expects seq_len={seq_len}, image_size={size}")
        kwargs.setdefault("device", "cuda")
        id2label = kwargs.pop("id2label", None)
        if id2label is None:
            id2label = _load_labels(Path(path))
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


class _ExportWrapper(torch.nn.Module):
    """Clip tensor in, ``(logits, pred_boxes)`` tuple out: a graph ONNX/TensorRT can trace."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(pixel_values=clip)
        return out.logits, out.pred_boxes


def export_onnx(
    checkpoint: str | Path,
    onnx_path: str | Path,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seq_len: int = DEFAULT_SEQ_LEN,
    opset: int = 17,
) -> None:
    """Export a ``save_pretrained`` checkpoint to ONNX with a fixed ``(1, T, 3, S, S)`` input.

    Also writes ``<onnx_path stem>.labels.json`` with the class names.
    """
    from boxing.model import BoxingDetector

    model = BoxingDetector.from_pretrained(str(checkpoint)).eval()
    onnx_path = Path(onnx_path)
    dummy = torch.zeros(1, seq_len, 3, image_size, image_size)
    with torch.inference_mode():
        torch.onnx.export(
            _ExportWrapper(model),
            (dummy,),
            str(onnx_path),
            input_names=["clip"],
            output_names=["logits", "pred_boxes"],
            opset_version=opset,
            dynamo=False,
        )
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    onnx_path.with_suffix(".labels.json").write_text(json.dumps(id2label))


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    fp16: bool = True,
    workspace_gb: float = 2.0,
) -> None:
    """Compile an ONNX file into a TensorRT engine. Run this on the Jetson that will use it."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed (see log above)")
    engine_path = Path(engine_path)
    engine_path.write_bytes(bytes(serialized))
    labels = Path(onnx_path).with_suffix(".labels.json")
    if labels.exists():
        engine_path.with_suffix(".labels.json").write_text(labels.read_text())


class TRTModel:
    """A TensorRT engine that quacks like the torch model: ``model(pixel_values=clip)``.

    Inputs and outputs are CUDA torch tensors, so no extra CUDA bindings are needed.
    """

    _DTYPES = {"FLOAT": torch.float32, "HALF": torch.float16, "INT32": torch.int32}

    def __init__(self, engine_path: str | Path):
        import tensorrt as trt

        self._trt = trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"Could not load {engine_path}; engines only run on the GPU and TensorRT "
                "version that built them. Rebuild with `build-engine` on this device."
            )
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        mode = trt.TensorIOMode
        self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == mode.INPUT)
        self.output_names = [n for n in names if self.engine.get_tensor_mode(n) == mode.OUTPUT]
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.device = torch.device("cuda")

    def __call__(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        if tuple(pixel_values.shape) != self.input_shape:
            raise ValueError(f"engine expects {self.input_shape}, got {tuple(pixel_values.shape)}")
        x = pixel_values.to(self.device, torch.float32).contiguous()
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        outputs = {}
        for name in self.output_names:
            dtype = self._DTYPES[self.engine.get_tensor_dtype(name).name]
            buf = torch.empty(tuple(self.engine.get_tensor_shape(name)), dtype=dtype, device=self.device)
            self.context.set_tensor_address(name, buf.data_ptr())
            outputs[name] = buf
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        stream.synchronize()
        return SimpleNamespace(logits=outputs["logits"].float(), pred_boxes=outputs["pred_boxes"].float())


def _load_labels(engine_path: Path) -> dict[int, str]:
    labels = engine_path.with_suffix(".labels.json")
    if not labels.exists():
        raise FileNotFoundError(f"{labels} not found; pass id2label=... to from_engine")
    return {int(k): v for k, v in json.loads(labels.read_text()).items()}


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m boxing.infer", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export-onnx", help="checkpoint -> ONNX (+ labels json)")
    p.add_argument("checkpoint")
    p.add_argument("onnx")
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--opset", type=int, default=17)

    p = sub.add_parser("build-engine", help="ONNX -> TensorRT engine (run on the Jetson)")
    p.add_argument("onnx")
    p.add_argument("engine")
    p.add_argument("--fp16", action="store_true", help="enable FP16 kernels")
    p.add_argument("--workspace-gb", type=float, default=2.0)

    args = parser.parse_args(argv)
    if args.cmd == "export-onnx":
        export_onnx(args.checkpoint, args.onnx, image_size=args.image_size, seq_len=args.seq_len, opset=args.opset)
    else:
        build_engine(args.onnx, args.engine, fp16=args.fp16, workspace_gb=args.workspace_gb)


if __name__ == "__main__":
    main()

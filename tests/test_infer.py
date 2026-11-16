from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxing.infer import PunchDetector, decode, preprocess_frame

ID2LABEL = {0: "jab", 1: "cross"}
SIZE, SEQ = 32, 4


class StubModel(torch.nn.Module):
    """Returns a fixed output per call and records the clip shapes it was given."""

    def __init__(self, logit: float = 5.0):
        super().__init__()
        self.logit = logit
        self.calls: list[tuple[int, ...]] = []

    def forward(self, pixel_values):
        b, t = pixel_values.shape[:2]
        self.calls.append(tuple(pixel_values.shape))
        logits = torch.full((b * t, 2, 2), -10.0)
        logits[:, 0, 0] = self.logit  # query 0 is confident about class 0 ("jab"); query 1 is not
        boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]]).repeat(b * t, 1, 1)
        return SimpleNamespace(logits=logits, pred_boxes=boxes)


def frame(h=48, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


def make(**kw):
    kw.setdefault("image_size", SIZE)
    kw.setdefault("seq_len", SEQ)
    kw.setdefault("device", "cpu")
    return PunchDetector(StubModel(), ID2LABEL, **kw)


def test_no_prediction_until_buffer_is_full():
    det = make()
    outs = [det.push(frame()) for _ in range(SEQ - 1)]
    assert outs == [None] * (SEQ - 1)
    assert det.model.calls == []


def test_first_prediction_is_centre_of_first_full_window():
    det = make()
    outs = [det.push(frame()) for _ in range(SEQ)]
    assert outs[-1] is not None
    assert outs[-1].frame_index == SEQ // 2  # window covers frames 0..3, centre is 2
    assert det.model.calls == [(1, SEQ, 3, SIZE, SIZE)]


def test_frame_index_advances_with_each_push():
    det = make()
    idx = [p.frame_index for p in (det.push(frame()) for _ in range(SEQ + 3)) if p is not None]
    assert idx == [SEQ // 2 + i for i in range(4)]


def test_stride_runs_model_every_n_frames():
    det = make(stride=3)
    preds = [det.push(frame()) for _ in range(SEQ + 6)]
    ran = [i for i, p in enumerate(preds) if p is not None]
    assert ran == [SEQ - 1, SEQ + 2, SEQ + 5]  # first run immediately, then every 3 frames
    assert len(det.model.calls) == 3


def test_reset_clears_buffer():
    det = make()
    for _ in range(SEQ):
        det.push(frame())
    det.reset()
    assert det.push(frame()) is None


def test_detections_are_scaled_to_original_frame():
    det = make(score_thresh=0.5)
    pred = None
    for _ in range(SEQ):
        pred = det.push(frame(h=48, w=64))
    assert pred is not None and len(pred.detections) == 1
    d = pred.detections[0]
    # Box centred at (0.5, 0.5) with size 0.5 -> (16, 12) to (48, 36) in a 64x48 frame.
    assert d.box == (16, 12, 48, 36)
    assert d.label == "jab" and d.class_id == 0
    assert d.score == pytest.approx(torch.sigmoid(torch.tensor(5.0)).item(), rel=1e-5)


def test_low_scores_are_dropped():
    det = PunchDetector(StubModel(logit=-5.0), ID2LABEL, image_size=SIZE, seq_len=SEQ, device="cpu")
    for _ in range(SEQ):
        pred = det.push(frame())
    assert pred is not None and pred.detections == []


def test_decode_clips_boxes_to_frame():
    logits = torch.tensor([[10.0, -10.0]])
    boxes = torch.tensor([[0.875, 0.875, 0.25, 0.25]])  # x1 = y1 = 0.75 exactly; runs off the bottom-right
    (d,) = decode(logits, boxes, frame_hw=(100, 200), id2label=ID2LABEL, score_thresh=0.5)
    assert d.box == (150, 75, 199, 99)


def test_preprocess_converts_bgr_to_normalised_rgb():
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # blue channel in OpenCV order
    x = preprocess_frame(bgr, 8)
    assert x.shape == (3, 8, 8)
    # After RGB conversion the blue value sits in channel 2, normalised with ImageNet stats.
    assert x[2, 0, 0].item() == pytest.approx((1.0 - 0.406) / 0.225, rel=1e-4)
    assert x[0, 0, 0].item() == pytest.approx((0.0 - 0.485) / 0.229, rel=1e-4)


@pytest.mark.parametrize("kwargs", [{"seq_len": 0}, {"stride": 0}])
def test_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)

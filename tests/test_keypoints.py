import numpy as np
import pytest

from boxing.keypoints import KP_SCORE_COL, decode

NAMES = ["nose", "left_eye", "right_eye"]


def outputs(score_logit: float, queries: int = 2, rows: int = 4):
    logits = np.full((1, queries, 4), -10.0, dtype=np.float32)  # 3 classes + background (last slot)
    logits[0, 0, 0] = score_logit
    boxes = np.zeros((1, queries, 4), dtype=np.float32)
    boxes[0, 0] = [0.5, 0.5, 0.25, 0.5]  # centred, quarter-width, half-height
    kps = np.zeros((1, queries, rows, 8), dtype=np.float32)
    for j in range(len(NAMES)):
        kps[0, 0, j, 0] = 0.5  # x, normalised
        kps[0, 0, j, 1] = 0.25 + 0.1 * j  # y, normalised
        kps[0, 0, j, KP_SCORE_COL] = 0.9
    return {"pred_boxes": boxes, "pred_logits": logits, "pred_keypoints": kps}


def test_decode_returns_person_in_pixel_coordinates():
    people = decode(outputs(score_logit=5.0), NAMES, width=200, height=100, threshold=0.5)

    assert len(people) == 1
    person = people[0]
    assert person.score > 0.99
    assert person.box == [75.0, 25.0, 125.0, 75.0]
    assert [j.name for j in person.keypoints] == NAMES
    assert person.keypoints[0].x == 100.0
    assert person.keypoints[0].y == 25.0
    assert person.keypoints[2].y == 45.0
    assert person.keypoints[0].score == pytest.approx(0.9)


def test_decode_drops_queries_below_threshold():
    assert decode(outputs(score_logit=-5.0), NAMES, width=200, height=100, threshold=0.5) == []

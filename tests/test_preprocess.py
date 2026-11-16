import json

import cv2
import numpy as np
import pytest

from boxing.preprocess import RAW_LABELS, clip_window, interpolate_track, merge_tracks, preprocess


def key(frame, points, outside=False):
    return {"type": "rectangle", "frame": frame, "points": points, "outside": outside}


def test_interpolate_track_fills_frames_and_stops_at_outside():
    boxes = interpolate_track([key(14, [0, 0, 8, 8], outside=True), key(10, [0, 0, 4, 4])])
    assert sorted(boxes) == [10, 11, 12, 13]  # the outside keyframe itself is not visible
    assert boxes[12] == [0, 0, 6, 6]  # CVAT interpolates towards the outside keyframe's box


def test_interpolate_track_is_linear_between_keyframes():
    boxes = interpolate_track([key(0, [0, 0, 10, 10]), key(4, [4, 0, 14, 10])])
    assert boxes[2] == [2, 0, 12, 10]
    assert boxes[4] == [4, 0, 14, 10]  # last keyframe is visible when no outside follows


def test_merge_tracks_keeps_overlapping_boxes():
    merged = merge_tracks([(0, {5: [1, 1, 2, 2]}), (3, {5: [3, 3, 4, 4], 6: [3, 3, 4, 4]})])
    assert merged[5] == [(0, [1, 1, 2, 2]), (3, [3, 3, 4, 4])]
    assert merged[6] == [(3, [3, 3, 4, 4])]


@pytest.mark.parametrize(
    ("first", "last", "n_frames", "start"),
    [
        (10, 14, 100, 4),  # centred: centre 12, 8 frames before it
        (1, 3, 100, 0),  # clamped at the start of the video
        (97, 99, 100, 84),  # clamped at the end of the video
        (0, 40, 100, 12),  # longer than seq_len: the centre 16 frames (centre 20)
    ],
)
def test_clip_window(first, last, n_frames, start):
    assert clip_window(first, last, 16, n_frames) == start


def test_clip_window_rejects_short_video():
    with pytest.raises(ValueError):
        clip_window(0, 2, 16, 10)


# --- end to end on a synthetic task -------------------------------------------------

W, H, N_FRAMES, SEQ = 64, 48, 40, 8


def make_task(root):
    """One task: a 40-frame video and three tracks (two of them overlap on frames 13-14)."""
    task = root / "task_kam2_gh000001"
    (task / "data").mkdir(parents=True)
    writer = cv2.VideoWriter(str(task / "data" / "GH000001.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), 50, (W, H))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4v video in this environment")
    for i in range(N_FRAMES):
        writer.write(np.full((H, W, 3), i * 5, dtype=np.uint8))
    writer.release()
    tracks = [
        # A: frames 10-14
        {"label": RAW_LABELS[0], "shapes": [key(10, [8, 8, 24, 24]), key(15, [8, 8, 24, 24], outside=True)]},
        # B: frames 13-16
        {"label": RAW_LABELS[3], "shapes": [key(13, [30, 10, 50, 30]), key(17, [30, 10, 50, 30], outside=True)]},
        # C: frame 38 only, near the end of the video
        {"label": RAW_LABELS[5], "shapes": [key(38, [4, 4, 20, 20])]},
    ]
    (task / "annotations.json").write_text(json.dumps([{"version": 1, "tags": [], "shapes": [], "tracks": tracks}]))


@pytest.fixture
def processed(tmp_path):
    make_task(tmp_path / "raw")
    dst = tmp_path / "out"
    preprocess(tmp_path / "raw", dst, val_fraction=0.0, seq_len=SEQ, show_progress=False)
    records = [json.loads(line) for line in (dst / "train.jsonl").read_text().splitlines()]
    return dst, {r["track"]: r for r in records}


def test_one_clip_per_track_with_seq_len_frames(processed):
    dst, clips = processed
    assert sorted(clips) == [0, 1, 2]
    for r in clips.values():
        frames = [int(f["image"][-10:-4]) for f in r["frames"]]
        assert frames == list(range(r["start"], r["start"] + SEQ))
        assert all((dst / f["image"]).is_file() for f in r["frames"])
        assert (r["width"], r["height"]) == (W, H)  # never upscaled
    assert (dst / "val.jsonl").read_text() == ""
    assert json.loads((dst / "labels.json").read_text())["seq_len"] == SEQ


def test_clip_targets_include_context_and_overlap(processed):
    _, clips = processed
    a = clips[0]
    assert (a["label"], a["start"]) == (0, 8)  # centre 12 -> frames 8..15
    labels = [sorted(f["class_labels"]) for f in a["frames"]]
    assert labels == [[], [], [0], [0], [0], [0, 3], [0, 3], [3]]
    assert all(len(f["boxes"]) == len(f["class_labels"]) for f in a["frames"])
    # box of track A in normalised cxcywh: x 8..24 of 64, y 8..24 of 48
    assert a["frames"][2]["boxes"][0] == pytest.approx([16 / W, 16 / H, 16 / W, 16 / H])


def test_clip_is_clamped_to_the_video_end(processed):
    _, clips = processed
    c = clips[2]
    assert (c["label"], c["start"]) == (5, N_FRAMES - SEQ)
    assert [f["class_labels"] for f in c["frames"]][-2:] == [[5], []]  # frames 38, 39

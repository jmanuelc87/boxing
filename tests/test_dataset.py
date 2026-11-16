import json

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from boxing.dataset import PunchSequenceDataset, collate_clips, flatten_clips, load_id2label

T, SIZE = 4, 32


@pytest.fixture
def root(tmp_path):
    """A processed dir with two 4-frame clips; the first frame of each clip is empty."""
    (tmp_path / "images" / "t").mkdir(parents=True)
    records = []
    for clip in range(2):
        frames = []
        for k in range(T):
            name = f"images/t/{clip * T + k:06d}.jpg"
            Image.new("RGB", (64, 36), (clip * 100, k * 50, 0)).save(tmp_path / name)
            empty = k == 0
            frames.append(
                {
                    "image": name,
                    "class_labels": [] if empty else [k],
                    "boxes": [] if empty else [[0.5, 0.5, 0.1, 0.2]],
                }
            )
        records.append({"task": "t", "track": clip, "label": 1, "start": clip * T, "width": 64, "height": 36, "frames": frames})
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    labels = {"names": ["a", "b"], "id2label": {"0": "a", "1": "b"}, "label2id": {"a": 0, "b": 1}, "seq_len": T}
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    return tmp_path


def test_item_shapes_and_targets(root):
    ds = PunchSequenceDataset(root, "train", image_size=SIZE)
    assert len(ds) == 2 and ds.seq_len == T
    x, targets = ds[0]
    assert x.shape == (T, 3, SIZE, SIZE) and x.dtype == torch.float32
    assert len(targets) == T
    empty = targets[0]
    assert empty["class_labels"].shape == (0,) and empty["class_labels"].dtype == torch.long
    assert empty["boxes"].shape == (0, 4) and empty["boxes"].dtype == torch.float32
    assert targets[2]["class_labels"].tolist() == [2]
    assert targets[2]["boxes"].shape == (1, 4)


def test_collate_and_flatten_keep_clip_order(root):
    ds = PunchSequenceDataset(root, "train", image_size=SIZE)
    x, targets = next(iter(DataLoader(ds, batch_size=2, collate_fn=collate_clips)))
    assert x.shape == (2, T, 3, SIZE, SIZE)
    assert [len(clip) for clip in targets] == [T, T]
    flat_x, flat_t = flatten_clips(x, targets)
    assert flat_x.shape == (2 * T, 3, SIZE, SIZE)
    assert len(flat_t) == 2 * T
    assert torch.equal(flat_x[T + 1], x[1, 1])  # clip 1, frame 1
    assert flat_t[T + 1] is targets[1][1]


def test_flatten_rejects_mismatched_targets(root):
    ds = PunchSequenceDataset(root, "train", image_size=SIZE)
    x, targets = collate_clips([ds[0], ds[1]])
    with pytest.raises(ValueError):
        flatten_clips(x, targets[:1])


def test_rejects_per_frame_dataset(root):
    labels = json.loads((root / "labels.json").read_text())
    del labels["seq_len"]
    (root / "labels.json").write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="per-frame"):
        PunchSequenceDataset(root, "train")


def test_load_id2label_uses_int_keys(root):
    assert load_id2label(root) == {0: "a", 1: "b"}

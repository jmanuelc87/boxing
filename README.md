# Boxing punch detection

## Dataset: Olympic Boxing Punch Classification Video Dataset

Raw data lives in `data/raw/punch/`. Each of the 29 folders is one recording session:

```
data/raw/punch/
└── task_kam{2|4}_gh{NNNNNN}/
    ├── annotations.json      # CVAT export (video 1.1 JSON)
    └── data/GH{NNNNNN}.mp4   # source video
```

- `kam2` and `kam4` are the two camera positions. There are 17 `kam2` tasks and 12 `kam4` tasks.
- Videos are 1920×1080, 50 fps, about 3,186 frames (~64 s) per file.

### Annotation format

Each `annotations.json` is a single-element JSON list. The element has four keys:

| key       | contents                                   |
|-----------|--------------------------------------------|
| `version` | CVAT export version (18 in the sample)     |
| `tags`    | always empty                               |
| `shapes`  | always empty (no frame-level shapes)       |
| `tracks`  | list of punch events, one per track        |

Each **track** is one punch event:

```json
{
  "frame": 653,            // first frame of the track
  "group": 0,
  "source": "manual",
  "label": "Chybienie lewą ręką",
  "attributes": [],
  "shapes": [
    {
      "type": "rectangle",
      "points": [x1, y1, x2, y2],   // pixel coordinates, top-left and bottom-right
      "frame": 653,
      "occluded": false,
      "outside": false,              // true = the box disappears on this frame
      "z_order": 0,
      "attributes": []
    }
  ]
}
```

- Shapes are **keyframes**, not one per frame. In the sample, a track spans about 5 keyframes. Interpolate between them to get the box on any frame.
- `outside: true` marks the last frame of the track.
- Boxes are axis-aligned rectangles around the **boxer's** fist/head/body, drawn by hand (`source: "manual"`). No track has attributes.

### Labels

The raw labels are in Polish. There are 8 classes: a target or outcome (head, body, block, miss) combined with a hand (left or right). `preprocess.py` translates each one to a Spanish boxing term and assigns a fixed class id:

| id | raw label (Polish)       | English               | processed name (Spanish) | count |
|---:|--------------------------|-----------------------|--------------------------|------:|
|  0 | `Głowa lewą ręką`        | head, left hand       | `Jab a la cabeza`        | 1,683 |
|  1 | `Głowa prawą ręką`       | head, right hand      | `Cruzado a la cabeza`    |   833 |
|  2 | `Chybienie lewą ręką`    | miss, left hand       | `Jab fallido`            |   734 |
|  3 | `Chybienie prawą ręką`   | miss, right hand      | `Cruzado fallido`        |   449 |
|  4 | `Blok lewą ręką`         | block, left hand      | `Bloqueo de jab`         |   360 |
|  5 | `Blok prawą ręką`        | block, right hand     | `Bloqueo de cruzado`     |   178 |
|  6 | `Korpus lewą ręką`       | body, left hand       | `Jab al cuerpo`          |   193 |
|  7 | `Korpus prawą ręką`      | body, right hand      | `Cruzado al cuerpo`      |   179 |

Counts are across all 29 files. The total is 4,609 tracks and 22,274 keyframe boxes.

**The jab/cross names are an assumption.** The annotations record the hand and the target, not the punch type. Left hand is mapped to *jab* and right hand to *cross* (`cruzado`), which holds for an orthodox (right-handed) boxer. If a boxer is southpaw the mapping is reversed, and hooks and uppercuts cannot be told apart in this data. The class ids come from the order of `RAW_LABELS` in `src/boxing/preprocess.py`, and the Spanish names from `LABELS` in the same file.

Notes:
- Left/right are encoded in the label. Use the label, not the box position, to decide the hand.
- The classes are imbalanced. Head punches are the largest group. Body punches are the smallest.
- Each label combines a **target** (head, body, or block/miss) with a **hand** (left or right). If the task is "which punch was thrown", you may want to merge the hand or the target.

### Dataset stats

- 29 tasks, 4,609 tracks, 22,274 keyframe boxes.
- Tracks per file vary a lot (50 to 162 in the samples checked).
- Only the `rectangle` shape type is used.

## Processed dataset

`BoxingDetector` (`src/boxing/model.py`) is an object detector. It needs per-frame boxes and a class for each box. `src/boxing/preprocess.py` turns the raw CVAT data into that:

```
uv run python -m boxing.preprocess data/raw/punch data/processed/punch
uv run python -m boxing.preprocess data/raw/punch data/processed/punch-test --test   # quick check
```

| option           | default | meaning                                                          |
|------------------|--------:|------------------------------------------------------------------|
| `--max-size`     |     640 | resize so the longest image side is this many pixels             |
| `--val-fraction` |     0.2 | fraction of tasks held out for validation                        |
| `--seed`         |       0 | seed for the train/val split                                     |
| `--test`         |     off | small end-to-end run: 3 tasks, first 20 annotated frames of each |

### Output layout

```
data/processed/punch/
├── labels.json                 # class names and ids
├── train.jsonl                 # one JSON record per annotated frame
├── val.jsonl
└── images/
    └── task_kam2_gh078416/
        └── 000653.jpg          # frame number, zero-padded to 6 digits
```

### What the script does

1. **Interpolate.** For each track, boxes are linearly interpolated between keyframes, so every frame from the first keyframe up to (not including) the `outside: true` keyframe gets a box.
2. **Extract frames.** Only frames that have at least one box are decoded from the MP4 and saved as JPEG (quality 95). Frames with no punch are not written.
3. **Resize.** Images are scaled so the longest side is `--max-size` (never upscaled). 1920×1080 becomes 640×360.
4. **Normalise boxes.** Pixel `(x1, y1, x2, y2)` becomes `(cx, cy, w, h)` in `[0, 1]`, relative to the source image size. Boxes are clipped to the image, and degenerate boxes are dropped. Because the values are relative, the resize does not change them.
5. **Map labels.** Each raw Polish label becomes the class id from the table above. An unknown label raises an error.
6. **Split by task.** Whole recording sessions go to either train or val, never both, so frames from one video cannot leak across splits. The split is seeded and reproducible.

### `train.jsonl` / `val.jsonl`

One JSON object per line, one line per annotated frame:

```json
{
  "image": "images/task_kam2_gh078416/000653.jpg",
  "width": 640,
  "height": 360,
  "class_labels": [3],
  "boxes": [[0.5, 0.45, 0.07, 0.07]]
}
```

| field          | meaning                                                                              |
|----------------|--------------------------------------------------------------------------------------|
| `image`        | path relative to the processed dataset folder                                        |
| `width`/`height` | size of the saved JPEG in pixels                                                   |
| `class_labels` | class id for each box (see the label table)                                          |
| `boxes`        | `[cx, cy, w, h]` per box, normalised to `[0, 1]`; same order as `class_labels`      |

`class_labels` and `boxes` have the same length. A frame can hold several boxes when punches overlap in time. This is the `{"class_labels", "boxes"}` target format that `RTDetrV2ForObjectDetection` takes as `labels`.

### `labels.json`

```json
{
  "names": ["Jab a la cabeza", "Cruzado a la cabeza", "..."],
  "id2label": {"0": "Jab a la cabeza", "1": "Cruzado a la cabeza", "...": "..."},
  "label2id": {"Jab a la cabeza": 0, "Cruzado a la cabeza": 1, "...": "..."}
}
```

`id2label` keys are strings because JSON requires it. `build_model` converts them to integers.

Pass `id2label` to `build_model(..., id2label=...)` so saved models carry the class names.

Labels are per-punch boxes, not per-clip, so they fit the detector. A video-level classifier would need a different target.

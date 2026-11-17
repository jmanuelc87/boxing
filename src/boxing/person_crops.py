"""Export one image crop per detected person, ready for labelling in Label Studio.

    python -m boxing.person_crops KEYPOINTS IMAGES DST [--pad 0.1] [--min-size 16]

KEYPOINTS is a directory of per-frame ``.json`` files from ``boxing.keypoints`` laid out
as ``<task>/<frame>.json`` (e.g. ``data/processed/punch-keypoints/images``). IMAGES holds
the matching frames as ``<task>/<frame>.jpg`` (e.g. ``data/processed/punch-seq/images``).

DST receives:

* ``<task>/<frame>_<person>.jpg``  one crop per person
* ``crops.jsonl``                  one record per crop: source frame, person index, box, score
* ``label_studio_tasks.json``      importable tasks for an image-classification project
* ``label_studio_tasks_http.json`` random sample of tasks whose image URLs point at an HTTP
                                   server serving DST (e.g. ``python -m http.server 8000``)

For Label Studio, serve DST as local files: set ``LOCAL_FILES_SERVING_ENABLED=true`` and
``LOCAL_FILES_DOCUMENT_ROOT`` to DST, or pass ``--ls-root`` if the document root is a
parent of DST. Task URLs are ``/data/local-files/?d=<crop path relative to the root>``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
from tqdm import tqdm

from boxing.color_roles import classify_person

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Label Studio config must use <Choices name="label" toName="image"> for these pre-labels to attach.
LS_FROM_NAME, LS_TO_NAME = "label", "image"


def find_image(images: Path, task: str, frame: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = images / task / f"{frame}{suffix}"
        if path.exists():
            return path
    return None


def padded_box(box: list[float], pad: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return (
        int(max(0, x1 - px)),
        int(max(0, y1 - py)),
        int(min(width, x2 + px)),
        int(min(height, y2 + py)),
    )


def export(
    keypoints: Path,
    images: Path,
    dst: Path,
    pad: float = 0.1,
    min_size: int = 16,
    ls_root: Path | None = None,
    prelabel: bool = True,
    http_base: str = "http://localhost:8000",
    sample: int = 200,
    seed: int | None = None,
) -> int:
    ls_root = ls_root or dst
    records: list[dict] = []
    tasks: list[dict] = []
    http_tasks: list[dict] = []
    skipped_frames = skipped_people = 0

    for json_path in tqdm(sorted(keypoints.glob("*/*.json")), desc="frames"):
        task, frame = json_path.parent.name, json_path.stem
        image_path = find_image(images, task, frame)
        image = cv2.imread(str(image_path)) if image_path else None
        if image is None:
            skipped_frames += 1
            continue
        height, width = image.shape[:2]

        people = json.loads(json_path.read_text(encoding="utf-8"))["people"]
        for index, person in enumerate(people):
            x1, y1, x2, y2 = padded_box(person["box"], pad, width, height)
            if min(x2 - x1, y2 - y1) < min_size:
                skipped_people += 1
                continue

            crop_path = dst / task / f"{frame}_{index:02d}.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), image[y1:y2, x1:x2])

            rel = crop_path.relative_to(dst).as_posix()
            records.append(
                {
                    "crop": rel,
                    "task": task,
                    "frame": frame,
                    "person": index,
                    "box": person["box"],
                    "score": person["score"],
                }
            )
            url = f"/data/local-files/?d={crop_path.relative_to(ls_root).as_posix()}"
            task_entry: dict = {"data": {"image": url, "task": task, "frame": frame, "person": index}}
            if prelabel:
                role = classify_person(image, person)
                records[-1]["prelabel"] = role
                task_entry["predictions"] = [
                    {
                        "model_version": "color-rules-v1",
                        "score": 0.5,
                        "result": [
                            {
                                "from_name": LS_FROM_NAME,
                                "to_name": LS_TO_NAME,
                                "type": "choices",
                                "value": {"choices": [role]},
                            }
                        ],
                    }
                ]
            tasks.append(task_entry)
            http_entry = {**task_entry, "data": {**task_entry["data"], "image": f"{http_base.rstrip('/')}/{rel}"}}
            http_tasks.append(http_entry)

    dst.mkdir(parents=True, exist_ok=True)
    with (dst / "crops.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    (dst / "label_studio_tasks.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")

    chosen = random.Random(seed).sample(http_tasks, min(sample, len(http_tasks)))
    (dst / "label_studio_tasks_http.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")

    print(f"{len(chosen)} random HTTP tasks -> {dst / 'label_studio_tasks_http.json'}")
    print(f"{len(records)} crops -> {dst} (skipped {skipped_frames} frames without an image, {skipped_people} tiny boxes)")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("keypoints", type=Path, help="directory of <task>/<frame>.json keypoint files")
    parser.add_argument("images", type=Path, help="directory of <task>/<frame>.jpg frames")
    parser.add_argument("dst", type=Path, help="output directory")
    parser.add_argument("--pad", type=float, default=0.1, help="fractional padding around each box")
    parser.add_argument("--min-size", type=int, default=16, help="skip crops with a side below this many pixels")
    parser.add_argument("--ls-root", type=Path, default=None, help="Label Studio local-files document root (default: DST)")
    parser.add_argument("--no-prelabel", action="store_true", help="do not attach colour-rule predictions to tasks")
    parser.add_argument("--http-base", default="http://localhost:8000", help="base URL serving DST, for label_studio_tasks_http.json")
    parser.add_argument("--sample", type=int, default=200, help="number of random crops in label_studio_tasks_http.json")
    parser.add_argument("--seed", type=int, default=None, help="random seed for the HTTP task sample")
    args = parser.parse_args()
    export(
        args.keypoints, args.images, args.dst, args.pad, args.min_size, args.ls_root,
        not args.no_prelabel, args.http_base, args.sample, args.seed,
    )


if __name__ == "__main__":
    main()

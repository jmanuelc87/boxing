"""Measure pose-estimation throughput (images/s) at several batch sizes on a small sample.

    python -m boxing.bench_keypoints SRC [--n 64] [--batch-sizes 1 4 8 16] [--backend yolo|rtdetr] [--weights W]

Images are decoded once up front, so the numbers cover preprocessing + inference + decoding only, not disk I/O.
Boxer filtering is off (``--boxers`` turns it on, with the default role model).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from boxing.keypoints import DEFAULT_ROLE_MODEL, IMAGE_SUFFIXES, build_estimator


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m boxing.bench_keypoints", description=__doc__.splitlines()[0])
    p.add_argument("src", type=Path, help="directory of images")
    p.add_argument("--n", type=int, default=64, help="sample size (default 64)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--backend", choices=["yolo", "rtdetr"], default="yolo")
    p.add_argument("--weights", default="yolo11m-pose.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--boxers", action="store_true", help="include the role-model boxer filter")
    args = p.parse_args(argv)

    paths = sorted(q for q in args.src.rglob("*") if q.suffix.lower() in IMAGE_SUFFIXES)[: args.n]
    if not paths:
        raise SystemExit(f"no images under {args.src}")
    images = [cv2.imread(str(q)) for q in paths]
    images = [im for im in images if im is not None]

    est = build_estimator(
        backend=args.backend,
        weights=args.weights,
        threshold=args.threshold,
        device=args.device,
        boxers_only=args.boxers,
        role_model_path=DEFAULT_ROLE_MODEL if args.boxers else None,
    )
    est.predict_batch(images[:2])  # warm-up: weight load, kernel compilation

    print(f"{len(images)} images, backend={args.backend}")
    print(f"{'batch':>6} {'seconds':>9} {'img/s':>8} {'speedup':>8}")
    base = None
    for b in args.batch_sizes:
        start = time.perf_counter()
        for i in range(0, len(images), b):
            est.predict_batch(images[i : i + b])
        dt = time.perf_counter() - start
        rate = len(images) / dt
        base = base or rate
        print(f"{b:>6} {dt:>9.2f} {rate:>8.1f} {rate / base:>7.2f}x")


if __name__ == "__main__":
    main()

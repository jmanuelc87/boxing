"""Heuristic role labels from clothing colour.

Samples the torso (shoulders to hips, falling back to the middle of the box) and labels
a person as one of:

* ``boxer_red`` / ``boxer_blue`` -- red or blue clothing dominates the torso
* ``referee``                    -- white shirt (the referee's black trousers are below the torso crop)
* ``other``                      -- anything else, including the audience

These are pre-labels for manual correction, not ground truth. The thresholds were tuned
by eye on a few frames from ``data/processed/punch-seq``.
"""

from __future__ import annotations

import cv2
import numpy as np

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
ROLES = ("boxer_red", "boxer_blue", "referee", "other")


def torso_roi(person: dict, height: int, width: int) -> tuple[int, int, int, int]:
    """Return ``(x1, y1, x2, y2)`` of the torso in pixels, clipped to the image."""
    kps = person["keypoints"]
    good = [
        (kps[i]["x"], kps[i]["y"])
        for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
        if kps[i]["score"] > 0.3
    ]
    if len(good) >= 3:
        xs, ys = zip(*good)
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        pad = 0.1 * (x2 - x1)
        x1, x2 = x1 + pad, x2 - pad
    else:
        bx1, by1, bx2, by2 = person["box"]
        bw, bh = bx2 - bx1, by2 - by1
        x1, x2 = bx1 + 0.25 * bw, bx2 - 0.25 * bw
        y1, y2 = by1 + 0.3 * bh, by1 + 0.7 * bh
    return int(max(0, x1)), int(max(0, y1)), int(min(width, x2)), int(min(height, y2))


def colour_fractions(bgr: np.ndarray) -> dict[str, float]:
    """Fraction of pixels in each colour range (OpenCV HSV: H in 0-180)."""
    if bgr.size == 0:
        return dict(red=0.0, blue=0.0, white=0.0, black=0.0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    n = h.size
    return dict(
        red=np.count_nonzero(((h <= 12) | (h >= 168)) & (s >= 80) & (v >= 60)) / n,
        blue=np.count_nonzero((h >= 95) & (h <= 130) & (s >= 60) & (v >= 25)) / n,
        white=np.count_nonzero((s <= 85) & (v >= 140)) / n,
        black=np.count_nonzero(v <= 60) / n,
    )


def classify(fractions: dict[str, float]) -> str:
    if fractions["red"] >= 0.25 and fractions["red"] >= fractions["blue"]:
        return "boxer_red"
    if fractions["blue"] >= 0.25:
        return "boxer_blue"
    if fractions["white"] >= 0.5 or (fractions["white"] >= 0.25 and fractions["black"] >= 0.15):
        return "referee"
    return "other"


def classify_person(image: np.ndarray, person: dict) -> str:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = torso_roi(person, height, width)
    return classify(colour_fractions(image[y1:y2, x1:x2]))

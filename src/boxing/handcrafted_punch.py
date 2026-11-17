"""Hand-crafted baseline for the punch classifier: engineered features + gradient boosting.

    uv run --with scikit-learn python -m boxing.handcrafted_punch
        [--data data/processed/punch-pairs] [--out runs/punch-baseline]

Uses the same normalised windows, the same fight-disjoint train/val split and the same metrics as
``boxing.train_punch_classifier``, so the two results are directly comparable.

Each window is reduced to per-stream statistics (max, min, mean, std, range, and the relative
position of the peak in time) over: wrist speed, wrist-to-own-shoulder extension, elbow angle,
wrist-to-opponent distances, closing speed toward the opponent, shoulder and hip rotation, torso
lean, and body movement for both boxers. Missing joints are NaN, which the booster handles natively.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from boxing.pair_dataset import (
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    VALID_CONF,
    PunchPairDataset,
    load_names,
)
from boxing.train_punch_classifier import class_metrics, confusion_matrix, print_report

L_ELBOW, R_ELBOW = 7, 8
STAT_NAMES = ("max", "min", "mean", "std", "range", "peak_pos")


def _stats(prefix: str, v: np.ndarray) -> dict[str, float]:
    """Summary statistics of a 1-D series; NaN where no frame is valid."""
    v = np.asarray(v, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return {f"{prefix}_{s}": np.nan for s in STAT_NAMES}
    vv = v[ok]
    pos = np.flatnonzero(ok)
    peak = int(pos[np.argmax(v[ok])])
    return {
        f"{prefix}_max": float(vv.max()),
        f"{prefix}_min": float(vv.min()),
        f"{prefix}_mean": float(vv.mean()),
        f"{prefix}_std": float(vv.std()),
        f"{prefix}_range": float(vv.max() - vv.min()),
        f"{prefix}_peak_pos": peak / max(1, len(v) - 1),
    }


def _speed(xy: np.ndarray) -> np.ndarray:
    """Per-frame displacement norm of a (T, 2) track, NaN where a frame is missing."""
    d = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    return np.concatenate([[np.nan], d])


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at b (degrees) between b->a and b->c, per frame."""
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _joints(w: np.ndarray) -> np.ndarray:
    """(T, 17, 2) positions with low-confidence joints set to NaN."""
    xy = w[..., :2].astype(np.float64).copy()
    xy[w[..., 2] <= VALID_CONF] = np.nan
    return xy


def _mid(xy: np.ndarray, i: int, j: int) -> np.ndarray:
    return (xy[:, i] + xy[:, j]) / 2


def window_features(w: np.ndarray) -> dict[str, float]:
    """Features for one normalised window ``(2, T, 17, 3)``; row 0 attacker, row 1 opponent."""
    a = _joints(w[0])
    o = _joints(w[1])
    f: dict[str, float] = {}

    # Presence and confidence.
    f["att_present"] = float(np.isfinite(a[:, L_HIP, 0]).mean())
    f["opp_present"] = float(np.isfinite(o[:, L_HIP, 0]).mean())
    f["att_mean_score"] = float(w[0][..., 2].mean())
    f["opp_mean_score"] = float(w[1][..., 2].mean())

    # Whole-body movement of each boxer.
    att_hip = _mid(a, L_HIP, R_HIP)
    opp_hip = _mid(o, L_HIP, R_HIP)
    f.update(_stats("att_hip_speed", _speed(att_hip)))
    f.update(_stats("opp_hip_speed", _speed(opp_hip)))
    f.update(_stats("opp_nose_speed", _speed(o[:, NOSE])))
    f.update(_stats("hip_gap", np.linalg.norm(att_hip - opp_hip, axis=-1)))

    # Torso orientation: shoulder line, hip line and lean.
    att_sh = _mid(a, L_SHOULDER, R_SHOULDER)
    att_sh_line = a[:, R_SHOULDER] - a[:, L_SHOULDER]
    att_hip_line = a[:, R_HIP] - a[:, L_HIP]
    f.update(_stats("att_shoulder_angle", np.degrees(np.arctan2(att_sh_line[:, 1], att_sh_line[:, 0]))))
    f.update(_stats("att_hip_angle", np.degrees(np.arctan2(att_hip_line[:, 1], att_hip_line[:, 0]))))
    lean = att_sh - att_hip
    f.update(_stats("att_lean", np.degrees(np.arctan2(lean[:, 0], -lean[:, 1]))))
    # Rotation between shoulders and hips, the twist that generates power.
    twist = np.degrees(np.arctan2(att_sh_line[:, 1], att_sh_line[:, 0]) - np.arctan2(att_hip_line[:, 1], att_hip_line[:, 0]))
    f.update(_stats("att_twist", twist))

    opp_nose = o[:, NOSE]
    opp_sh = _mid(o, L_SHOULDER, R_SHOULDER)
    opp_hips = opp_hip
    per_hand = {}
    for side, wrist_i, shoulder_i, elbow_i in (("L", L_WRIST, L_SHOULDER, L_ELBOW), ("R", R_WRIST, R_SHOULDER, R_ELBOW)):
        wr = a[:, wrist_i]
        sh = a[:, shoulder_i]
        el = a[:, elbow_i]
        speed = _speed(wr)
        extension = np.linalg.norm(wr - sh, axis=-1)
        elbow = _angle(sh, el, wr)
        to_head = np.linalg.norm(wr - opp_nose, axis=-1)
        to_shoulders = np.linalg.norm(wr - opp_sh, axis=-1)
        to_hips = np.linalg.norm(wr - opp_hips, axis=-1)
        closing = -np.diff(to_head, prepend=np.nan)  # positive when moving toward the opponent's head

        f.update(_stats(f"{side}_wrist_speed", speed))
        f.update(_stats(f"{side}_extension", extension))
        f.update(_stats(f"{side}_elbow", elbow))
        f.update(_stats(f"{side}_to_head", to_head))
        f.update(_stats(f"{side}_to_shoulders", to_shoulders))
        f.update(_stats(f"{side}_to_hips", to_hips))
        f.update(_stats(f"{side}_closing", closing))
        # Net displacement over the window, and how far the wrist moves relative to the body.
        ok = np.isfinite(wr[:, 0])
        net = float(np.linalg.norm(wr[ok][-1] - wr[ok][0])) if ok.sum() >= 2 else np.nan
        f[f"{side}_net_disp"] = net
        per_hand[side] = {"speed": speed, "extension": extension, "closing": closing}

    # Lead/rear cues: the hand that peaks first, moves more and extends more. Trees cannot
    # subtract features, so these differences are given explicitly.
    for key, label in (("speed", "wrist_speed"), ("extension", "extension"), ("closing", "closing")):
        lmax = _stats("", per_hand["L"][key])["_max"]
        rmax = _stats("", per_hand["R"][key])["_max"]
        f[f"LR_{label}_max_diff"] = lmax - rmax
    lpk = _stats("", per_hand["L"]["speed"])["_peak_pos"]
    rpk = _stats("", per_hand["R"]["speed"])["_peak_pos"]
    f["LR_speed_peak_diff"] = lpk - rpk

    # Attacker's guard: how close the opponent's wrists get to the attacker's head.
    for side, wrist_i in (("L", L_WRIST), ("R", R_WRIST)):
        f.update(_stats(f"opp_{side}_to_att_head", np.linalg.norm(o[:, wrist_i] - a[:, NOSE], axis=-1)))
    f.update(_stats("opp_wrist_speed", np.nanmax(np.stack([_speed(o[:, L_WRIST]), _speed(o[:, R_WRIST])]), axis=0)))
    return f


def build_features(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    rows = [window_features(w) for w in X]
    names = list(rows[0].keys())
    mat = np.array([[r[n] for n in names] for r in rows], dtype=np.float32)
    return mat, names


def majority_baseline(y_train: np.ndarray, y_val: np.ndarray, n: int) -> float:
    return float((y_val == np.bincount(y_train, minlength=n).argmax()).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/processed/punch-pairs"))
    parser.add_argument("--out", type=Path, default=Path("runs/punch-baseline"))
    parser.add_argument("--quality", choices=["all", "good"], default="all")
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices are expected
    names = load_names(args.data)
    n = len(names)

    train = PunchPairDataset(args.data, "train", args.quality)
    val = PunchPairDataset(args.data, "val", args.quality)
    X_tr, feat_names = build_features(train.X)
    X_va, _ = build_features(val.X)
    y_tr, y_va = train.y, val.y
    print(f"train {X_tr.shape}  val {X_va.shape}  features {len(feat_names)}")

    clf = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=args.seed,
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_va)

    cm = confusion_matrix(y_va, pred, n)
    m = class_metrics(cm)
    result = {"cm": cm, "accuracy": float((y_va == pred).mean()), "macro_f1": float(m["f1"].mean()), **m}
    majority = majority_baseline(y_tr, y_va, n)

    print(f"majority-class accuracy {majority:.3f}")
    print_report(result, names)

    # Which features the model leans on. Permutation importance on the validation set, macro F1 drop.
    from sklearn.inspection import permutation_importance

    imp = permutation_importance(clf, X_va, y_va, scoring="f1_macro", n_repeats=5, random_state=args.seed, n_jobs=1)
    order = np.argsort(imp.importances_mean)[::-1][:20]
    print("\ntop features (permutation importance, drop in macro F1):")
    for i in order:
        print(f"  {feat_names[i]:<32} {imp.importances_mean[i]:+.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "majority_accuracy": majority,
                "per_class": {
                    nm: {k: float(result[k][i]) for k in ("precision", "recall", "f1", "support")} for i, nm in enumerate(names)
                },
                "confusion_matrix": cm.tolist(),
                "n_features": len(feat_names),
                "top_features": [
                    {"name": feat_names[i], "importance": float(imp.importances_mean[i])} for i in order
                ],
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

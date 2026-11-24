"""Evaluation metrics. Label 1 = fake, score = predicted probability of fake.

* AUC (area under the ROC curve): the chance a random fake scores higher than a random real.
  Threshold-free, so it is the standard headline number in deepfake papers.
* AP (average precision): area under the precision/recall curve for the fake class.
* EER (equal error rate): the error at the threshold where false alarms equal misses.
* Accuracy at a fixed threshold (0.5 unless calibrated) and balanced accuracy (mean of the accuracy on
  reals and the accuracy on fakes, which is not inflated when one class dominates).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def binary_metrics(labels, scores, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    out: dict[str, float] = {"n": int(len(y)), "n_fake": int(y.sum()), "n_real": int((1 - y).sum())}
    pred = s >= threshold
    out["accuracy"] = float((pred == y).mean()) if len(y) else float("nan")
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")  # fakes caught
    tnr = float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan")  # reals passed
    out["fake_recall"], out["real_recall"] = tpr, tnr
    present = [v for v in (tpr, tnr) if not np.isnan(v)]
    out["balanced_accuracy"] = float(np.mean(present)) if present else float("nan")
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, s))
        out["ap"] = float(average_precision_score(y, s))
        fpr, tpr_curve, _ = roc_curve(y, s)
        fnr = 1 - tpr_curve
        i = int(np.nanargmin(np.abs(fnr - fpr)))
        out["eer"] = float((fpr[i] + fnr[i]) / 2)
        eps = 1e-7
        p = np.clip(s, eps, 1 - eps)
        out["log_loss"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return out


def video_scores(crops: pd.DataFrame) -> pd.DataFrame:
    """One score per video, by the same rule the app uses: take the highest-scoring face in each frame,
    then average those per-frame scores over the video's frames."""
    if "frame" in crops.columns:
        per_frame = crops.groupby(["dataset", "video_id", "frame"], sort=False).agg(
            label=("label", "first"), method=("method", "first"), score=("score", "max")).reset_index()
    else:
        per_frame = crops
    agg = per_frame.groupby(["dataset", "video_id"], sort=False).agg(
        label=("label", "first"), method=("method", "first"), score=("score", "mean"), n_frames=("score", "size"))
    return agg.reset_index()


def best_threshold(labels, scores) -> float:
    """Threshold that maximises balanced accuracy (chosen on validation data, then frozen)."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.argmax(tpr + (1 - fpr)))
    return float(np.clip(thr[i], 0.0, 1.0))

"""Score every validation and test crop, then report metrics per dataset and per manipulation method.

    python -m deepfake_detector.evaluate --data /kaggle/input --weights best.safetensors --out results/ours
    python -m deepfake_detector.evaluate --data /kaggle/input --weights gend --out results/gend   # baseline

Outputs (in --out):
    predictions.parquet   one row per crop: dataset, method, video, frame, label, score
    metrics.json          frame- and video-level metrics for every dataset and method
    threshold             the decision threshold picked on validation videos (max balanced accuracy)

The threshold is chosen on the validation split only and then applied unchanged to the test split,
so the reported test accuracy is not tuned on test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import CropDataset
from .metrics import best_threshold, binary_metrics, video_scores
from .model import DeepfakeClassifier, normalize
from .shards import find_shard_files, load_indexes


@torch.no_grad()
def score_index(model, index: pd.DataFrame, shard_paths, device, batch_size=256, workers=4) -> np.ndarray:
    loader = DataLoader(CropDataset(index, shard_paths), batch_size=batch_size, num_workers=workers,
                        pin_memory=device.type == "cuda")
    net = torch.nn.DataParallel(model) if device.type == "cuda" and torch.cuda.device_count() > 1 else model
    out = []
    for i, (images, _, _) in enumerate(loader):
        x = normalize(images.to(device, non_blocking=True))
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _ = net(x)
        out.append(logits.float().softmax(-1)[:, 1].cpu().numpy())
        if i % 100 == 0:
            print(f"  scored {min((i + 1) * batch_size, len(index)):,}/{len(index):,} crops", flush=True)
    return np.concatenate(out) if out else np.zeros(0)


def report(preds: pd.DataFrame, threshold: float) -> dict:
    """Metrics for every dataset (frame- and video-level) and every fake method (video-level vs. reals)."""
    out = {}
    for name, crops in preds.groupby("dataset"):
        vids = video_scores(crops)
        entry = {"frame": binary_metrics(crops.label, crops.score, threshold),
                 "video": binary_metrics(vids.label, vids.score, threshold),
                 "methods": {}}
        reals = vids[vids.label == 0]
        for method, fakes in vids[vids.label == 1].groupby("method"):
            both = pd.concat([reals, fakes])
            entry["methods"][method] = binary_metrics(both.label, both.score, threshold)
        out[name] = entry
    vids_all = video_scores(preds)
    out["ALL"] = {"video": binary_metrics(vids_all.label, vids_all.score, threshold)}
    out["mean_video_auc"] = float(np.mean([v["video"]["auc"] for k, v in out.items()
                                           if k not in ("ALL", "mean_video_auc") and "auc" in v["video"]]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--weights", required=True, help='trained .safetensors, or "gend" for the authors\' FF++ model')
    p.add_argument("--out", required=True)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--datasets", nargs="*", default=None, help="only these datasets (default: all found)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    index = load_indexes(args.data)
    index = index[index["split"].isin(args.splits)]
    if args.datasets:
        index = index[index["dataset"].isin(args.datasets)]
    index = index.reset_index(drop=True)
    print(f"{len(index):,} crops from {index.video_id.nunique():,} videos")
    print(index.groupby(["dataset", "split", "label"]).video_id.nunique().unstack(-1))

    model = DeepfakeClassifier()
    if args.weights == "gend":
        from huggingface_hub import hf_hub_download

        model.load_gend_checkpoint(hf_hub_download("yermandy/GenD_CLIP_L_14", "model.safetensors"))
    else:
        model.load_trainable(args.weights)
    model.eval().to(device)

    preds = index[["dataset", "method", "label", "video_id", "split", "frame", "face"]].copy()
    preds["score"] = score_index(model, index, find_shard_files(args.data), device, args.batch_size, args.workers)
    preds.to_parquet(out / "predictions.parquet", index=False)

    result = {"weights": str(args.weights)}
    threshold = 0.5
    if "val" in args.splits:
        val_vids = video_scores(preds[preds.split == "val"])
        threshold = best_threshold(val_vids.label, val_vids.score)
    result["threshold"] = threshold
    for split in args.splits:
        sub = preds[preds.split == split]
        result[split] = {"at_0.5": report(sub, 0.5), "at_val_threshold": report(sub, threshold)}
    (out / "metrics.json").write_text(json.dumps(result, indent=1, default=float))
    (out / "threshold").write_text(f"{threshold:.6f}\n")

    print(f"\nthreshold chosen on validation videos: {threshold:.4f}")
    for split in args.splits:
        r = result[split]["at_val_threshold"]
        print(f"\n== {split} (video level; accuracy at the validation threshold)")
        for name, v in r.items():
            if name in ("mean_video_auc",):
                continue
            m = v["video"]
            print(f"  {name:8s} AUC {m.get('auc', float('nan')):.4f}  acc {m['accuracy']:.4f}  "
                  f"bal.acc {m['balanced_accuracy']:.4f}  EER {m.get('eer', float('nan')):.4f}  "
                  f"({m['n_real']} real / {m['n_fake']} fake)")


if __name__ == "__main__":
    main()

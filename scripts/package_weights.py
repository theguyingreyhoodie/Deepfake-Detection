"""Copy trained weights into weights/ with their provenance and the calibrated decision threshold.

    python scripts/package_weights.py build/outputs/train-main/run/best.safetensors \
        build/outputs/train-main/eval/metrics.json weights/deepfake_detector.safetensors

The threshold is the one evaluate.py picked on validation videos (maximum balanced accuracy); the app
and the command line read it from the file's metadata.
"""

import json
import sys

from safetensors import safe_open
from safetensors.torch import save_file

src, metrics_path, dst = sys.argv[1:4]
metrics = json.load(open(metrics_path))
with safe_open(src, "pt") as f:
    tensors = {k: f.get_tensor(k) for k in f.keys()}
    meta = dict(f.metadata() or {})
meta["threshold"] = f"{metrics['threshold']:.6f}"
meta["val_mean_video_auc"] = f"{metrics['val']['at_0.5']['mean_video_auc']:.6f}"
meta["test_mean_video_auc"] = f"{metrics['test']['at_0.5']['mean_video_auc']:.6f}"
save_file(tensors, dst, metadata=meta)
print(f"wrote {dst} ({len(tensors)} tensors) with metadata {meta}")

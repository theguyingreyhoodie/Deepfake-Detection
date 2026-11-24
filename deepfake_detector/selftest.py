"""Write a small synthetic crop set (random images) to test the training pipeline end to end on new
hardware: multi-GPU, mixed precision, memory, throughput, checkpointing. Not for real training."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from .shards import ShardWriter


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--videos", type=int, default=200)
    p.add_argument("--frames", type=int, default=8)
    a = p.parse_args(argv)
    rng = np.random.default_rng(0)
    w = ShardWriter(a.out, "synthetic")
    for v in range(a.videos):
        label = v % 2
        split = "val" if v % 10 == 0 else "train"
        dataset = "SYN-A" if v % 3 else "SYN-B"
        for f in range(a.frames):
            img = cv2.GaussianBlur(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8), (5, 5), 0)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            w.add(buf.tobytes(), dataset=dataset, method="real" if label == 0 else "fake", label=label,
                  video_id=f"v{v}", split=split, group=f"v{v}", frame=f, face=0)
    print(f"wrote {len(w.close()):,} synthetic crops to {a.out}")


if __name__ == "__main__":
    main()

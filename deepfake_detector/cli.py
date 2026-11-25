"""Command line: ``deepfake-detect photo.jpg clip.mp4 [--frames 32] [--json]``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .inference import DEFAULT_WEIGHTS, DeepfakeDetector


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="deepfake-detect", description="Detect face-manipulation deepfakes in images and videos.")
    p.add_argument("paths", nargs="+", help="image or video files")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help='trained weights, or "gend" for the GenD baseline')
    p.add_argument("--frames", type=int, default=32, help="frames sampled per video")
    p.add_argument("--threshold", type=float, default=None, help="override the stored decision threshold")
    p.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = p.parse_args(argv)

    detector = DeepfakeDetector(args.weights, device=args.device, threshold=args.threshold)
    results = []
    for path in args.paths:
        if not Path(path).exists():
            print(f"{path}: file not found", file=sys.stderr)
            continue
        r = detector.analyze(path, n_frames=args.frames) if not path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")) else detector.analyze(path)
        results.append({"path": path, "verdict": r.verdict, "fake_probability": r.fake_probability,
                        "threshold": r.threshold, "frames_analyzed": r.frames_analyzed,
                        "faces": [{"frame": f.frame_index, "box": f.box, "fake_probability": round(f.fake_probability, 4)}
                                  for f in r.faces]})
        if not args.json:
            prob = "n/a" if r.fake_probability is None else f"{r.fake_probability:.3f}"
            faces, frames = len(r.faces), r.frames_analyzed
            print(f"{path}: {r.verdict}  (fake probability {prob}, threshold {r.threshold:.3f}, "
                  f"{faces} face crop{'s' * (faces != 1)} from {frames} frame{'s' * (frames != 1)})")
    if args.json:
        print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

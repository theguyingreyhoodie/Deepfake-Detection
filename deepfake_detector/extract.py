"""Face-crop extraction: videos -> aligned 224 x 224 face crops, packed into shards.

For each video: read evenly spaced frames, detect faces, choose which faces to keep, align each kept
face (see ``faces.py``), JPEG-encode at quality 95 and append it to a shard. Work is spread over all
CPU cores with one face detector per worker process.

Which faces are kept (``choose_faces``):
* faces smaller than 40 px or with detector confidence below 0.6 are ignored;
* in a REAL video every prominent face is genuine, so up to two faces are kept per frame;
* in a FAKE video we do not know which face was swapped when two people are visible, so a frame is
  kept only when one face clearly dominates (the second is under a quarter of its area). This avoids
  labelling an untouched bystander's face as "fake".

Example (Kaggle)::

    python -m deepfake_detector.extract --dataset FF++ --root /kaggle/input --out /kaggle/working
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

import cv2
import numpy as np
import pandas as pd

from . import sources
from .faces import SCRFD, Face, align_face, scrfd_model_path
from .shards import ShardWriter
from .video import read_frames

ADAPTERS = {"FF++": sources.faceforensics, "CDFv2": sources.celeb_df, "DFDC": sources.dfdc, "DFD": sources.dfd}
META_COLUMNS = ["dataset", "method", "label", "video_id", "split", "group"]

_detector: SCRFD | None = None
_decode_threads = 0


def choose_faces(faces: list[Face], label: int, min_size: float = 40, min_score: float = 0.6,
                 dominance: float = 0.25) -> list[Face]:
    faces = [f for f in faces if f.score >= min_score and min(f.box[2] - f.box[0], f.box[3] - f.box[1]) >= min_size]
    if len(faces) <= 1:
        return faces
    faces = sorted(faces, key=lambda f: f.area, reverse=True)
    if label == 1:
        return [faces[0]] if faces[1].area < dominance * faces[0].area else []
    return [f for f in faces[:2] if f.area >= dominance * faces[0].area]


def _init_worker(model_path: str, det_size: int, threads: int, decode_threads: int) -> None:
    global _detector, _decode_threads
    cv2.setNumThreads(1)
    _detector = SCRFD(model_path, det_size=det_size, threads=threads)
    _decode_threads = decode_threads


def process_job(job: dict) -> tuple[dict, list[tuple[dict, bytes]], str | None]:
    crops: list[tuple[dict, bytes]] = []
    try:
        t0 = time.time()
        frames = read_frames(job["path"], int(job["n_frames"]), threads=_decode_threads)
        t_decode = time.time() - t0
        for idx, frame in frames:
            faces = _detector.detect(frame)
            for k, face in enumerate(choose_faces(faces, int(job["label"]))):
                crop = align_face(frame, face.landmarks, out_size=224)
                ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    crops.append(({"frame": idx, "face": k, "det_score": round(face.score, 4),
                                   "face_px": int(math.sqrt(face.area)), "n_faces": len(faces)}, buf.tobytes()))
        job = {**job, "frames_read": len(frames), "t_decode": t_decode, "t_total": time.time() - t0}
        return job, crops, None
    except Exception:  # one unreadable video must not stop a multi-hour job
        return {**job, "frames_read": 0, "t_decode": 0.0, "t_total": 0.0}, [], traceback.format_exc(limit=2)


def build_jobs(dataset: str, root: str, train_fake: int, train_real: int, eval_frames: int,
               shard: int = 0, num_shards: int = 1) -> pd.DataFrame:
    jobs = ADAPTERS[dataset](root)
    jobs["n_frames"] = sources.frames_per_video(jobs, train_fake, train_real, eval_frames)
    jobs = jobs.sort_values("video_id").reset_index(drop=True)
    if num_shards > 1:
        jobs = jobs[jobs.index % num_shards == shard].reset_index(drop=True)
    return jobs


def run(jobs: pd.DataFrame, out_dir: str, prefix: str, workers: int, det_size: int, threads: int,
        time_budget_hours: float = 11.0, decode_threads: int = 1) -> pd.DataFrame:
    writer = ShardWriter(out_dir, prefix)
    videos = []
    t0 = time.time()
    n_crops = 0
    records = jobs.to_dict("records")
    # fork is fast and safe on Linux; macOS frameworks are not fork-safe, so use spawn there.
    context = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
    model_path = str(scrfd_model_path())  # download once, before any worker starts
    with context.Pool(workers, initializer=_init_worker, initargs=(model_path, det_size, threads, decode_threads)) as pool:
        for i, (job, crops, err) in enumerate(pool.imap_unordered(process_job, records, chunksize=2), 1):
            for meta, data in crops:
                writer.add(data, **{k: job[k] for k in META_COLUMNS}, **meta)
            n_crops += len(crops)
            videos.append({**{k: job[k] for k in META_COLUMNS + ["path", "n_frames", "frames_read", "t_decode", "t_total"]},
                           "crops": len(crops), "error": err})
            if err:
                print(f"ERROR {job['path']}: {err.strip().splitlines()[-1]}", flush=True)
            if i in (1, 10) or i % 100 == 0 or i == len(records):
                el = time.time() - t0
                dec = np.mean([v["t_decode"] for v in videos[-100:]])
                tot = np.mean([v["t_total"] for v in videos[-100:]])
                print(f"[{el / 60:6.1f} min] {i}/{len(records)} videos, {n_crops:,} crops, "
                      f"{i / el:.2f} videos/s, ETA {(len(records) - i) / (i / el) / 60:.0f} min "
                      f"(per video per worker: decode {dec:.1f}s, total {tot:.1f}s)", flush=True)
            if (time.time() - t0) / 3600 > time_budget_hours:
                print("time budget reached: stopping early; unprocessed videos are listed as missing", flush=True)
                pool.terminate()
                break
    index = writer.close(f"{prefix}_index.parquet")
    vids = pd.DataFrame(videos)
    done = set(vids["video_id"]) if len(vids) else set()
    missing = jobs[~jobs["video_id"].isin(done)]
    vids.to_parquet(os.path.join(out_dir, f"{prefix}_videos.parquet"), index=False)
    missing.to_parquet(os.path.join(out_dir, f"{prefix}_missing.parquet"), index=False)
    print(f"done: {len(vids)} videos, {len(index):,} crops, {len(missing)} not processed, "
          f"{(vids['error'].notna()).sum() if len(vids) else 0} errors", flush=True)
    if len(index):
        print(index.groupby(["split", "method"]).agg(crops=("video_id", "size"), videos=("video_id", "nunique")))
    return index


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prefix", default=None)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--threads", type=int, default=1, help="ONNX Runtime threads per worker")
    p.add_argument("--det-size", type=int, default=640, help="longer side the frame is shrunk to for detection")
    p.add_argument("--decode-threads", type=int, default=1, help="decoder threads per worker (0 = FFmpeg default)")
    p.add_argument("--train-fake", type=int, default=32, help="frames per fake training video")
    p.add_argument("--train-real", type=int, default=64, help="frames per real training video")
    p.add_argument("--eval-frames", type=int, default=32, help="frames per val/test video")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N videos")
    p.add_argument("--time-budget-hours", type=float, default=11.0)
    args = p.parse_args(argv)
    prefix = args.prefix or f"{args.dataset.lower().replace('+', 'p')}{args.shard}"
    jobs = build_jobs(args.dataset, args.root, args.train_fake, args.train_real, args.eval_frames,
                      args.shard, args.num_shards)
    if args.limit:
        jobs = jobs.sample(n=min(args.limit, len(jobs)), random_state=0).reset_index(drop=True)
    print(f"{args.dataset}: {len(jobs)} videos to process, {int(jobs.n_frames.sum()):,} frames requested")
    print(jobs.groupby(["split", "method"]).size().unstack(0).fillna(0).astype(int))
    run(jobs, args.out, prefix, args.workers, args.det_size, args.threads, args.time_budget_hours,
        args.decode_threads)


if __name__ == "__main__":
    main()

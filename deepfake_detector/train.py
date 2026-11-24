"""Training loop (single GPU, multi-GPU via ``torchrun``, or Apple-silicon MPS for smoke tests).

Usage on a machine with two GPUs (e.g. Kaggle's T4 x2)::

    torchrun --nproc_per_node 2 -m deepfake_detector.train --data /kaggle/input --out /kaggle/working

What happens:
1. Load the crop index (Parquet) from every ``--data`` folder; training rows come from split=="train".
2. Each step draws a balanced batch (see ``data.py``), augments it, and computes
   loss = cross-entropy + 0.1 * alignment + 0.5 * uniformity (the GenD loss weights).
3. Only LayerNorm parameters and the linear head are updated (Adam, warm-up then cosine decay).
4. After every "epoch" (a fixed number of samples) the model is scored on held-out validation
   videos; the checkpoint with the best mean validation AUC is kept as ``best.safetensors``.
5. A full resumable checkpoint (``last.pt``) is written regularly so a 12-hour session limit only
   pauses training: ``--resume path/to/last.pt`` continues exactly where it stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .augment import train_transform
from .data import CropDataset, WeightedEpochSampler, balanced_weights, subsample_videos
from .losses import alignment_loss, uniformity_loss
from .metrics import binary_metrics, video_scores
from .model import DeepfakeClassifier, normalize
from .shards import find_shard_files, load_indexes


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, help="folders containing *_index.parquet and *.bin shards")
    p.add_argument("--out", required=True)
    p.add_argument("--backbone", default="openai/clip-vit-large-patch14")
    p.add_argument("--datasets", nargs="*", default=None, help="train only on these datasets (default: all found)")
    p.add_argument("--exclude-datasets", nargs="*", default=[], help="datasets left out of training (held out)")
    p.add_argument("--exclude-methods", nargs="*", default=[], help="fake methods left out of training")
    p.add_argument("--dataset-weights", default="{}", help='JSON, e.g. {"DFDC": 2.0}; default 1 each')
    p.add_argument("--batch-size", type=int, default=128, help="global batch size")
    p.add_argument("--micro-batch", type=int, default=32, help="per-GPU forward size (gradient accumulation fills the rest)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--samples-per-epoch", type=int, default=250_000)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--align-weight", type=float, default=0.1)
    p.add_argument("--uniform-weight", type=float, default=0.5)
    p.add_argument("--val-videos", type=int, default=100, help="validation videos per (dataset, method)")
    p.add_argument("--val-crops", type=int, default=8, help="validation crops per video")
    p.add_argument("--workers", type=int, default=3, help="DataLoader workers per GPU process")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--time-limit-hours", type=float, default=11.3, help="save and stop before this wall time")
    p.add_argument("--max-steps", type=int, default=0, help="debug: stop after this many optimizer steps")
    return p.parse_args(argv)


def setup_distributed():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from datetime import timedelta

        # Long timeout: rank 0 validates alone at the end of each epoch while the others wait.
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return rank, world, torch.device("cuda", local)
    if torch.cuda.is_available():
        return 0, 1, torch.device("cuda", 0)
    if torch.backends.mps.is_available():
        return 0, 1, torch.device("mps")
    return 0, 1, torch.device("cpu")


def lr_at(step: int, total: int, warmup: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return min_lr + (lr - min_lr) * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def predict(model, loader, device, amp_dtype) -> np.ndarray:
    model.eval()
    scores = []
    for images, _, _ in loader:
        x = normalize(images.to(device, non_blocking=True))
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, _ = model(x)
        scores.append(logits.float().softmax(-1)[:, 1].cpu().numpy())
    model.train()
    return np.concatenate(scores) if scores else np.zeros(0)


def validate(model, val_index, shard_paths, device, amp_dtype, workers) -> dict:
    loader = DataLoader(CropDataset(val_index, shard_paths), batch_size=128, num_workers=workers,
                        pin_memory=device.type == "cuda")
    frames = val_index[["dataset", "video_id", "method", "label", "frame"]].copy()
    frames["score"] = predict(model, loader, device, amp_dtype)
    report = {}
    for name, group in frames.groupby("dataset"):
        vids = video_scores(group)
        report[name] = {"frame_auc": binary_metrics(group.label, group.score).get("auc", float("nan")),
                        "video_auc": binary_metrics(vids.label, vids.score).get("auc", float("nan"))}
    report["mean_video_auc"] = float(np.nanmean([v["video_auc"] for v in report.values()]))
    return report


def main(argv=None):
    args = parse_args(argv)
    rank, world, device = setup_distributed()
    is_main = rank == 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    def log(msg: str):
        if is_main:
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            print(line, flush=True)
            with open(out / "train_log.txt", "a") as f:
                f.write(line + "\n")

    # ---------------- data ----------------
    index = load_indexes(args.data)
    if args.datasets:
        index = index[index["dataset"].isin(args.datasets)]
    index = index[~index["dataset"].isin(args.exclude_datasets) & ~index["method"].isin(args.exclude_methods)]
    index = index.reset_index(drop=True)
    shard_paths = find_shard_files(args.data)
    train_index = index[index["split"] == "train"].reset_index(drop=True)
    val_index = subsample_videos(index[index["split"] == "val"], args.val_videos, args.val_crops, seed=args.seed)
    weights = balanced_weights(train_index, json.loads(args.dataset_weights))
    log(f"train crops {len(train_index):,} from {train_index.video_id.nunique():,} videos; "
        f"val crops {len(val_index):,}; world size {world}; device {device}")
    for (d, lab), g in train_index.groupby(["dataset", "label"]):
        log(f"  {d:10s} {'fake' if lab else 'real'}: {len(g):8,} crops, {g.video_id.nunique():6,} videos, "
            f"sampling share {weights[g.index].sum():.3f}")

    per_rank = args.samples_per_epoch // world
    sampler = WeightedEpochSampler(weights, per_rank, seed=args.seed, rank=rank)
    loader = DataLoader(CropDataset(train_index, shard_paths, train_transform()), batch_size=args.micro_batch,
                        sampler=sampler, num_workers=args.workers, pin_memory=device.type == "cuda",
                        drop_last=True, persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)

    # ---------------- model ----------------
    model = DeepfakeClassifier(args.backbone).to(device)
    amp_dtype = torch.float16 if device.type == "cuda" else None
    if device.type == "cuda":  # frozen weights in fp16 (half the memory); trainable ones stay fp32
        for name, p in model.named_parameters():
            if not p.requires_grad:
                p.data = p.data.half()
    if args.grad_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable parameters: {sum(p.numel() for p in params):,} of {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    ddp_model = model
    if world > 1:
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    accum = max(1, args.batch_size // (args.micro_batch * world))
    steps_per_epoch = per_rank // (args.micro_batch * accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)
    start_epoch, step, best = 0, 0, -1.0
    history = []
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model._load_canonical(ckpt["trainable"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, step, best, history = ckpt["epoch"], ckpt["step"], ckpt["best"], ckpt["history"]
        log(f"resumed from {args.resume}: epoch {start_epoch}, step {step}, best {best:.4f}")
    log(f"{steps_per_epoch} steps/epoch x {args.epochs} epochs, accumulation {accum}, "
        f"global batch {args.micro_batch * accum * world}")

    def save_last(epoch_done: int):
        if is_main:
            torch.save({"trainable": model.trainable_state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": epoch_done, "step": step, "best": best,
                        "history": history, "args": vars(args)}, out / "last.pt.tmp")
            os.replace(out / "last.pt.tmp", out / "last.pt")

    def should_stop() -> bool:
        """Every rank must agree, or one GPU would wait forever for the other: rank 0 decides."""
        hours = (time.time() - t_start) / 3600
        flag = torch.tensor([float(hours > args.time_limit_hours or (args.max_steps and step >= args.max_steps))],
                            device=device)
        if world > 1:
            dist.broadcast(flag, src=0)
        return bool(flag.item())

    t_start = time.time()
    stop = False
    for epoch in range(start_epoch, args.epochs):
        done_in_epoch = max(0, step - epoch * steps_per_epoch)  # > 0 only when resuming mid-epoch
        sampler.set_epoch(epoch, start=done_in_epoch * args.micro_batch * accum)
        if done_in_epoch:
            log(f"resuming epoch {epoch} after {done_in_epoch} of its {steps_per_epoch} steps")
        model.train()
        sums = {"loss": 0.0, "ce": 0.0, "align": 0.0, "unif": 0.0, "acc": 0.0}
        n_micro, t_epoch = 0, time.time()
        t_wait, t_mark = 0.0, time.time()
        optimizer.zero_grad(set_to_none=True)
        for i, (images, labels, _) in enumerate(loader):
            t_wait += time.time() - t_mark  # time spent waiting for the DataLoader
            x = normalize(images.to(device, non_blocking=True))
            y = labels.to(device, non_blocking=True)
            sync = (i + 1) % accum == 0
            ctx = ddp_model.no_sync() if (world > 1 and not sync) else _nullcontext()
            with ctx:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits, z = ddp_model(x)
                ce = torch.nn.functional.cross_entropy(logits.float(), y)
                al = alignment_loss(z, y)
                un = uniformity_loss(z)
                loss = ce + args.align_weight * al + args.uniform_weight * un
                scaler.scale(loss / accum).backward()
            sums["loss"] += loss.item(); sums["ce"] += ce.item(); sums["align"] += al.item(); sums["unif"] += un.item()
            sums["acc"] += (logits.argmax(-1) == y).float().mean().item()
            n_micro += 1
            t_mark = time.time()
            if not sync:
                continue
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup_steps, args.lr, args.min_lr)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                rate = n_micro * args.micro_batch * world / (time.time() - t_epoch)
                log(f"epoch {epoch} step {step}/{total_steps} lr {optimizer.param_groups[0]['lr']:.2e} "
                    + " ".join(f"{k} {v / n_micro:.4f}" for k, v in sums.items())
                    + f" | {rate:.0f} img/s, waiting for data {100 * t_wait / (time.time() - t_epoch):.0f}% of the time")
            if step % 500 == 0:
                save_last(epoch)
            if step % 25 == 0 and should_stop():
                stop = True
                break
        if stop and not (args.max_steps and step >= args.max_steps):
            save_last(epoch)  # --resume continues this epoch from the next unseen batch
            log(f"time limit reached at step {step}; saved last.pt for --resume")
            break

        # ---------------- end of epoch: validate ----------------
        if world > 1:
            dist.barrier()
        if is_main:
            report = validate(model, val_index, shard_paths, device, amp_dtype, args.workers)
            report.update(epoch=epoch, step=step, train={k: v / max(1, n_micro) for k, v in sums.items()})
            history.append(report)
            log("validation: " + json.dumps({k: v for k, v in report.items() if k != "train"}, default=float))
            if device.type == "cuda":
                log(f"peak GPU memory {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB")
            if report["mean_video_auc"] > best:
                best = report["mean_video_auc"]
                model.save_trainable(out / "best.safetensors", epoch=epoch, step=step, mean_val_video_auc=best)
                log(f"new best mean validation video AUC {best:.4f}")
            model.save_trainable(out / "final.safetensors", epoch=epoch, step=step)
            json.dump(history, open(out / "history.json", "w"), indent=1, default=float)
        save_last(epoch + 1)
        if world > 1:
            dist.barrier()
        if stop:
            break

    if world > 1:
        dist.destroy_process_group()
    log("done")


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()

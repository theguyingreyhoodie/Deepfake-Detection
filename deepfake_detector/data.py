"""PyTorch dataset over packed face crops, plus the balanced sampling used for training.

The training pool mixes datasets of very different sizes and real/fake ratios (DFDC alone has five
fakes per real video). Training on it as-is would teach the model "most things are fake from DFDC".
Instead every training batch is drawn with replacement so that:

* each dataset contributes a chosen share of samples (``dataset_weights``),
* within a dataset, real and fake crops are 50 / 50,
* within the fakes, every manipulation method gets an equal share.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.io import decode_jpeg

from .shards import ShardReader


class CropDataset(Dataset):
    def __init__(self, index: pd.DataFrame, shard_paths: dict[str, str], transform=None):
        # Shard names as small integer codes: a numpy array of a million Python strings would be
        # slowly copied into every DataLoader worker (reference counting touches each object).
        codes, names = pd.factorize(index["shard"])
        self.shard_codes = codes.astype(np.int32)
        self.shard_names = list(names)
        self.offsets = index["offset"].to_numpy(np.int64)
        self.lengths = index["length"].to_numpy(np.int64)
        self.labels = index["label"].to_numpy(np.int64)
        self.reader = ShardReader(shard_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        data = self.reader.read(self.shard_names[self.shard_codes[i]], self.offsets[i], self.lengths[i])
        img = decode_jpeg(torch.frombuffer(bytearray(data), dtype=torch.uint8))  # (3, H, W) RGB uint8
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), int(i)


def balanced_weights(index: pd.DataFrame, dataset_weights: dict[str, float] | None = None) -> np.ndarray:
    """Per-row sampling probabilities implementing the three balancing rules in the module docstring."""
    weights = np.zeros(len(index), dtype=np.float64)
    datasets = sorted(index["dataset"].unique())
    shares = {d: (dataset_weights or {}).get(d, 1.0) for d in datasets}
    total = sum(shares.values())
    for d in datasets:
        in_d = index["dataset"] == d
        share = shares[d] / total
        real = in_d & (index["label"] == 0)
        fake = in_d & (index["label"] == 1)
        if real.any() and fake.any():
            real_share, fake_share = share / 2, share / 2
        else:  # a dataset with only one class still gets its full share
            real_share, fake_share = (share, 0.0) if real.any() else (0.0, share)
        if real.any():
            weights[real.to_numpy()] = real_share / real.sum()
        methods = index.loc[fake, "method"].unique()
        for m in methods:
            rows = (fake & (index["method"] == m)).to_numpy()
            weights[rows] = fake_share / len(methods) / rows.sum()
    return weights / weights.sum()


class WeightedEpochSampler(Sampler[int]):
    """Draws ``samples_per_epoch`` indices with replacement; each DDP rank gets a different stream."""

    def __init__(self, weights: np.ndarray, samples_per_epoch: int, seed: int = 0, rank: int = 0):
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.samples_per_epoch = samples_per_epoch
        self.seed, self.rank, self.epoch = seed, rank, 0
        self.start = 0  # samples of this epoch already consumed (used when resuming mid-epoch)

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = epoch, start

    def __len__(self) -> int:
        return self.samples_per_epoch - self.start

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch * 1_009 + self.rank)
        # The full epoch's stream is regenerated identically, then the consumed prefix is skipped.
        draws = torch.multinomial(self.weights, self.samples_per_epoch, replacement=True, generator=g)
        return iter(draws[self.start:].tolist())


def subsample_videos(index: pd.DataFrame, max_videos_per_class: int, max_crops_per_video: int = 0,
                     seed: int = 0) -> pd.DataFrame:
    """A random subset of videos per (dataset, method), optionally thinned to a few crops per video.

    Used for the quick validation pass after each epoch; final evaluation uses every test crop.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for (_, _), group in index.groupby(["dataset", "method"]):
        vids = group["video_id"].unique()
        if len(vids) > max_videos_per_class:
            vids = rng.choice(vids, max_videos_per_class, replace=False)
        keep.append(group[group["video_id"].isin(vids)])
    out = pd.concat(keep, ignore_index=True)
    if max_crops_per_video:
        out = (out.sample(frac=1.0, random_state=seed).groupby(["dataset", "video_id"], sort=False)
               .head(max_crops_per_video).sort_index().reset_index(drop=True))
    return out

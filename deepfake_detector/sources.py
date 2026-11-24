"""Dataset adapters: turn each dataset's folder layout into a table of labelled video jobs.

Every adapter returns a DataFrame with one row per video:

    path      absolute path of the video file
    dataset   short dataset name (FF++, CDFv2, DFDC, DFD, ...)
    method    "real" or the manipulation method that produced the fake
    label     0 = real, 1 = fake
    video_id  unique within the dataset
    split     train / val / test (official splits where the dataset defines them)
    group     videos sharing a group must stay in the same split (same source footage or person)

Splits are decided here, once, before any training, so test videos can never leak into training.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd

SPLITS_DIR = Path(__file__).parent / "splits"
FFPP_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]


def _stable_fraction(key: str) -> float:
    """Deterministic pseudo-random number in [0, 1) from a string (same answer on every machine)."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 2**32


def find_dir(root: str | Path, required: list[str], max_depth: int = 6) -> Path | None:
    """First folder under ``root`` that contains all ``required`` entries (files or folders)."""
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = Path(dirpath).relative_to(root).parts
        if len(depth) > max_depth:
            dirnames[:] = []
            continue
        names = set(dirnames) | set(filenames)
        if all(r in names for r in required):
            return Path(dirpath)
    return None


def _videos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in (".mp4", ".avi", ".mov"))


# ---------------------------------------------------------------------------------------------
# FaceForensics++ (c23): 1000 YouTube videos and 5 manipulation methods, official 720/140/140 split.
# ---------------------------------------------------------------------------------------------
def faceforensics(root: str | Path) -> pd.DataFrame:
    base = find_dir(root, ["original", "Deepfakes", "Face2Face"])
    if base is None:
        raise FileNotFoundError(f"FaceForensics++ not found under {root}")
    split_of: dict[str, str] = {}
    pair_split: dict[frozenset, str] = {}
    for split in ("train", "val", "test"):
        for a, b in json.load(open(SPLITS_DIR / f"ff_{split}.json")):
            split_of[a] = split_of[b] = split
            pair_split[frozenset((a, b))] = split
    rows = []
    for p in _videos(base / "original"):
        vid = p.stem
        rows.append(dict(path=str(p), dataset="FF++", method="real", label=0, video_id=f"real/{vid}",
                         split=split_of[vid], group=vid))
    for method in FFPP_METHODS:
        if not (base / method).is_dir():
            continue
        for p in _videos(base / method):
            target, source = p.stem.split("_")[:2]
            rows.append(dict(path=str(p), dataset="FF++", method=method, label=1, video_id=f"{method}/{p.stem}",
                             split=pair_split[frozenset((target, source))], group=target))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Celeb-DF v2: celebrity interviews, 5639 face swaps; the official test list has 518 videos.
# Non-test videos are split train/val by person, so validation faces are unseen during training.
# ---------------------------------------------------------------------------------------------
def celeb_df(root: str | Path, val_fraction: float = 0.1) -> pd.DataFrame:
    base = find_dir(root, ["Celeb-real", "Celeb-synthesis", "YouTube-real"])
    if base is None:
        raise FileNotFoundError(f"Celeb-DF v2 not found under {root}")
    test_list = base / "List_of_testing_videos.txt"
    test = set()
    if test_list.exists():
        for line in test_list.read_text().split("\n"):
            parts = line.split()
            if len(parts) == 2:
                test.add(parts[1].strip())
    rows = []
    for folder, method, label in (("Celeb-real", "real", 0), ("YouTube-real", "real", 0), ("Celeb-synthesis", "faceswap", 1)):
        for p in _videos(base / folder):
            rel = f"{folder}/{p.name}"
            m = re.match(r"id(\d+)_", p.stem)
            person = f"id{m.group(1)}" if m else f"yt{p.stem}"
            if rel in test:
                split = "test"
            else:
                split = "val" if _stable_fraction("cdf:" + person) < val_fraction else "train"
            rows.append(dict(path=str(p), dataset="CDFv2", method=method, label=label,
                             video_id=f"{folder}/{p.stem}", split=split, group=person))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# DFDC (Facebook Deepfake Detection Challenge): paid actors, many face-swap methods.
# Labels come from each part's metadata.json. Whole parts are held out for val/test, so the
# held-out videos come from different recording batches than the training videos.
# ---------------------------------------------------------------------------------------------
def dfdc(root: str | Path, val_parts=(8,), test_parts=(9,)) -> pd.DataFrame:
    rows = []
    for dirpath, _, filenames in os.walk(root):
        if "metadata.json" not in filenames:
            continue
        m = re.search(r"dfdc_train_part_(\d+)", dirpath)
        if not m:
            continue
        part = int(m.group(1))
        split = "test" if part in test_parts else "val" if part in val_parts else "train"
        meta = json.load(open(os.path.join(dirpath, "metadata.json")))
        for name, info in meta.items():
            path = os.path.join(dirpath, name)
            if not os.path.exists(path):
                continue
            fake = info.get("label") == "FAKE"
            original = info.get("original") or name
            rows.append(dict(path=path, dataset="DFDC", method="dfdc_fake" if fake else "real", label=int(fake),
                             video_id=f"part{part}/{Path(name).stem}", split=split,
                             group=f"part{part}/{Path(original).stem}"))
    if not rows:
        raise FileNotFoundError(f"No DFDC metadata.json under {root}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# DFD (Google/Jigsaw DeepFakeDetection): 28 consenting actors in staged scenes.
# Split by actor: a fake belongs to the split of the actor whose video was manipulated.
# ---------------------------------------------------------------------------------------------
DFD_TEST_ACTORS = {"02", "07", "12", "17", "22", "27"}
DFD_VAL_ACTORS = {"05", "15", "25"}


def dfd(root: str | Path) -> pd.DataFrame:
    rows = []
    seen = set()
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if not name.lower().endswith(".mp4"):
                continue
            stem = Path(name).stem
            m_fake = re.match(r"(\d\d)_(\d\d)__(.+)__([A-Z0-9]+)$", stem)
            m_real = re.match(r"(\d\d)__(.+)$", stem)
            if m_fake:
                actor, label, method = m_fake.group(1), 1, "dfd_fake"
            elif m_real:
                actor, label, method = m_real.group(1), 0, "real"
            else:
                continue
            if stem in seen:  # the same file can appear in two mirrors
                continue
            seen.add(stem)
            split = "test" if actor in DFD_TEST_ACTORS else "val" if actor in DFD_VAL_ACTORS else "train"
            rows.append(dict(path=os.path.join(dirpath, name), dataset="DFD", method=method, label=label,
                             video_id=stem, split=split, group=f"actor{actor}"))
    if not rows:
        raise FileNotFoundError(f"No DFD videos under {root}")
    return pd.DataFrame(rows)


def frames_per_video(jobs: pd.DataFrame, train_fake: int, train_real: int, eval_frames: int = 32) -> pd.Series:
    """How many frames to sample: more for real training videos (reals are the scarcer class),
    and a fixed 32 for every validation/test video (the standard evaluation protocol)."""
    n = pd.Series(eval_frames, index=jobs.index)
    train = jobs["split"] == "train"
    n[train & (jobs["label"] == 1)] = train_fake
    n[train & (jobs["label"] == 0)] = train_real
    return n

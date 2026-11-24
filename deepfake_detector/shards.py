"""Packed storage for face crops.

A training set here is around a million 224 x 224 JPEG face crops. Kept as a million small files they
are slow to copy and exceed Kaggle's limits on output file counts, so they are appended back to back
into a few large ``.bin`` shard files instead. A Parquet index stores, for every crop, which shard it
is in, its byte offset and length, and its labels (dataset, method, real/fake, video, split, ...).
Reading one crop is a single ``pread`` call: no unpacking, no per-file overhead.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


class ShardWriter:
    """Appends encoded crops to ``<prefix>_NNN.bin`` files, starting a new file every ``max_bytes``."""

    def __init__(self, out_dir: str | Path, prefix: str, max_bytes: int = 2_000_000_000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.rows: list[dict] = []
        self._index = -1
        self._file = None
        self._size = 0
        self._open_next()

    def _open_next(self):
        if self._file:
            self._file.close()
        self._index += 1
        self._name = f"{self.prefix}_{self._index:03d}.bin"
        self._file = open(self.out_dir / self._name, "wb")
        self._size = 0

    def add(self, data: bytes, **meta) -> None:
        if self._size + len(data) > self.max_bytes and self._size > 0:
            self._open_next()
        self._file.write(data)
        self.rows.append({"shard": self._name, "offset": self._size, "length": len(data), **meta})
        self._size += len(data)

    def close(self, index_name: str | None = None) -> pd.DataFrame:
        if self._file:
            self._file.close()
            self._file = None
        if self._size == 0:  # drop a trailing empty shard
            (self.out_dir / self._name).unlink(missing_ok=True)
        index = pd.DataFrame(self.rows)
        index.to_parquet(self.out_dir / (index_name or f"{self.prefix}_index.parquet"), index=False)
        return index


def find_shard_files(roots: list[str | Path]) -> dict[str, str]:
    """Map shard file name -> full path, searching every root folder recursively."""
    found: dict[str, str] = {}
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for name in files:
                if name.endswith(".bin"):
                    found.setdefault(name, os.path.join(dirpath, name))
    return found


def load_indexes(roots: list[str | Path]) -> pd.DataFrame:
    """Concatenate every ``*_index.parquet`` found under the given roots."""
    frames = []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for name in sorted(files):
                if name.endswith("_index.parquet"):
                    frames.append(pd.read_parquet(os.path.join(dirpath, name)))
    if not frames:
        raise FileNotFoundError(f"No *_index.parquet files under {roots}")
    return pd.concat(frames, ignore_index=True)


class ShardReader:
    """Random access to crops. File handles are opened lazily, per process (safe with DataLoader workers)."""

    def __init__(self, shard_paths: dict[str, str]):
        self.shard_paths = shard_paths
        self._fds: dict[str, int] = {}
        self._pid = os.getpid()

    def read(self, shard: str, offset: int, length: int) -> bytes:
        if os.getpid() != self._pid:  # forked into a worker: never share file descriptors
            self._fds, self._pid = {}, os.getpid()
        fd = self._fds.get(shard)
        if fd is None:
            fd = self._fds[shard] = os.open(self.shard_paths[shard], os.O_RDONLY)
        return os.pread(fd, int(length), int(offset))

    def __getstate__(self):  # pickled into workers without open descriptors
        return {"shard_paths": self.shard_paths}

    def __setstate__(self, state):
        self.__init__(state["shard_paths"])

"""Reading a fixed number of evenly spaced frames from a video.

A deepfake video is judged from several frames spread over its whole length, not from one frame:
the per-frame fake probabilities are averaged. These helpers pick which frames to read and read them.
"""

from __future__ import annotations

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".3gp"}


def evenly_spaced(total: int, count: int) -> list[int]:
    """``count`` frame indices spread evenly over ``total`` frames (fewer if the video is short)."""
    if total <= 0:
        return []
    if count >= total:
        return list(range(total))
    # Centre of each of `count` equal segments, so the first and last frames (often fades) are avoided.
    return sorted({int((i + 0.5) * total / count) for i in range(count)})


def frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    return max(n, 0)


def open_video(path: str, threads: int = 0) -> cv2.VideoCapture:
    """Open a video; ``threads`` > 0 caps the decoder's threads (useful when many workers decode at once)."""
    if threads > 0:
        try:
            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, threads])
            if cap.isOpened():
                return cap
        except (cv2.error, AttributeError, TypeError):
            pass
    return cv2.VideoCapture(path)


def read_frames(path: str, count: int, seek_threshold: int = 900, threads: int = 0) -> list[tuple[int, np.ndarray]]:
    """Return up to ``count`` evenly spaced ``(frame_index, BGR image)`` pairs.

    Short clips (all dataset videos) are decoded sequentially, which is exact. Long videos (uploads)
    jump straight to each wanted frame, which is much faster than decoding everything in between.
    """
    cap = open_video(path, threads)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[tuple[int, np.ndarray]] = []
    try:
        if total <= 0:  # container does not report a length: decode everything, then subsample
            every = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                every.append(frame)
            return [(i, every[i]) for i in evenly_spaced(len(every), count)]

        wanted = evenly_spaced(total, count)
        if total > seek_threshold:
            for idx in wanted:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok:
                    frames.append((idx, frame))
            return frames

        wanted_set, last = set(wanted), wanted[-1] if wanted else -1
        idx = 0
        while idx <= last:
            if not cap.grab():  # grab() decodes but skips colour conversion: cheap for unwanted frames
                break
            if idx in wanted_set:
                ok, frame = cap.retrieve()
                if ok:
                    frames.append((idx, frame))
            idx += 1
        if not frames and total > 0:  # some files report a bogus length; fall back to seeking
            cap.release()
            cap = open_video(path, threads)
            for idx in wanted:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok:
                    frames.append((idx, frame))
        return frames
    finally:
        cap.release()

"""Face detection and alignment.

Every image or video frame goes through the same two steps before the classifier sees it:

1. **Detect.** SCRFD-10G (InsightFace's ``det_10g.onnx``) finds each face and five landmarks:
   left eye, right eye, nose tip, left mouth corner, right mouth corner.
2. **Align.** A similarity transform (rotation + uniform scale + shift, no shearing) maps the five
   landmarks onto a fixed template. The result is an upright, square crop with the eyes at the same
   height in every image and a 1.3x margin of context around the face, resized to 224 x 224.

Training crops and inference crops are produced by exactly this code, so the classifier never sees
a crop style at test time that it did not see during training. The template and the 1.3 margin follow
the GenD paper (Yermakov et al., WACV 2026), whose training recipe this project builds on.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Official InsightFace "buffalo_l" detector. The checksum pins the exact file, whichever mirror serves it.
SCRFD_SHA256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
SCRFD_URLS = [
    "https://huggingface.co/datasets/theanhntp/Liblib/resolve/ae4357741af379482690fe3e0f2fa6fd32ba33b4"
    "/insightface/models/buffalo_l/det_10g.onnx",
]
SCRFD_ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

# Where the five landmarks should land in a unit square (x, y); the same numbers GenD uses.
LANDMARK_TEMPLATE = np.array(
    [[0.34, 0.46], [0.66, 0.46], [0.50, 0.64], [0.37, 0.82], [0.63, 0.82]], dtype=np.float32
)


def cache_dir() -> Path:
    """Folder for downloaded model files (override with DEEPFAKE_CACHE)."""
    path = Path(os.environ.get("DEEPFAKE_CACHE", Path.home() / ".cache" / "deepfake_detector"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scrfd_model_path() -> Path:
    """Return a verified local copy of det_10g.onnx, downloading it on first use.

    Safe when several processes call it at once: each downloads to its own temporary file and the
    finished file is moved into place atomically.
    """
    target = cache_dir() / "det_10g.onnx"
    if target.exists() and _sha256(target) == SCRFD_SHA256:
        return target
    tmp = target.with_suffix(f".part{os.getpid()}")
    for url in SCRFD_URLS:
        try:
            urllib.request.urlretrieve(url, tmp)
            if _sha256(tmp) == SCRFD_SHA256:
                os.replace(tmp, target)
                return target
        except OSError:
            pass
    # Fallback: the official release zip (about 280 MB) contains the same file.
    import zipfile

    zip_path = cache_dir() / f"buffalo_l.{os.getpid()}.zip"
    urllib.request.urlretrieve(SCRFD_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z, open(tmp, "wb") as dst:
        dst.write(z.read("det_10g.onnx"))
    zip_path.unlink(missing_ok=True)
    if _sha256(tmp) != SCRFD_SHA256:
        raise RuntimeError("Downloaded face detector does not match the pinned checksum")
    os.replace(tmp, target)
    return target


@dataclass
class Face:
    """One detected face, in the pixel coordinates of the image it came from."""

    box: np.ndarray  # (4,) x1, y1, x2, y2
    landmarks: np.ndarray  # (5, 2) eyes, nose, mouth corners
    score: float  # detector confidence in [0, 1]

    @property
    def area(self) -> float:
        w = max(0.0, float(self.box[2] - self.box[0]))
        h = max(0.0, float(self.box[3] - self.box[1]))
        return w * h


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression: keep the best box, drop boxes overlapping it, repeat."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]
    return keep


class SCRFD:
    """Minimal ONNX Runtime wrapper around InsightFace's SCRFD-10G face detector.

    The network predicts, at three scales (strides 8, 16 and 32 pixels), a face score, a box and five
    landmarks for two anchors per grid cell. We decode those, drop low scores and remove duplicates.
    """

    STRIDES = (8, 16, 32)
    ANCHORS_PER_CELL = 2

    def __init__(
        self,
        model_path: str | Path | None = None,
        det_size: int = 640,
        threshold: float = 0.5,
        nms_threshold: float = 0.4,
        providers: list[str] | None = None,
        threads: int | None = None,
    ):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path or scrfd_model_path()),
            sess_options=options,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.det_size = det_size
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self._centers: dict[tuple[int, int, int], np.ndarray] = {}

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        if key not in self._centers:
            grid = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centers = (grid * stride).reshape(-1, 2)
            self._centers[key] = np.repeat(centers, self.ANCHORS_PER_CELL, axis=0)
        return self._centers[key]

    def detect(self, image_bgr: np.ndarray) -> list[Face]:
        """Find faces in a BGR image (OpenCV channel order). Returns faces sorted largest first."""
        h, w = image_bgr.shape[:2]
        # Shrink so the longer side is det_size, then pad up to a multiple of 32 (the coarsest stride).
        scale = self.det_size / max(h, w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        pad_w, pad_h = -(-new_w // 32) * 32, -(-new_h // 32) * 32
        canvas = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (pad_w, pad_h), (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        all_scores, all_boxes, all_kps = [], [], []
        n = len(self.STRIDES)
        for i, stride in enumerate(self.STRIDES):
            scores = outputs[i].reshape(-1)
            keep = scores >= self.threshold
            if not keep.any():
                continue
            centers = self._anchor_centers(pad_h // stride, pad_w // stride, stride)[keep]
            dist = outputs[i + n].reshape(-1, 4)[keep] * stride
            kps = outputs[i + 2 * n].reshape(-1, 10)[keep] * stride
            boxes = np.concatenate([centers - dist[:, :2], centers + dist[:, 2:]], axis=1)
            all_scores.append(scores[keep])
            all_boxes.append(boxes)
            all_kps.append(kps.reshape(-1, 5, 2) + centers[:, None, :])
        if not all_scores:
            return []

        scores = np.concatenate(all_scores)
        boxes = np.concatenate(all_boxes) / scale
        kps = np.concatenate(all_kps) / scale
        keep = _nms(boxes, scores, self.nms_threshold)
        faces = [Face(box=boxes[k], landmarks=kps[k], score=float(scores[k])) for k in keep]
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces


def align_face(image_bgr: np.ndarray, landmarks: np.ndarray, out_size: int | None = 224, margin: float = 1.3):
    """Warp a face so its landmarks sit on the template, with ``margin`` x context, then resize.

    The warp is done at the face's own resolution first (so no detail is thrown away by the rotation),
    and only then resized to ``out_size`` with an anti-aliasing filter (``None`` keeps the native size).
    """
    src = np.asarray(landmarks, dtype=np.float32)
    # Native size: how many pixels the template spans in this image, times the margin.
    tri = np.triu_indices(5, k=1)
    src_d = np.linalg.norm(src[:, None] - src[None], axis=-1)[tri]
    dst_d = np.linalg.norm(LANDMARK_TEMPLATE[:, None] - LANDMARK_TEMPLATE[None], axis=-1)[tri]
    native = int(np.clip(np.round(np.mean(src_d / dst_d) * margin), 32, 4096))

    # Template shrunk by the margin and centred, in pixels of a native x native canvas.
    dst = (LANDMARK_TEMPLATE + (margin - 1) / 2) / margin * native
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:  # degenerate landmarks: fall back to an axis-aligned square around them
        x1, y1 = src.min(0)
        x2, y2 = src.max(0)
        # In the template the landmarks span about 0.36 of the face square's side.
        side = max(float(max(x2 - x1, y2 - y1)) / 0.36 * margin, 1.0)
        s = native / side
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        matrix = np.array([[s, 0, native / 2 - s * cx], [0, s, native / 2 - s * cy]], np.float32)
    crop = cv2.warpAffine(image_bgr, matrix, (native, native), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    if out_size and native != out_size:
        interp = cv2.INTER_AREA if native > out_size else cv2.INTER_CUBIC
        crop = cv2.resize(crop, (out_size, out_size), interpolation=interp)
    return crop

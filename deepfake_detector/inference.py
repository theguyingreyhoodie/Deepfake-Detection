"""Run the trained detector on images and videos.

    from deepfake_detector.inference import DeepfakeDetector
    detector = DeepfakeDetector()                  # loads weights/deepfake_detector.safetensors
    result = detector.analyze("clip.mp4")          # or a photo
    print(result.verdict, result.fake_probability)

How a verdict is reached:
* image: every face found gets a fake probability; the image's score is the highest one
  (one manipulated face is enough to make the image a deepfake).
* video: ``n_frames`` evenly spaced frames are analysed the same way; the video's score is the average
  over frames of each frame's highest face score. Averaging over many frames smooths out the odd
  misleading frame (motion blur, a hand over the face) - the same rule used in the evaluation.
* The score is compared with a decision threshold chosen on validation videos and stored with the
  weights (0.5 when none is stored).

Each crop is JPEG-encoded at quality 95 and decoded again before scoring, exactly as the training and
test crops were stored, so the model sees the same kind of input it was evaluated on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from .faces import SCRFD, Face, align_face
from .model import DeepfakeClassifier, normalize
from .video import IMAGE_EXTENSIONS, read_frames

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "deepfake_detector.safetensors"


@dataclass
class FaceResult:
    frame_index: int
    box: list[float]
    detector_score: float
    fake_probability: float
    crop: np.ndarray | None = field(default=None, repr=False)  # RGB 224 x 224, for display


@dataclass
class Result:
    kind: str  # "image" or "video"
    fake_probability: float | None  # None when no face was found
    threshold: float
    faces: list[FaceResult]
    frames_analyzed: int
    frame_scores: list[tuple[int, float]]  # (frame index, highest face score in that frame)

    @property
    def verdict(self) -> str:
        if self.fake_probability is None:
            return "NO FACE FOUND"
        return "FAKE" if self.fake_probability >= self.threshold else "REAL"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DeepfakeDetector:
    def __init__(self, weights: str | Path | None = DEFAULT_WEIGHTS, device: str | None = None,
                 threshold: float | None = None, det_size: int = 640, min_face: int = 40):
        self.device = torch.device(device) if device else pick_device()
        self.model = DeepfakeClassifier()
        stored_threshold = None
        if weights == "gend":  # the GenD authors' released FF++ model, for comparison
            from huggingface_hub import hf_hub_download

            self.model.load_gend_checkpoint(hf_hub_download("yermandy/GenD_CLIP_L_14", "model.safetensors"))
        elif weights is not None:
            from safetensors import safe_open

            self.model.load_trainable(weights)
            with safe_open(str(weights), "pt") as f:
                stored_threshold = (f.metadata() or {}).get("threshold")
        self.model.eval().to(self.device)
        self.threshold = threshold if threshold is not None else float(stored_threshold or 0.5)
        self.detector = SCRFD(det_size=det_size)
        self.min_face = min_face

    @torch.no_grad()
    def score_crops(self, crops_bgr: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        tensors = []
        for crop in crops_bgr:
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            crop = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            tensors.append(torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
        scores = []
        for i in range(0, len(tensors), batch_size):
            x = normalize(torch.stack(tensors[i:i + batch_size]).to(self.device))
            scores.append(self.model.fake_probability(x).float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)

    def _faces(self, frame_bgr: np.ndarray, max_faces: int) -> list[Face]:
        """Faces in a frame. If none is found at the normal detection size, retry smaller: a photo that is
        already a tight face crop would otherwise be enlarged until the face is too big for the detector."""
        default = self.detector.det_size
        try:
            for size in (default, 320, 160):
                if size > default:
                    continue
                self.detector.det_size = size
                faces = [f for f in self.detector.detect(frame_bgr)
                         if f.score >= 0.6 and min(f.box[2] - f.box[0], f.box[3] - f.box[1]) >= self.min_face]
                if faces:
                    return faces[:max_faces]
            return []
        finally:
            self.detector.det_size = default

    def analyze_frames(self, frames: list[tuple[int, np.ndarray]], kind: str, max_faces: int = 4,
                       keep_crops: bool = False) -> Result:
        found: list[tuple[int, Face, np.ndarray]] = []
        for idx, frame in frames:
            for face in self._faces(frame, max_faces):
                found.append((idx, face, align_face(frame, face.landmarks, out_size=224)))
        scores = self.score_crops([c for _, _, c in found])
        faces = [FaceResult(idx, [round(float(v), 1) for v in f.box], round(f.score, 3), float(s),
                            cv2.cvtColor(c, cv2.COLOR_BGR2RGB) if keep_crops else None)
                 for (idx, f, c), s in zip(found, scores)]
        per_frame: dict[int, float] = {}
        for fr in faces:
            per_frame[fr.frame_index] = max(per_frame.get(fr.frame_index, 0.0), fr.fake_probability)
        frame_scores = sorted(per_frame.items())
        prob = float(np.mean([s for _, s in frame_scores])) if frame_scores else None
        return Result(kind, prob, self.threshold, faces, len(frames), frame_scores)

    def analyze_image(self, image: str | Path | np.ndarray, keep_crops: bool = False) -> Result:
        frame = cv2.imread(str(image), cv2.IMREAD_COLOR) if not isinstance(image, np.ndarray) else image
        if frame is None:
            raise ValueError(f"Could not read image {image}")
        return self.analyze_frames([(0, frame)], "image", keep_crops=keep_crops)

    def analyze_video(self, path: str | Path, n_frames: int = 32, keep_crops: bool = False) -> Result:
        frames = read_frames(str(path), n_frames)
        if not frames:
            raise ValueError(f"Could not read any frames from {path}")
        return self.analyze_frames(frames, "video", max_faces=2, keep_crops=keep_crops)

    def analyze(self, path: str | Path, **kwargs) -> Result:
        if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
            return self.analyze_image(path, keep_crops=kwargs.get("keep_crops", False))
        return self.analyze_video(path, **kwargs)

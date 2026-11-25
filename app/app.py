"""Web demo: upload a photo or a video, get a verdict, per-face scores and heatmaps.

    python app/app.py                 # uses weights/deepfake_detector.safetensors
    python app/app.py --weights gend  # the GenD authors' FF++-only model, for comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepfake_detector.explain import gradcam, overlay  # noqa: E402
from deepfake_detector.inference import DEFAULT_WEIGHTS, DeepfakeDetector, Result  # noqa: E402
from deepfake_detector.model import normalize  # noqa: E402
from deepfake_detector.video import read_frames  # noqa: E402

DETECTOR: DeepfakeDetector | None = None


def verdict_markdown(r: Result) -> str:
    if r.fake_probability is None:
        return "## No face found\nThe detector judges faces; try a clearer, front-facing shot."
    p = r.fake_probability
    label = "FAKE" if r.verdict == "FAKE" else "REAL"
    color = "#c62828" if label == "FAKE" else "#2e7d32"
    n, f = len(r.faces), r.frames_analyzed
    what = "image" if r.kind == "image" else f"video ({f} frame{'s' * (f != 1)} sampled)"
    return (f"<h2 style='color:{color};margin:0'>{label}</h2>"
            f"<p>Fake probability <b>{p:.1%}</b> (decision threshold {r.threshold:.1%}) for this {what}; "
            f"{n} face crop{'s' * (n != 1)} analysed.</p>")


def face_pairs(r: Result, limit: int) -> list:
    """Most suspicious faces first, each as (face, its crop beside its Grad-CAM heatmap)."""
    faces = sorted(r.faces, key=lambda f: f.fake_probability, reverse=True)[:limit]
    if not faces:
        return []
    crops = np.stack([f.crop for f in faces])
    x = normalize(torch.from_numpy(crops).permute(0, 3, 1, 2).to(DETECTOR.device))
    heat = gradcam(DETECTOR.model, x)
    return [(f, np.concatenate([f.crop, overlay(f.crop, h)], axis=1)) for f, h in zip(faces, heat)]


def face_gallery(r: Result, limit: int = 8) -> list:
    """Video tab: crop and heatmap pairs, captioned with their frame and score."""
    return [(side, f"frame {f.frame_index}: fake {f.fake_probability:.1%}") for f, side in face_pairs(r, limit)]


def face_sheet(r: Result, limit: int = 4) -> np.ndarray | None:
    """Image tab: the crop and heatmap pairs stacked into one picture, each under a caption. (Gradio's
    gallery gives a lone item a full-width square cell, which pushes a single face out of view.)"""
    rows = []
    for k, (f, side) in enumerate(face_pairs(r, limit), 1):
        bar = np.full((30, side.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"face {k}: fake {f.fake_probability:.1%}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (40, 40, 40), 1, cv2.LINE_AA)
        rows.append(np.concatenate([bar, side], axis=0))
    return np.concatenate(rows, axis=0) if rows else None


def annotate(frame_bgr: np.ndarray, r: Result, frame_index: int) -> np.ndarray:
    img = frame_bgr.copy()
    thick = max(2, img.shape[1] // 400)
    for f in r.faces:
        if f.frame_index != frame_index:
            continue
        fake = f.fake_probability >= r.threshold
        color = (40, 40, 220) if fake else (60, 170, 60)
        x1, y1, x2, y2 = map(int, f.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        cv2.putText(img, f"{'FAKE' if fake else 'REAL'} {f.fake_probability:.0%}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, img.shape[1] / 1400), color, thick)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def analyze_image(path):
    if not path:
        return "Upload an image first.", None, None
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        return "Could not read that image.", None, None
    r = DETECTOR.analyze_image(frame, keep_crops=True)
    return verdict_markdown(r), annotate(frame, r, 0), face_sheet(r)


def analyze_video(path, n_frames):
    if not path:
        return "Upload a video first.", None, None, []
    frames = read_frames(path, int(n_frames))
    if not frames:
        return "Could not read that video.", None, None, []
    r = DETECTOR.analyze_frames(frames, "video", max_faces=2, keep_crops=True)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    if r.frame_scores:
        idx, scores = zip(*r.frame_scores)
        ax.plot(idx, scores, marker="o", lw=1.5, color="#5b5bd6")
    ax.axhline(r.threshold, color="#c62828", ls="--", lw=1, label="decision threshold")
    ax.set_ylim(-0.03, 1.03)  # keep points at 0 and 1 off the frame
    ax.set_xlabel("frame")
    ax.set_ylabel("fake probability")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), frameon=False, fontsize=8)  # above the plot, off the data
    fig.tight_layout()
    # show the frame whose most suspicious face scored highest
    key = max(r.frame_scores, key=lambda t: t[1])[0] if r.frame_scores else frames[0][0]
    key_frame = dict(frames)[key]
    return verdict_markdown(r), fig, annotate(key_frame, r, key), face_gallery(r)


def model_card() -> str:
    metrics = ROOT / "results" / "summary.json"
    try:  # show the weights file relative to the repository, not the full local path
        weights = Path(DETECTOR_WEIGHTS).resolve().relative_to(ROOT)
    except ValueError:
        weights = DETECTOR_WEIGHTS if DETECTOR_WEIGHTS == "gend" else Path(DETECTOR_WEIGHTS).name
    text = f"Weights: `{weights}` · device: `{DETECTOR.device}` · threshold {DETECTOR.threshold:.3f}"
    if metrics.exists():
        s = json.loads(metrics.read_text())
        rows = " · ".join(f"{k} {v:.1%}" for k, v in s.get("test_video_auc", {}).items())
        text += f"<br>Test video-level AUC: {rows}"
    return text


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Deepfake Detector") as demo:
        gr.Markdown("# Deepfake Detector\nDetects **face manipulation** (face swaps and facial reenactment) "
                    "in photos and videos. It analyses the faces it finds; it is not built to judge fully "
                    "AI-generated scenes. Heatmaps show which regions drove the *fake* score.")
        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    v_in = gr.Video(label="Video")
                    v_frames = gr.Slider(8, 64, value=32, step=8, label="Frames to sample")
                    v_btn = gr.Button("Analyse video", variant="primary")
                with gr.Column():
                    v_out = gr.HTML()
                    v_plot = gr.Plot(label="Fake probability per frame")
            with gr.Row():
                v_key = gr.Image(label="Most suspicious frame", height=360)
                v_gal = gr.Gallery(label="Face crops (left) and heatmaps (right), most suspicious first",
                                   columns=2, height=360, object_fit="contain")
            v_btn.click(analyze_video, [v_in, v_frames], [v_out, v_plot, v_key, v_gal])
        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    i_in = gr.Image(type="filepath", label="Image")
                    i_btn = gr.Button("Analyse image", variant="primary")
                with gr.Column():
                    i_out = gr.HTML()
                    i_img = gr.Image(label="Detected faces", height=360)
            i_gal = gr.Image(label="Face crops (left) and heatmaps (right), most suspicious first", height=360)
            i_btn.click(analyze_image, [i_in], [i_out, i_img, i_gal])
        gr.Markdown(model_card())
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    a = ap.parse_args()
    DETECTOR_WEIGHTS = a.weights
    DETECTOR = DeepfakeDetector(a.weights)
    build_ui().launch(server_port=a.port, share=a.share)

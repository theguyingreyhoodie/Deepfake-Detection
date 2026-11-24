"""Deepfake (face-manipulation) detection for images and videos.

Pipeline: detect and align faces (faces.py) -> score each face crop with a CLIP ViT-L/14 whose
LayerNorms were fine-tuned to spot manipulation (model.py) -> average over frames for videos
(inference.py).
"""

__version__ = "0.1.0"

# Deepfake Detection

Detects **face-manipulation deepfakes** in videos and images: face swaps (one person's face pasted
onto another's body) and facial reenactment (someone's expressions or mouth movements driven by
another person). Give it a video or a photo; it finds every face, scores each one, and returns a
verdict with a fake probability, per-frame scores and heatmaps of the regions that looked
manipulated.

It is built on the GenD recipe (Yermakov et al., WACV 2026), which reports the best average
cross-dataset AUC among recent detectors over 14 deepfake benchmarks: a frozen CLIP ViT-L/14 image
encoder of which only the LayerNorm layers are fine-tuned. The difference here is scale: GenD was trained on one dataset
(FaceForensics++); this model is trained on four public deepfake datasets together.

**Scope.** "Deepfake" here means a real person's face that was manipulated. Fully AI-generated
images or videos (text-to-image, text-to-video, GAN faces of people who do not exist) are a
different problem and are not what this model was trained to recognise.

<!-- RESULTS:START -->
*Results are added here by `scripts/make_results.py` once the evaluation has run.*
<!-- RESULTS:END -->

## How it works

A longer, step-by-step explanation (with the reasoning behind each design choice) is in
[docs/EXPLAINER.md](docs/EXPLAINER.md).

```
video ──► 32 evenly spaced frames ──► SCRFD face detector ──► 5 landmarks per face
                                                                    │
     verdict ◄── average over frames ◄── highest face score   ◄── align: upright 224×224 crop,
     (vs. calibrated threshold)            in each frame            1.3× margin of context
                                                ▲
                                   CLIP ViT-L/14 (frozen) with
                                   fine-tuned LayerNorms + linear head
```

1. **Faces.** [SCRFD-10G](https://github.com/deepinsight/insightface) finds faces and five landmarks
   (eyes, nose tip, mouth corners). A similarity transform rotates and scales each face onto a fixed
   template, so every crop is upright with the eyes in the same place, plus a 1.3× margin so the face
   boundary (where swaps blend in) is visible. Training and inference use the same code.
2. **Classifier.** OpenAI's CLIP ViT-L/14 image encoder, pre-trained on 400 million image–text pairs,
   is kept frozen. Only its LayerNorm scale/shift parameters (102,400 numbers, 0.03% of the network)
   and a 2-way linear layer on the L2-normalised 1024-d CLS feature are trained. Loss:
   cross-entropy + 0.1 × alignment + 0.5 × uniformity (the last two keep the features well spread on
   the unit sphere). Changing so few parameters re-uses CLIP's knowledge instead of memorising one
   dataset's quirks, which is why this recipe transfers to unseen manipulation methods.
3. **Training mix.** Every batch is balanced: each dataset gets an equal share, real and fake are
   50/50 within a dataset, and each manipulation method gets an equal share of the fakes. Crops are
   augmented (flip, small rotation/shift/zoom, blur, colour jitter, re-scaling, noise and JPEG
   re-compression at quality 40–100) so the model does not rely on any one sharpness or compression level.
4. **Videos.** 32 frames are sampled across the clip; in each frame the most suspicious face counts;
   the video score is the mean over frames. The decision threshold is chosen once on validation
   videos (maximum balanced accuracy) and stored with the weights.

## Training data

| Dataset | Real videos | Fake videos | Manipulations | Split used |
|---|---:|---:|---|---|
| [FaceForensics++](https://github.com/ondyari/FaceForensics) (c23) | 1,000 | 5,000 | Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter | official 720 / 140 / 140 |
| [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) | 890 | 5,639 | improved face swap | official 518-video test list; rest split by person |
| [DFD](https://research.google/blog/contributing-data-to-deepfake-detection-research/) (Google/Jigsaw) | 363 | 3,068 | face swap | by actor (19 train / 3 val / 6 test) |
| [DFDC](https://ai.meta.com/datasets/dfdc/) (parts 0–9) | 2,512 | 17,397 | several face-swap and reenactment methods | parts 0–7 train, 8 val, 9 test |

Real training videos contribute more frames (64–96) than fake ones (12–32) because real faces are
the scarcer class; every validation and test video contributes 32 frames.

## Using it

```bash
pip install -r requirements.txt          # Python 3.10+; PyTorch uses CUDA, Apple MPS or CPU
deepfake-detect clip.mp4 photo.jpg       # after `pip install -e .`, or: python -m deepfake_detector.cli ...
python app/app.py                        # web demo at http://127.0.0.1:7860
```

```python
from deepfake_detector.inference import DeepfakeDetector

detector = DeepfakeDetector()            # weights/deepfake_detector.safetensors
result = detector.analyze("clip.mp4")    # or an image path
print(result.verdict, result.fake_probability)
for face in result.faces:
    print(face.frame_index, face.box, face.fake_probability)
```

The first run downloads the frozen CLIP encoder (about 1.7 GB, from Hugging Face) and the face
detector (17 MB). The trained part of the model is the small file in `weights/`.

## Reproducing the training

Training ran on Kaggle (two NVIDIA T4 GPUs; face extraction on CPU sessions). Each step is a job
whose script embeds the exact package code it ran with:

```bash
python kaggle/launch.py queue extract-ffpp-0 extract-ffpp-1 extract-cdf-0 extract-cdf-1 \
    extract-dfd-0 extract-dfd-1 extract-dfdc-{0..7}      # videos -> aligned face crops (shards)
python kaggle/launch.py push train-main                   # training + evaluation on 2 x T4
python kaggle/launch.py fetch train-main                  # logs, metrics and weights
```

The same modules run anywhere: `python -m deepfake_detector.extract`, `torchrun -m
deepfake_detector.train`, `python -m deepfake_detector.evaluate`.

## Project layout

```
deepfake_detector/
  faces.py       SCRFD face detector + landmark alignment
  video.py       evenly spaced frame sampling
  sources.py     dataset adapters and train/val/test splits
  extract.py     videos -> face crops, packed into shards (shards.py)
  model.py       CLIP ViT-L/14 with trainable LayerNorms + linear head
  losses.py      alignment and uniformity losses
  augment.py     training augmentations
  data.py        dataset over shards, balanced sampling
  train.py       training loop (multi-GPU, mixed precision, resumable)
  evaluate.py    metrics per dataset and manipulation method
  inference.py   image/video API used by the CLI and the app
  explain.py     Grad-CAM heatmaps
app/app.py       Gradio web demo
kaggle/          job definitions and launcher
tests/           unit tests (pytest)
```

## Limitations

- **Faces only.** No face, no verdict; tiny faces (under about 40 px) are skipped.
- **New manipulation methods.** Accuracy on methods unlike any in the training data will be lower
  than on the test sets above; the cross-dataset results show how much lower.
- **Heavy re-compression and low resolution** (small messaging-app videos) make every detector less
  reliable; the training augmentations reduce, but do not remove, that effect.
- The score is evidence, not proof. Use it to decide what deserves a closer look.

## Data sources and licences

The datasets are research datasets with their own terms of use (non-commercial research use;
FaceForensics++, Celeb-DF and DFDC require accepting their terms). They are not redistributed here.
The face detector weights (InsightFace) are for non-commercial research use. CLIP is MIT-licensed.
The GenD reference implementation is MIT-licensed; this repository re-implements its training
recipe and uses the authors' released model only as a comparison baseline.

## References

- Yermakov, Čech, Matas, Fritz. *Deepfake Detection that Generalizes Across Benchmarks.* WACV 2026. [arXiv:2508.06248](https://arxiv.org/abs/2508.06248)
- A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
- T. Wang, P. Isola. *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML 2020.
- J. Guo et al. *Sample and Computation Redistribution for Efficient Face Detection (SCRFD).* ICLR 2022.
- A. Rössler et al. *FaceForensics++.* ICCV 2019. · Y. Li et al. *Celeb-DF.* CVPR 2020. · B. Dolhansky et al. *The DeepFake Detection Challenge Dataset.* 2020.

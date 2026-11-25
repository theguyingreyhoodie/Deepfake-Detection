# How the deepfake detector works, step by step

This is the long-form companion to the README: what every part of the system does, why it was
designed that way, and the questions it should be able to answer. All numbers referred to here are in
the README's results section, which is generated from the evaluation files.

## 1. The problem

A **deepfake** here is a video or photo of a real person whose face was manipulated by software:

- **Face swap**: another person's face is pasted onto the target's head and body (DeepFaceLab-style
  autoencoders, FaceSwap, FaceShifter, and modern one-shot swappers such as InSwapper and SimSwap, the
  engines behind tools like Roop and FaceFusion).
- **Face reenactment / lip-sync**: the target keeps their identity but their expressions or mouth are
  driven by someone else (Face2Face, NeuralTextures, Wav2Lip, SadTalker).

Fully AI-generated people or scenes (text-to-video, GAN faces) are a different problem and out of scope.

The hard part is not detecting the fakes in one dataset (many models reach about 99% there) but
**generalising**: catching fakes made by a method the detector never saw, in videos recorded and
compressed differently. Classic CNN detectors trained on one dataset often lose 20–30 AUC points on another.

## 2. Pipeline

```
video -> 32 frames -> face detection -> alignment -> 224x224 crop -> classifier -> per-face probability
      -> max over faces in a frame -> mean over frames -> compare with the threshold -> verdict
```

### 2.1 Frames
32 evenly spaced frames (the centre of 32 equal segments). One frame can mislead (motion blur, a hand
over the face); averaging over 32 makes the decision stable. Still images are treated as a one-frame video.

### 2.2 Face detection: SCRFD
SCRFD-10G (InsightFace) predicts, on three grids (strides 8, 16, 32 px), a face score, a box and five
landmarks per cell. Boxes above 0.5 confidence go through non-maximum suppression. Faces smaller than
40 px or below 0.6 confidence are ignored. If nothing is found, detection is retried at smaller input sizes,
because a photo that is already a tight face crop would otherwise be enlarged beyond the detector's range.

### 2.3 Alignment
The five landmarks are mapped onto a fixed template by a similarity transform (rotation, uniform scale,
shift). Every crop therefore shows an upright face with the eyes at the same place, plus a 1.3x margin so
that the face boundary (where a swap is blended in) and some hair and neck are visible. The warp is done
at the face's own resolution and then shrunk to 224 x 224 with an anti-aliasing filter, so no detail is
lost to the rotation. **The same function makes the training crops and the app's crops**, so the model is
never shown a crop style it was not trained on.

### 2.4 Classifier: CLIP ViT-L/14 with LayerNorm tuning (GenD recipe)
- **CLIP** is an image encoder trained by OpenAI on 400 million image–caption pairs. It is a Vision
  Transformer: the image is cut into 16 x 16 patches of 14 x 14 px, each patch becomes a 1024-number
  vector, and 24 transformer layers let every patch exchange information with every other. A special CLS
  vector summarises the whole image.
- **Frozen backbone.** All 303 million weights stay as CLIP learned them.
- **Only LayerNorms are trained.** Each transformer layer has two LayerNorms that rescale and shift every
  feature channel (a scale and a shift per channel). There are 102,400 such numbers, about 0.03% of the network,
  plus a 2-way linear layer: 104,450 trainable parameters. Adjusting only these re-weights features CLIP
  already has (texture, sharpness, lighting consistency, skin detail) towards the question "manipulated
  or not?" without the capacity to memorise one dataset's quirks. The GenD paper shows that full
  fine-tuning overfits fast and generalises worse.
- **L2-normalised features.** The CLS vector is scaled to length 1 before the linear layer, so all
  features live on a sphere and only their direction matters.
- **Loss** = cross-entropy + 0.1 x alignment + 0.5 x uniformity.
  *Alignment* is the average squared distance between features of the same class (pull each class
  together); *uniformity* is the log of the average Gaussian similarity between all pairs (spread
  features over the sphere so they do not collapse onto a single shortcut direction).

### 2.5 Video decision
Per frame, the most suspicious face counts (one fake face makes the frame fake); the video score is the
mean of the per-frame scores. The threshold is chosen once, on validation videos, as the value with the
best balanced accuracy, and stored inside the weights file.

## 3. Data and how leakage was prevented

| Source | Used for | Split rule |
|---|---|---|
| FaceForensics++ c23 | train / val / test | the official 720 / 140 / 140 video split |
| Celeb-DF v2 | train / val / test | official 518-video test list; the rest split by person (the official list holds out videos, not people: its celebrities also appear in the other videos) |
| DFD (Google) | train / val / test | by the actor whose video was manipulated: 19 train, 3 val, 6 test (the pasted face can belong to an actor on the other side) |
| DFDC parts 0–9 | train / val / test | whole recording batches: parts 0–7 train, 8 val, 9 test |
| DF40 face-swap images | train / val / test | by the FaceForensics++ video each image came from (the Kaggle copy's own split put FF++ test videos into training, so it was ignored) |
| DF40 fake videos | test only | reenactment methods never used in training |

Rules that keep the evaluation honest:
- splits are fixed before any training and never depend on results;
- the model is chosen and the threshold is set on **validation** data; test data is used once;
- the label may shape *training* crops only (in a fake training video with two equally large faces we do
  not know which was swapped, so such frames are skipped); **validation and test crops ignore the label**
  and keep every prominent face, exactly as the app does.

**Balanced sampling.** Each batch draws equally from each full dataset (DF40 images at half weight), 50/50
real/fake within a dataset, and equally from each manipulation method. Real videos are scarcer (DFDC has
seven fakes per real), so real training videos contribute more frames (64–96 vs. 12–32).

**Augmentation.** Flip, rotation up to 10 degrees, 10% shift, 10% zoom, blur, brightness/contrast jitter,
down-and-up rescaling, noise and JPEG re-compression at quality 40–100. The model should key on
manipulation traces, not on one dataset's sharpness or compression level.

## 4. Training setup

- Kaggle, two NVIDIA T4 GPUs, PyTorch DistributedDataParallel, fp16 mixed precision (frozen weights kept
  in fp16, trainable ones in fp32).
- Adam (no weight decay), global batch 128, learning rate warmed up from 1e-5 to 3e-4 over the first epoch
  then cosine-decayed back to 1e-5; 9 epochs of 250,000 samples each.
- After each epoch the model is scored on validation videos; the best mean video AUC is kept.
- Faces were extracted beforehand by CPU jobs into packed shard files (one large file of JPEGs plus an
  index), so the GPUs never wait for video decoding.

## 5. Metrics

- **AUC**: probability that a random fake scores higher than a random real. Threshold-free; the standard
  headline number in deepfake papers. 50% is chance, 100% perfect separation.
- **Balanced accuracy**: mean of the accuracy on reals and on fakes. Plain accuracy is misleading when one
  class dominates (FF++ test has five fakes per real).
- **EER**: error rate at the threshold where false alarms equal misses.
- **Detection rate** on the fakes-only DF40 videos: the share caught at the validation threshold.

## 6. Experiments

1. **Pilot (reproduction).** The GenD setup alone (FF++, four methods) to check the implementation, then
   tested on FaceShifter (unseen method) and on every other dataset (all unseen for it), side by side with
   the authors' released model.
2. **Main model.** All sources together; tested on every dataset's held-out test split.
3. **Unseen methods.** DF40 videos from four reenactment / lip-sync methods that no training set contains.

The main model, the pilot and the authors' model are all scored in one final job on the same evaluation
crops, so every comparison in the README is like for like.

## 7. Questions to expect

**Why not train a CNN like XceptionNet or EfficientNet end to end?**
They learn dataset-specific artefacts and generalise poorly to new manipulation methods. Large
pre-trained encoders with small, targeted fine-tuning currently generalise best (GenD, Effort, ForAda).

**Why train so few parameters?** Fewer free parameters means less room to memorise the training
datasets' quirks; the pre-trained features already describe fine texture and lighting.

**What does a LayerNorm do, and why is tuning it enough?** It normalises each token's features and then
applies a learned per-channel scale and shift. Changing those scales decides which existing features are
amplified or suppressed in every layer, a cheap but powerful way to re-purpose the network.

**Why the 1.3x margin?** Swaps are blended into the original head; the boundary between the pasted face
and the real skin, hair and jaw is one of the strongest cues, so it must be inside the crop.

**How do you know the test results are not inflated?** Official or identity-based splits; whole DFDC
recording batches held out; DF40 re-split by FF++ source video; threshold and model chosen on validation;
evaluation crops independent of labels; unseen datasets and methods reported separately.

**Do test videos show people the model saw in training?** For FaceForensics++ and the DF40 images, no:
both people involved in every fake are on the same side of the split. For two datasets, partly, and the
README counts it: Celeb-DF's official test list shares its celebrities with the other Celeb-DF videos, and
a DFD fake is assigned to the actor whose video was manipulated while the pasted face often belongs to
another actor. DFDC has no actor labels, so shared actors cannot be ruled out. Those in-domain numbers are
therefore optimistic. The honest tests of generalisation are the ones where the data really is new: the
DF40 reenactment videos (methods in no training set) and the FF++-only pilot scored on the other datasets.
A stricter split (dropping every fake whose two people fall on different sides) would remove a large
share of DFD's fakes and needs a new training run.

**Why does the score sometimes disagree with a human?** Heavy compression, very small faces, extreme pose
and occlusion remove the traces the model relies on. The score is evidence, not proof.

**What would improve it further?** More modern manipulation methods in training (the full DF40, DeepSpeak),
an ensemble with a second backbone (DINOv3 or Perception Encoder performed similarly in the GenD paper),
temporal models for lip-sync fakes, and audio–visual consistency checks.

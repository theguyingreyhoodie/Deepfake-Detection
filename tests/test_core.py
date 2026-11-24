"""Fast unit tests for the pipeline's core logic (no model downloads needed)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from deepfake_detector import sources
from deepfake_detector.data import WeightedEpochSampler, balanced_weights, subsample_videos
from deepfake_detector.extract import choose_faces
from deepfake_detector.faces import LANDMARK_TEMPLATE, Face, align_face
from deepfake_detector.losses import alignment_loss, uniformity_loss
from deepfake_detector.metrics import best_threshold, binary_metrics, video_scores
from deepfake_detector.shards import ShardReader, ShardWriter, find_shard_files, load_indexes
from deepfake_detector.train import lr_at
from deepfake_detector.video import evenly_spaced


def test_evenly_spaced():
    idx = evenly_spaced(300, 32)
    assert len(idx) == 32 and idx == sorted(set(idx)) and 0 < idx[0] and idx[-1] < 300
    assert evenly_spaced(5, 32) == [0, 1, 2, 3, 4]
    assert evenly_spaced(0, 8) == []


def _dots_image(points, size=(480, 640)):
    img = np.zeros((*size, 3), np.uint8)
    for x, y in points:
        cv2.circle(img, (int(round(x)), int(round(y))), 3, (255, 255, 255), -1)
    return img


def _blob_centres(gray):
    n, _, stats, cents = cv2.connectedComponentsWithStats((gray > 100).astype(np.uint8))
    return cents[1:][stats[1:, cv2.CC_STAT_AREA] > 3]


def test_align_face_rotates_landmarks_onto_template():
    # a "face" whose 5 landmarks are the template, scaled by 150 px, rotated 30 degrees and moved
    angle = np.deg2rad(30)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pts = (LANDMARK_TEMPLATE - 0.5) * 150 @ rot.T + np.array([320, 240])
    crop = align_face(_dots_image(pts), pts, out_size=224, margin=1.3)
    assert crop.shape == (224, 224, 3)
    found = _blob_centres(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    expected = (LANDMARK_TEMPLATE + 0.15) / 1.3 * 224  # template shrunk by the 1.3 margin
    assert len(found) == 5
    for e in expected:  # every template point has a dot within 3 px
        assert np.min(np.linalg.norm(found - e, axis=1)) < 3


def _face(x, y, side, score=0.9):
    return Face(box=np.array([x, y, x + side, y + side], float), landmarks=np.zeros((5, 2)), score=score)


def test_choose_faces_rules():
    big, other, small = _face(0, 0, 200), _face(300, 0, 180), _face(300, 0, 60)
    assert choose_faces([big, other], label=0) == [big, other]  # real: both genuine faces kept
    assert choose_faces([big, other], label=1) == []  # fake: ambiguous which face was swapped
    assert choose_faces([big, small], label=1) == [big]  # fake: one face clearly dominates
    assert choose_faces([_face(0, 0, 30)], label=0) == []  # too small
    assert choose_faces([_face(0, 0, 200, score=0.3)], label=0) == []  # low confidence


def test_losses():
    z = torch.nn.functional.normalize(torch.randn(64, 16), dim=1)
    labels = torch.randint(0, 2, (64,))
    same = torch.nn.functional.normalize(torch.ones(8, 16), dim=1)
    assert alignment_loss(same, torch.zeros(8)).item() == pytest.approx(0.0, abs=1e-6)
    assert alignment_loss(z, labels).item() > 0.5
    assert uniformity_loss(z).item() < uniformity_loss(same).item()  # spread out is "more uniform"


def test_balanced_weights_shares():
    rows = []
    for ds, n_real, fakes in [("A", 10, {"m1": 50, "m2": 10}), ("B", 5, {"m3": 100})]:
        rows += [dict(dataset=ds, label=0, method="real")] * n_real
        for m, n in fakes.items():
            rows += [dict(dataset=ds, label=1, method=m)] * n
    index = pd.DataFrame(rows)
    w = balanced_weights(index, {"A": 3.0, "B": 1.0})
    share = lambda mask: w[mask.to_numpy()].sum()  # noqa: E731
    assert share(index.dataset == "A") == pytest.approx(0.75)
    assert share((index.dataset == "A") & (index.label == 0)) == pytest.approx(0.375)
    assert share(index.method == "m1") == pytest.approx(share(index.method == "m2"))
    assert w.sum() == pytest.approx(1.0)


def test_sampler_resume_continues_the_same_stream():
    w = np.random.default_rng(0).random(1000)
    full = WeightedEpochSampler(w / w.sum(), 500, seed=1, rank=0)
    full.set_epoch(3)
    stream = list(full)
    full.set_epoch(3, start=123)
    assert list(full) == stream[123:]
    other_rank = WeightedEpochSampler(w / w.sum(), 500, seed=1, rank=1)
    other_rank.set_epoch(3)
    assert list(other_rank) != stream


def test_lr_schedule():
    assert lr_at(0, 1000, 100, 3e-4, 1e-5) == pytest.approx(1e-5)
    assert lr_at(100, 1000, 100, 3e-4, 1e-5) == pytest.approx(3e-4)
    assert lr_at(1000, 1000, 100, 3e-4, 1e-5) == pytest.approx(1e-5)


def test_shards_roundtrip(tmp_path):
    w = ShardWriter(tmp_path, "t", max_bytes=10)  # tiny limit forces several shard files
    blobs = [bytes([i]) * (i + 3) for i in range(8)]
    for i, b in enumerate(blobs):
        w.add(b, dataset="X", label=i % 2, video_id=f"v{i}")
    w.close()
    index = load_indexes([tmp_path])
    reader = ShardReader(find_shard_files([tmp_path]))
    assert index.shard.nunique() > 1
    for row, b in zip(index.itertuples(), blobs):
        assert reader.read(row.shard, row.offset, row.length) == b


def test_ffpp_official_splits_are_disjoint():
    ids = {}
    for split in ("train", "val", "test"):
        pairs = json.load(open(sources.SPLITS_DIR / f"ff_{split}.json"))
        ids[split] = {x for p in pairs for x in p}
    assert [len(ids[s]) for s in ("train", "val", "test")] == [720, 140, 140]
    assert not (ids["train"] & ids["test"]) and not (ids["train"] & ids["val"])


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_faceforensics_adapter_assigns_pairs_to_the_right_split(tmp_path):
    root = tmp_path / "FaceForensics++_C23"
    test_a, test_b = json.load(open(sources.SPLITS_DIR / "ff_test.json"))[0]
    train_a, _ = json.load(open(sources.SPLITS_DIR / "ff_train.json"))[0]
    _touch(root / "original" / f"{test_a}.mp4")
    _touch(root / "original" / f"{train_a}.mp4")
    _touch(root / "Deepfakes" / f"{test_b}_{test_a}.mp4")
    _touch(root / "Face2Face" / "placeholder.txt")
    jobs = sources.faceforensics(tmp_path).set_index("video_id")
    assert jobs.loc[f"real/{test_a}", "split"] == "test"
    assert jobs.loc[f"real/{train_a}", "split"] == "train"
    assert jobs.loc[f"Deepfakes/{test_b}_{test_a}", "split"] == "test"
    assert jobs.loc[f"Deepfakes/{test_b}_{test_a}", "label"] == 1


def test_dfd_adapter_parses_names_and_splits_by_actor(tmp_path):
    _touch(tmp_path / "orig" / "02__kitchen_pan.mp4")
    _touch(tmp_path / "fake" / "02_13__kitchen_pan__ABCD1234.mp4")
    _touch(tmp_path / "fake" / "04_13__kitchen_pan__ABCD1234.mp4")
    jobs = sources.dfd(tmp_path).set_index("video_id")
    assert jobs.loc["02__kitchen_pan", ["label", "split"]].tolist() == [0, "test"]  # actor 02 is a test actor
    assert jobs.loc["02_13__kitchen_pan__ABCD1234", ["label", "split"]].tolist() == [1, "test"]
    assert jobs.loc["04_13__kitchen_pan__ABCD1234", "split"] == "train"


def test_metrics_and_video_rule():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["auc"] == 1.0 and m["accuracy"] == 1.0 and m["eer"] == 0.0
    crops = pd.DataFrame({"dataset": "D", "video_id": ["a"] * 4 + ["b"] * 2, "frame": [0, 0, 1, 1, 0, 1],
                          "label": [1, 1, 1, 1, 0, 0], "method": "x", "score": [0.9, 0.1, 0.5, 0.3, 0.2, 0.4]})
    v = video_scores(crops).set_index("video_id")
    assert v.loc["a", "score"] == pytest.approx((0.9 + 0.5) / 2)  # max face per frame, then mean
    assert v.loc["b", "score"] == pytest.approx(0.3)
    assert 0.0 <= best_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.9]) <= 1.0


def test_subsample_videos_limits_videos_and_crops():
    index = pd.DataFrame({"dataset": "D", "method": "m", "video_id": np.repeat([f"v{i}" for i in range(10)], 5),
                          "label": 1, "frame": np.tile(range(5), 10)})
    out = subsample_videos(index, max_videos_per_class=4, max_crops_per_video=2)
    assert out.video_id.nunique() == 4 and out.groupby("video_id").size().max() == 2

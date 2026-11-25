"""Turn the evaluation and extraction outputs into results/ files and the README's generated sections.

    python scripts/make_results.py \
        --main build/outputs/eval-final/eval_ours --gend build/outputs/eval-final/eval_gend \
        --pilot build/outputs/eval-final/eval_pilot \
        --history build/outputs/train-main/run/history.json --manifests build/outputs

--main, --gend and --pilot are evaluation folders (metrics.json, predictions.parquet) of three models scored
on the same validation and test crops. --history is the training run's per-epoch validation log and
--manifests the folder holding the extraction jobs' outputs (one *_videos.parquet row per video or image).

Every number in the README comes from these files; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASETS = ["FF++", "CDFv2", "DFD", "DFDC", "DF40-img"]  # test sets with real and fake items
NAMES = {"FF++": "FaceForensics++", "CDFv2": "Celeb-DF v2", "DFD": "DFD (Google)", "DFDC": "DFDC",
         "DF40-img": "DF40 face swaps (images)", "DF40-vid": "DF40 fake videos"}
FFPP_FOUR = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]  # GenD's FF++ training methods
REENACTMENT = ["dagan", "hyperreenact", "sadtalker", "wav2lip"]  # DF40 video methods absent from all training data
METHOD_NAMES = {"dagan": "DaGAN", "hyperreenact": "HyperReenact", "sadtalker": "SadTalker", "wav2lip": "Wav2Lip",
                "blendface": "BlendFace", "e4s": "E4S", "facedancer": "FaceDancer", "faceswap": "FaceSwap",
                "fsgan": "FSGAN", "inswap": "InSwapper", "mobileswap": "MobileSwap", "simswap": "SimSwap",
                "uniface": "UniFace"}
# Facts about each source that are design choices, not results: name, manipulations, split rule.
SOURCES = {
    "FF++": ("[FaceForensics++](https://github.com/ondyari/FaceForensics) (c23)",
             "Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter", "official 720 / 140 / 140 source videos"),
    "CDFv2": ("[Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)", "improved face swap",
              "official 518-video test list; the rest by person"),
    "DFD": ("[DFD](https://research.google/blog/contributing-data-to-deepfake-detection-research/) (Google/Jigsaw)",
            "face swap", "by the actor whose video was manipulated: 19 / 3 / 6 actors"),
    "DFDC": ("[DFDC](https://ai.meta.com/datasets/dfdc/), parts 0–9", "several face-swap and reenactment methods",
             "recording batches: parts 0–7 / 8 / 9"),
    "DF40-img": ("[DF40](https://github.com/YZY-stack/DF40) face-swap images", None,
                 "by the FaceForensics++ video each image comes from"),
    "DF40-vid": ("DF40 fake videos", None, "test only"),
}
# Light and dark renderings of the ROC figure: chart surface, ink, grid and three series colours
# (the first three slots of a palette checked for colour-blind separation in both modes).
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70"]},
}


def load(path: str | None) -> dict | None:
    return json.loads((Path(path) / "metrics.json").read_text()) if path else None


def pct(x) -> str:
    return "–" if x is None or x != x else f"{100 * x:.1f}"


def count(x: int) -> str:
    return f"{x:,}" if x else "–"


def bold_best(values: list) -> list[str]:
    """Format as percentages and bold the highest value (all of them when they tie at the shown precision)."""
    shown = [pct(v) for v in values]
    nums = [float(s) for s in shown if s != "–"]
    if len(nums) < 2:
        return shown
    return [f"**{s}**" if s != "–" and float(s) == max(nums) else s for s in shown]


def test_auc(m: dict, dataset: str, methods: list[str] | None = None) -> float | None:
    """Video-level test AUC on a dataset, or the mean over some of its manipulation methods."""
    t = m["test"]["at_0.5"].get(dataset)  # AUC does not depend on the threshold
    if t is None:
        return None
    if methods is None:
        return t["video"].get("auc")
    vals = [t["methods"][x]["auc"] for x in methods if x in t["methods"]]
    return sum(vals) / len(vals) if vals else None


# ---- tables -----------------------------------------------------------------------------------------

def main_table(m: dict, gend: dict | None) -> str:
    t = m["test"]["at_val_threshold"]
    head = "| Test set | Videos (real / fake) | AUC | Balanced accuracy | Accuracy | EER |"
    head += " GenD released model, AUC |" if gend else ""
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for d in DATASETS:
        if d not in t:
            continue
        v = t[d]["video"]
        auc = bold_best([v.get("auc"), test_auc(gend, d)] if gend else [v.get("auc")])
        row = (f"| {NAMES[d]} | {v['n_real']:,} / {v['n_fake']:,} | {auc[0]} | "
               f"{pct(v['balanced_accuracy'])} | {pct(v['accuracy'])} | {pct(v.get('eer'))} |")
        lines.append(row + (f" {auc[1]} |" if gend else ""))
    means = bold_best([t["mean_video_auc"]] + ([gend["test"]["at_0.5"]["mean_video_auc"]] if gend else []))
    lines.append(f"| **Mean of the test sets** | | {means[0]} | | | |" + (f" {means[1]} |" if gend else ""))
    return "\n".join(lines)


FFPP_WHAT = {"NeuralTextures": "a reenactment method that re-renders only the mouth region",
             "Face2Face": "a reenactment method that transfers expressions and keeps the identity",
             "Deepfakes": "an autoencoder face swap", "FaceSwap": "a graphics-based face swap",
             "FaceShifter": "a GAN face swap"}


def ffpp_note(m: dict, gend: dict | None) -> str:
    """A sentence on FaceForensics++ when the released model, trained on it alone, scores higher there."""
    ours, theirs = test_auc(m, "FF++"), test_auc(gend, "FF++") if gend else None
    if ours is None or theirs is None or float(pct(theirs)) <= float(pct(ours)):
        return ""
    methods = m["test"]["at_0.5"]["FF++"].get("methods", {})
    gaps = {x: test_auc(gend, "FF++", [x]) - test_auc(m, "FF++", [x]) for x in methods
            if test_auc(gend, "FF++", [x]) is not None}
    note = (f"On FaceForensics++, the one dataset the released GenD model was trained on, that model scores higher "
            f"({pct(theirs)} vs. {pct(ours)}).")
    if gaps:
        x = max(gaps, key=gaps.get)
        what = f", {FFPP_WHAT[x]}" if x in FFPP_WHAT else ""
        note += (f" The largest gap is on {x}{what} ({pct(test_auc(gend, 'FF++', [x]))} vs. "
                 f"{pct(test_auc(m, 'FF++', [x]))}; see the per-method table). FaceForensics++ is one of five datasets "
                 "in this model's training mix, so each of its methods is a small share of the fakes it learned from.")
    return note


def frame_table(m: dict) -> str:
    t = m["test"]["at_val_threshold"]
    lines = ["| Test set | Face crops (real / fake) | AUC | Balanced accuracy |", "|---|---|---|---|"]
    for d in DATASETS:
        if d in t:
            f = t[d]["frame"]
            lines.append(f"| {NAMES[d]} | {f['n_real']:,} / {f['n_fake']:,} | {pct(f.get('auc'))} | "
                         f"{pct(f['balanced_accuracy'])} |")
    return "\n".join(lines)


def method_table(m: dict, gend: dict | None) -> str:
    """Datasets with several manipulation methods: each method's fakes against the dataset's real items."""
    t = m["test"]["at_val_threshold"]
    head = "| Test set | Manipulation | Fakes | AUC (vs. the same real items) |"
    head += (" GenD released model, AUC |" if gend else "") + " Caught by this model |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for d in DATASETS:
        methods = t.get(d, {}).get("methods", {})
        if len(methods) < 2:
            continue
        for method, v in sorted(methods.items()):
            auc = bold_best([v.get("auc"), test_auc(gend, d, [method])] if gend else [v.get("auc")])
            lines.append(f"| {NAMES[d]} | {METHOD_NAMES.get(method, method)} | {v['n_fake']:,} | "
                         + " | ".join(auc) + f" | {pct(v['fake_recall'])} |")
    return "\n".join(lines)


def unseen_table(models: list[tuple[str, dict]]) -> str:
    """DF40 fake videos from methods no training set contains; each model at its own validation threshold."""
    head = "| Method (in no training set) | Fake videos | " + " | ".join(f"Caught by {n}" for n, _ in models) + " |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    first = models[0][1]["test"]["at_val_threshold"]["DF40-vid"]["methods"]
    for meth in REENACTMENT:
        if meth not in first:
            continue
        rates = [m["test"]["at_val_threshold"].get("DF40-vid", {}).get("methods", {}).get(meth, {}).get("fake_recall")
                 for _, m in models]
        lines.append(f"| {METHOD_NAMES[meth]} | {first[meth]['n_fake']} | " + " | ".join(bold_best(rates)) + " |")
    reals = bold_best([m["test"]["at_val_threshold"]["ALL"]["video"]["real_recall"] for _, m in models])
    lines.append("| *For comparison: real test videos passed as real* | | " + " | ".join(reals) + " |")
    return "\n".join(lines)


def pilot_table(pilot: dict, gend: dict) -> str:
    rows = [("FaceForensics++, the four training methods", "in-domain", "FF++", FFPP_FOUR),
            ("FaceShifter (FaceForensics++ videos)", "unseen method", "FF++", ["FaceShifter"]),
            ("Celeb-DF v2", "unseen dataset", "CDFv2", None),
            ("DFD (Google)", "unseen dataset", "DFD", None),
            ("DFDC", "unseen dataset", "DFDC", None),
            ("DF40 face swaps (images)", "unseen methods", "DF40-img", None)]
    lines = ["| Test set | Setting | This re-implementation, AUC | GenD released model, AUC |", "|---|---|---|---|"]
    unseen = {"p": [], "g": []}
    for name, setting, d, methods in rows:
        p, g = test_auc(pilot, d, methods), test_auc(gend, d, methods)
        if p is None and g is None:
            continue
        if setting != "in-domain" and p is not None and g is not None:
            unseen["p"].append(p)
            unseen["g"].append(g)
        lines.append(f"| {name} | {setting} | " + " | ".join(bold_best([p, g])) + " |")
    if unseen["p"]:
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        cells = bold_best([mean(unseen["p"]), mean(unseen["g"])])
        lines.append("| **Mean of the unseen sets** | | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def history_table(history: list[dict]) -> str:
    sets = [d for d in DATASETS if d in history[0]]
    best = max(range(len(history)), key=lambda i: history[i]["mean_video_auc"])  # first maximum, as in train.py
    lines = ["| Epoch | " + " | ".join(NAMES[d] for d in sets) + " | Mean |", "|" + "---|" * (len(sets) + 2)]
    for i, e in enumerate(history):
        cells = " | ".join(pct(e[d]["video_auc"]) for d in sets)
        label = f"**{e['epoch'] + 1} (kept)**" if i == best else str(e["epoch"] + 1)
        lines.append(f"| {label} | {cells} | {pct(e['mean_video_auc'])} |")
    return "\n".join(lines)


# ---- training data ----------------------------------------------------------------------------------

def data_summary(manifests: Path) -> dict:
    """Real and fake items per source, the train / val / test split and the face crops used for training,
    from the manifests written by the training extraction jobs."""
    import pandas as pd

    files = sorted(f for f in manifests.glob("extract-*/*_videos.parquet")
                   if not f.parent.name.startswith("extract-eval-"))  # eval jobs re-extract the same videos
    if not files:
        raise SystemExit(f"no extraction manifests under {manifests}")
    v = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sources = {}
    for d, g in v.groupby("dataset"):
        sources[d] = {"real": int((g.label == 0).sum()), "fake": int((g.label == 1).sum()),
                      "splits": {s: int((g.split == s).sum()) for s in ("train", "val", "test")},
                      "train_crops": int(g.loc[g.split == "train", "crops"].sum()),
                      "fake_methods": sorted(g.loc[g.label == 1, "method"].unique().tolist())}
    return {"sources": sources, "shared_faces": shared_faces(v)}


def shared_faces(v) -> dict:
    """Where held-out videos show people who also appear in training videos.

    Celeb-DF names videos idX_NNNN (real) and idX_idY_NNNN (fake); its official test list holds out videos,
    not people. DFD names fakes AA_BB__scene: actor AA's video with actor BB's face pasted in, and a fake
    belongs to the split of AA, so the pasted face can be a training actor's.
    """
    out = {}
    c = v[v.dataset == "CDFv2"]
    if len(c):
        people = c.video_id.map(lambda s: set(re.findall(r"id(\d+)", s)))
        test, train = set().union(*people[c.split == "test"]), set().union(*people[c.split == "train"])
        out["CDFv2"] = {"test_people": len(test), "test_people_in_training": len(test & train)}
    d = v[v.dataset == "DFD"]
    if len(d):
        split_of = dict(zip(d.loc[d.label == 0, "group"], d.loc[d.label == 0, "split"]))
        fakes = d[d.label == 1]
        face_split = ("actor" + fakes.video_id.str.extract(r"^\d\d_(\d\d)__")[0]).map(split_of)
        out["DFD"] = {"test_fakes": int((fakes.split == "test").sum()),
                      "test_fakes_with_training_face": int(((fakes.split == "test") & (face_split == "train")).sum()),
                      "training_fakes_with_test_face": int(((fakes.split == "train") & (face_split == "test")).sum())}
    return out


def data_table(summary: dict) -> str:
    s = summary["sources"]
    lines = ["| Source | Manipulations | Real | Fake | Train / val / test | Face crops for training | Split |",
             "|---|---|---:|---:|---|---:|---|"]
    for d in DATASETS + ["DF40-vid"]:
        if d not in s:
            continue
        name, manipulations, rule = SOURCES[d]
        e, sp = s[d], s[d]["splits"]
        manipulations = manipulations or ", ".join(METHOD_NAMES.get(x, x) for x in e["fake_methods"])
        lines.append(f"| {name} | {manipulations} | {count(e['real'])} | {count(e['fake'])} | "
                     f"{count(sp['train'])} / {count(sp['val'])} / {count(sp['test'])} | "
                     f"{count(e['train_crops'])} | {rule} |")
    total = sum(e["train_crops"] for e in s.values())
    return "\n".join(lines) + (f"\n\nDF40 images are counted per image, the rest per video. In total "
                               f"**{total:,} face crops** were used for training.")


def independence_note(shared: dict) -> str:
    text = ("**How independent the test sets are.** FaceForensics++ and the DF40 images hold out whole source "
            "videos, with both people of every fake on the same side of the split. DFDC holds out whole recording "
            "batches; it has no actor labels, so the same paid actors may appear on both sides.")
    c, d = shared.get("CDFv2"), shared.get("DFD")
    if c and d:
        people = (f"all {c['test_people']}" if c["test_people_in_training"] == c["test_people"]
                  else f"{c['test_people_in_training']} of the {c['test_people']}")
        text += (f" Two datasets hold out videos rather than faces: {people} people in Celeb-DF's official test list "
                 f"also appear in its other videos, and {d['test_fakes_with_training_face']:,} of DFD's "
                 f"{d['test_fakes']:,} test fakes carry the pasted face of a training actor (and "
                 f"{d['training_fakes_with_test_face']:,} training fakes the face of a test actor). Their in-domain "
                 "results are therefore optimistic. Methods the model has never seen are tested by the DF40 "
                 "reenactment videos below, and datasets it has never seen by the FaceForensics++-only check.")
    return text


# ---- figure -----------------------------------------------------------------------------------------

def roc_figure(models: list[tuple[str, str, dict, Path]], out: Path, theme: dict) -> None:
    """Video-level test ROC curves: one panel per test set (three per row), one line per model, each line
    labelled with its AUC; the spare cell explains the three models."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import roc_curve

    from deepfake_detector.metrics import video_scores

    preds = [pd.read_parquet(folder / "predictions.parquet") for *_, folder in models]
    sets = [d for d in DATASETS if d in set(preds[0].dataset)]
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 9})
    rows = math.ceil((len(sets) + 1) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(9.6, 3.25 * rows), squeeze=False)
    fig.patch.set_facecolor(theme["surface"])
    axes = axes.ravel()
    for ax, d in zip(axes, sets):
        ax.set_facecolor(theme["surface"])
        ax.plot([0, 1], [0, 1], color=theme["axis"], lw=0.8, ls=(0, (3, 3)), zorder=1)  # chance
        for i, ((short, _, metrics, _), p, color) in enumerate(zip(models, preds, theme["series"])):
            v = video_scores(p[(p.dataset == d) & (p.split == "test")])
            fpr, tpr, _ = roc_curve(v.label, v.score)
            ax.plot(fpr, tpr, color=color, lw=1.6, solid_capstyle="round", solid_joinstyle="round",
                    zorder=3 + len(models) - i,  # the first model is drawn on top
                    label=f"{short}  {pct(test_auc(metrics, d))}")
        ax.set_title(NAMES[d], color=theme["ink"], fontsize=10, pad=7, loc="left")
        ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), aspect="equal")
        ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
        ax.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
        ax.grid(color=theme["grid"], lw=0.6, zorder=0)
        ax.tick_params(labelcolor=theme["ink2"], length=0, pad=4)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color(theme["axis"])
        leg = ax.legend(loc="lower right", frameon=False, fontsize=8, handlelength=1.4, borderaxespad=0.2,
                        title="AUC", title_fontsize=8, alignment="right")
        for text in leg.get_texts():
            text.set_color(theme["ink2"])
        leg.get_title().set_color(theme["muted"])
    for ax in axes[len(sets):]:
        ax.axis("off")
    key = axes[len(sets)]
    key.text(0.02, 0.95, "Test sets (DF40: single images)", color=theme["ink"], fontsize=10,
             va="top", transform=key.transAxes)
    y = 0.80
    for (short, long, _, _), color in zip(models, theme["series"]):
        key.plot([0.02, 0.12], [y - 0.025, y - 0.025], color=color, lw=1.6, solid_capstyle="round",
                 transform=key.transAxes)
        key.text(0.16, y, short, color=theme["ink"], fontsize=9, va="top", transform=key.transAxes)
        key.text(0.16, y - 0.075, long, color=theme["ink2"], fontsize=8, va="top", transform=key.transAxes)
        y -= 0.2
    key.text(0.02, y - 0.02, "Further up and to the left is better;\nthe dashed line is guessing.",
             color=theme["ink2"], fontsize=8, va="top", transform=key.transAxes, linespacing=1.4)
    fig.supxlabel("Real videos flagged as fake", color=theme["ink2"], fontsize=9)
    fig.supylabel("Fake videos caught", color=theme["ink2"], fontsize=9)
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    fig.savefig(out, dpi=200, facecolor=theme["surface"])
    plt.close(fig)


# ---- README -----------------------------------------------------------------------------------------

def replace_block(readme: str, tag: str, body: str) -> str:
    pattern = rf"<!-- {tag}:START -->.*?<!-- {tag}:END -->"
    if not re.search(pattern, readme, flags=re.S):
        raise SystemExit(f"README has no {tag} markers")
    return re.sub(pattern, lambda _: f"<!-- {tag}:START -->\n{body}\n<!-- {tag}:END -->", readme, flags=re.S)


def results_section(main_m, gend, pilot, history, summary, figure: bool) -> str:
    t = main_m["test"]["at_val_threshold"]
    parts = ["## Results", "",
             "Held-out **test** videos of each dataset, never used for training or for choosing the model. Each "
             "video is scored from 32 frames. Accuracy and balanced accuracy use the decision threshold chosen on "
             f"validation videos ({main_m['threshold']:.3f}); AUC and EER do not depend on a chosen threshold. The "
             "last column is the GenD authors' released model, trained on FaceForensics++ only and scored on the same "
             "crops: for it, every test set except FaceForensics++ is unseen, so the gap shows what training on the "
             "other datasets adds. In every comparison below, bold marks the best value in the row.",
             "", main_table(main_m, gend), ""]
    if ffpp_note(main_m, gend):
        parts += [ffpp_note(main_m, gend), ""]
    if summary:
        parts += [independence_note(summary["shared_faces"]), ""]
    if figure:
        parts += ["<picture>",
                  '  <source media="(prefers-color-scheme: dark)" srcset="results/roc_test_dark.png">',
                  '  <img alt="Video-level ROC curves of the three models on each test set" '
                  'src="results/roc_test.png">',
                  "</picture>", ""]
    parts += ["### Single images (one face crop at a time)", "",
              "The same test sets with every face crop scored on its own, as when checking a photo.", "",
              frame_table(main_m), "",
              "### Per manipulation method", "", method_table(main_m, gend), ""]
    if "DF40-vid" in t:
        models = [("this model", main_m)] + [(n, m) for n, m in (("the pilot", pilot), ("GenD", gend)) if m]
        parts += ["### Deepfake methods that no training set contains", "",
                  "DF40 videos made with four face-reenactment and lip-sync methods that appear in **no** training set. "
                  "They are all fakes, so the table reports the share caught by each model at its own "
                  "validation-chosen threshold; the last row shows the price of that threshold on real videos. "
                  "(DF40's face-swap videos are left out: the same swap methods, and possibly the same footage, "
                  "are in the DF40 training images.)", "", unseen_table(models), ""]
    if pilot and gend:
        parts += ["### Reproduction check: trained on FaceForensics++ only", "",
                  "Before the main run, the GenD setup was reproduced (FaceForensics++ c23, four methods, no other "
                  "data) to check the implementation and to see how the recipe handles data it never saw. Both models "
                  "below were trained on FaceForensics++ alone and are scored on the same crops; \"the pilot\" in the "
                  "tables above is this re-implementation.", "", pilot_table(pilot, gend), ""]
    if history:
        parts += ["<details>", "<summary>Validation AUC after each training epoch (the kept checkpoint is marked)"
                  "</summary>", "",
                  "Video-level AUC on a fixed sample of validation videos (up to 100 per dataset and manipulation, "
                  "8 face crops each), computed after every epoch to pick the checkpoint.", "",
                  history_table(history), "", "</details>", ""]
    return "\n".join(parts).rstrip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", help="evaluation folder of the main model")
    ap.add_argument("--gend", help="evaluation folder of the GenD released model")
    ap.add_argument("--pilot", help="evaluation folder of the FF++-only re-implementation")
    ap.add_argument("--history", help="history.json of the main training run")
    ap.add_argument("--manifests", help="folder with the extraction jobs' outputs")
    a = ap.parse_args()
    main_m, gend, pilot = load(a.main), load(a.gend), load(a.pilot)
    history = json.loads(Path(a.history).read_text()) if a.history else None
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    readme = (ROOT / "README.md").read_text()

    for name, m in (("main", main_m), ("gend_baseline", gend), ("pilot_ffpp_only", pilot)):
        if m:
            (out / f"{name}_metrics.json").write_text(json.dumps(m, indent=1))
    if history:
        (out / "main_validation_history.json").write_text(json.dumps(history, indent=1))
    summary = None
    if a.manifests:
        summary = data_summary(Path(a.manifests))
        (out / "data_summary.json").write_text(json.dumps(summary, indent=1))
        readme = replace_block(readme, "DATA", data_table(summary))
        print(data_table(summary), "\n")
    if main_m:
        t = main_m["test"]["at_val_threshold"]
        (out / "summary.json").write_text(json.dumps(
            {"threshold": main_m["threshold"], "test_mean_video_auc": t["mean_video_auc"],
             "test_video_auc": {d: t[d]["video"]["auc"] for d in DATASETS if d in t}}, indent=1))
        folders = [("This model", "trained on all five datasets", main_m, Path(a.main))]
        folders += [(short, long, m, Path(p)) for short, long, m, p in (
            ("Pilot", "same recipe, FaceForensics++ only", pilot, a.pilot),
            ("GenD", "authors' released model, FaceForensics++ only", gend, a.gend)) if m]
        figure = all((f / "predictions.parquet").exists() for *_, f in folders)
        if figure:
            roc_figure(folders, out / "roc_test.png", THEMES["light"])
            roc_figure(folders, out / "roc_test_dark.png", THEMES["dark"])
        section = results_section(main_m, gend, pilot, history, summary, figure)
        readme = replace_block(readme, "RESULTS", section)
        print(section)
    (ROOT / "README.md").write_text(readme)


if __name__ == "__main__":
    main()

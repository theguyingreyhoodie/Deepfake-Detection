"""Turn evaluation outputs into results/*.json and the README results section.

    python scripts/make_results.py --main build/outputs/train-main/eval \
        --gend build/outputs/train-ffpp-pilot/eval_gend --pilot build/outputs/train-ffpp-pilot/eval

Every number in the README comes from these metrics files; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["FF++", "CDFv2", "DFD", "DFDC"]
NAMES = {"FF++": "FaceForensics++", "CDFv2": "Celeb-DF v2", "DFD": "DFD (Google)", "DFDC": "DFDC"}


def load(path: str | None) -> dict | None:
    if not path:
        return None
    return json.loads((Path(path) / "metrics.json").read_text())


def pct(x) -> str:
    return "–" if x is None or x != x else f"{100 * x:.1f}"


def main_table(m: dict, gend: dict | None) -> str:
    t = m["test"]["at_val_threshold"]
    g = gend["test"]["at_0.5"] if gend else None
    head = "| Test set | Videos (real / fake) | AUC | Balanced accuracy | Accuracy | EER |"
    if g:
        head += " GenD (FF++ only) AUC |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for d in DATASETS + ["ALL"]:
        if d not in t:
            continue
        v = t[d]["video"]
        row = (f"| {NAMES.get(d, 'All test videos')} | {v['n_real']:,} / {v['n_fake']:,} | **{pct(v.get('auc'))}** | "
               f"{pct(v['balanced_accuracy'])} | {pct(v['accuracy'])} | {pct(v.get('eer'))} |")
        if g:
            row += f" {pct(g.get(d, {}).get('video', {}).get('auc'))} |"
        lines.append(row)
    return "\n".join(lines)


def frame_table(m: dict) -> str:
    t = m["test"]["at_val_threshold"]
    lines = ["| Test set | Face crops (real / fake) | AUC | Balanced accuracy |", "|---|---|---|---|"]
    for d in DATASETS:
        if d in t:
            f = t[d]["frame"]
            lines.append(f"| {NAMES[d]} | {f['n_real']:,} / {f['n_fake']:,} | {pct(f.get('auc'))} | {pct(f['balanced_accuracy'])} |")
    return "\n".join(lines)


def method_table(m: dict) -> str:
    t = m["test"]["at_val_threshold"]
    lines = ["| Dataset | Manipulation | Fake videos | AUC (vs. the dataset's real videos) | Fakes caught |",
             "|---|---|---|---|---|"]
    for d in DATASETS:
        for method, v in sorted(t.get(d, {}).get("methods", {}).items()):
            lines.append(f"| {NAMES[d]} | {method} | {v['n_fake']:,} | {pct(v.get('auc'))} | {pct(v['fake_recall'])} |")
    return "\n".join(lines)


def pilot_table(p: dict, gend: dict | None) -> str:
    t = p["test"]["at_0.5"]
    g = gend["test"]["at_0.5"] if gend else {}
    rows = [("FaceForensics++ (DF, F2F, FS, NT)", "in-domain", "FF++", ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]),
            ("FaceShifter (FF++ videos, unseen method)", "unseen method", "FF++", ["FaceShifter"]),
            ("Celeb-DF v2", "unseen dataset", "CDFv2", None)]
    lines = ["| Test set | Setting | Ours, re-implemented (AUC) | GenD released model (AUC) |", "|---|---|---|---|"]
    for name, setting, d, methods in rows:
        def auc(rep):
            if d not in rep:
                return None
            if methods is None:
                return rep[d]["video"].get("auc")
            vals = [rep[d]["methods"][x]["auc"] for x in methods if x in rep[d]["methods"]]
            return sum(vals) / len(vals) if vals else None
        lines.append(f"| {name} | {setting} | {pct(auc(t))} | {pct(auc(g))} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main")
    ap.add_argument("--gend")
    ap.add_argument("--pilot")
    a = ap.parse_args()
    main_m, gend, pilot = load(a.main), load(a.gend), load(a.pilot)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    for name, m in (("main", main_m), ("gend_baseline", gend), ("pilot_ffpp_only", pilot)):
        if m:
            (out / f"{name}_metrics.json").write_text(json.dumps(m, indent=1))

    parts = []
    if main_m:
        t = main_m["test"]["at_val_threshold"]
        summary = {"threshold": main_m["threshold"],
                   "test_video_auc": {d: t[d]["video"]["auc"] for d in DATASETS if d in t and "auc" in t[d]["video"]}}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        parts += ["## Results", "",
                  "Held-out **test** videos of each dataset (never used for training or for choosing the "
                  "model). Video level: 32 frames per video. Accuracy uses the decision threshold "
                  f"chosen on validation videos ({main_m['threshold']:.3f}). The last column is the GenD authors' "
                  "released model (trained on FaceForensics++ only), scored on the same crops at its default 0.5.",
                  "", main_table(main_m, gend), "",
                  "### Single images (one face crop at a time)", "", frame_table(main_m), "",
                  "### Per manipulation method", "", method_table(main_m), ""]
    if pilot:
        parts += ["### Generalisation check: trained on FaceForensics++ only", "",
                  "Before the main run, the GenD setup was reproduced (FF++ c23, four methods, no other data) to "
                  "check the implementation and to measure how the recipe handles data it never saw.", "",
                  pilot_table(pilot, gend), ""]
    if not parts:
        raise SystemExit("no metrics given")
    readme = (ROOT / "README.md").read_text()
    block = "<!-- RESULTS:START -->\n" + "\n".join(parts) + "\n<!-- RESULTS:END -->"
    readme = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: block, readme, flags=re.S)
    (ROOT / "README.md").write_text(readme)
    print("\n".join(parts))


if __name__ == "__main__":
    main()

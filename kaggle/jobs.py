"""Kaggle job definitions used by ``launch.py``.

datasets: Kaggle datasets mounted under /kaggle/input (raw videos)
inputs:   earlier jobs whose /kaggle/working output is mounted under /kaggle/input
gpu:      True -> 2 x NVIDIA T4; otherwise a 4-vCPU machine (free, no weekly quota)
"""

EXTRACT = "python -m deepfake_detector.extract --root /kaggle/input --out /kaggle/working --workers 4"

FFPP = "xdxd003/ff-c23"
CDF = "reubensuju/celeb-df-v2"
DFDC10 = "pranay22077/dfdc-10"
DFD = "sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset"

JOBS: dict[str, dict] = {}

for shard in range(2):
    JOBS[f"extract-ffpp-{shard}"] = {
        "datasets": [FFPP],
        "commands": [f"{EXTRACT} --dataset FF++ --prefix ffpp{shard} --shard {shard} --num-shards 2 --det-size 320"],
    }
    JOBS[f"extract-cdf-{shard}"] = {
        "datasets": [CDF],
        "commands": [f"{EXTRACT} --dataset CDFv2 --prefix cdf{shard} --shard {shard} --num-shards 2 --det-size 320"],
    }

# DFD: long 1080p clips, 8.5 fakes per real -> fewer frames per fake, many per real.
for shard in range(2):
    JOBS[f"extract-dfd-{shard}"] = {
        "datasets": [DFD],
        "commands": [f"{EXTRACT} --dataset DFD --prefix dfd{shard} --shard {shard} --num-shards 2 "
                     "--train-fake 16 --train-real 96"],
    }

# DFDC (parts 0-9): 20k 1080p clips, 7 fakes per real. Eight shards pack well into Kaggle's
# limit of five concurrent CPU sessions.
DFDC_SHARDS = 8
for shard in range(DFDC_SHARDS):
    JOBS[f"extract-dfdc-{shard}"] = {
        "datasets": [DFDC10],
        "commands": [f"{EXTRACT} --dataset DFDC --prefix dfdc{shard} --shard {shard} --num-shards {DFDC_SHARDS} "
                     "--train-fake 12 --train-real 64"],
    }

# GPU smoke test on synthetic crops: DDP over 2 x T4, fp16, memory and throughput at two batch settings.
TRAIN = "torchrun --nproc_per_node 2 -m deepfake_detector.train"
JOBS["gpu-smoke"] = {
    "gpu": True,
    "pip": "safetensors opencv-python-headless",
    "commands": [
        "python -m deepfake_detector.selftest --out /tmp/synth --videos 400 --frames 8",
        "nvidia-smi --query-gpu=name,memory.total --format=csv",
        f"{TRAIN} --data /tmp/synth --out /kaggle/working/mb32 --micro-batch 32 --batch-size 128 "
        "--samples-per-epoch 12800 --epochs 1 --max-steps 60 --val-videos 20 --val-crops 4 --workers 3",
        f"{TRAIN} --data /tmp/synth --out /kaggle/working/mb64ckpt --micro-batch 64 --batch-size 128 "
        "--samples-per-epoch 12800 --epochs 1 --max-steps 40 --val-videos 20 --val-crops 4 --workers 3 "
        "--grad-checkpointing",
    ],
}

# DF40: modern one-shot face swaps (images, used for training and testing) and 500 videos of eight
# modern swap / reenactment methods (testing only).
JOBS["extract-df40img"] = {
    "datasets": ["mdsahidkhan/df40-dataset"],
    "commands": [f"{EXTRACT} --dataset DF40-img --prefix df40img --det-size 320"],
}
JOBS["extract-df40vid"] = {
    "datasets": ["hafsanaz56/df40-fake-500"],
    "commands": [f"{EXTRACT} --dataset DF40-vid --prefix df40vid"],
}

EVAL = "python -m deepfake_detector.evaluate --data /kaggle/input --splits val test --workers 4"

# Pilot: reproduce the GenD setup (FF++ c23, 4 methods) and test on unseen datasets (Celeb-DF) and
# an unseen method (FaceShifter). Validates the implementation before the main run.
JOBS["train-ffpp-pilot"] = {
    "gpu": True,
    "pip": "safetensors opencv-python-headless",
    "inputs": ["extract-ffpp-0", "extract-ffpp-1", "extract-cdf-0", "extract-cdf-1"],
    "commands": [
        f"{TRAIN} --data /kaggle/input --out /kaggle/working/run --datasets FF++ --exclude-methods FaceShifter "
        "--epochs 6 --samples-per-epoch 100000 --micro-batch 32 --batch-size 128 --workers 3 --time-limit-hours 4.5",
        f"{EVAL} --weights /kaggle/working/run/best.safetensors --out /kaggle/working/eval",
        f"{EVAL} --weights gend --out /kaggle/working/eval_gend",
    ],
}

ALL_EXTRACTS = ([f"extract-ffpp-{i}" for i in range(2)] + [f"extract-cdf-{i}" for i in range(2)]
                + [f"extract-dfd-{i}" for i in range(2)] + [f"extract-dfdc-{i}" for i in range(DFDC_SHARDS)]
                + ["extract-df40img", "extract-df40vid"])

# Main model: every dataset, every method.
JOBS["train-main"] = {
    "gpu": True,
    "pip": "safetensors opencv-python-headless",
    "inputs": ALL_EXTRACTS,
    "commands": [
        f"{TRAIN} --data /kaggle/input --out /kaggle/working/run --epochs 9 --samples-per-epoch 250000 "
        "--micro-batch 32 --batch-size 128 --workers 3 --time-limit-hours 10.8 "
        "--dataset-weights '{\"DF40-img\": 0.5}'",
        f"{EVAL} --weights /kaggle/working/run/best.safetensors --out /kaggle/working/eval",
    ],
}

# The GenD authors' released model (FF++ only), scored on exactly the same val/test crops as ours.
JOBS["eval-gend"] = {
    "gpu": True,
    "pip": "safetensors opencv-python-headless",
    "inputs": ALL_EXTRACTS,
    "commands": [f"{EVAL} --weights gend --out /kaggle/working/eval_gend"],
}


# ---- Evaluation crops, label-independent face selection (see extract.py) -------------------------
# Validation and test videos re-extracted so that every prominent face is kept whatever the label,
# as in the app. The final evaluation reads only these crops.
EVAL_EXTRACTS = []


def _eval_job(name, dataset_ref, flags):
    JOBS[name] = {"datasets": [dataset_ref],
                  "commands": [f"{EXTRACT} --splits val test {flags}"]}
    EVAL_EXTRACTS.append(name)


for shard in range(2):
    _eval_job(f"extract-eval-ffpp-{shard}", FFPP,
              f"--dataset FF++ --prefix effpp{shard} --shard {shard} --num-shards 2 --det-size 320")
    _eval_job(f"extract-eval-dfd-{shard}", DFD,
              f"--dataset DFD --prefix edfd{shard} --shard {shard} --num-shards 2")
_eval_job("extract-eval-cdf", CDF, "--dataset CDFv2 --prefix ecdf --det-size 320")
for shard in range(4):
    _eval_job(f"extract-eval-dfdc-{shard}", DFDC10,
              f"--dataset DFDC --prefix edfdc{shard} --shard {shard} --num-shards 4")
_eval_job("extract-eval-df40img", "mdsahidkhan/df40-dataset", "--dataset DF40-img --prefix edf40img --det-size 320")
_eval_job("extract-eval-df40vid", "hafsanaz56/df40-fake-500", "--dataset DF40-vid --prefix edf40vid")

def _checkpoint_eval(job: str, out: str) -> str:
    """Evaluate the best checkpoint saved by an earlier training job. Kaggle mounts each input job's
    output in a folder named after that job; if the checkpoint is not there, list what is mounted and stop."""
    return (f"W=$(find /kaggle/input -path '*{job}/*' -name best.safetensors | head -1); "
            "if [ -z \"$W\" ]; then find /kaggle/input -maxdepth 4 -type d | head -60; exit 1; fi; "
            f"echo \"weights: $W\"; {EVAL} --weights \"$W\" --out /kaggle/working/{out}")


# Final evaluation on the label-independent evaluation crops. Three models scored on identical crops:
# our main model (train-main's best checkpoint), the GenD authors' released model, and our FF++-only
# pilot, so that the reproduction check compares like with like.
JOBS["eval-final"] = {
    "gpu": True,
    "pip": "safetensors opencv-python-headless",
    "inputs": EVAL_EXTRACTS + ["train-main", "train-ffpp-pilot"],
    "commands": [
        _checkpoint_eval("train-main", "eval_ours"),
        f"{EVAL} --weights gend --out /kaggle/working/eval_gend",
        _checkpoint_eval("train-ffpp-pilot", "eval_pilot"),
    ],
}

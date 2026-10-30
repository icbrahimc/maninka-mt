"""Run this project's steps on Hugging Face Jobs (pay-per-second cloud GPUs).

Runs on YOUR LAPTOP. It doesn't train anything itself: it asks Hugging Face to
start a GPU container, clone this repo from GitHub, and run one step there.
Push your changes to GitHub before launching, or the job runs the old code.

Setup (once):
    pip install -U huggingface_hub
    hf auth login                  # token with Write access
    # add credits: huggingface.co/settings/billing

Steps, in order:
    python launch.py baseline      # untouched NLLB on the test set (the number to beat)
    python launch.py smoke         # tiny training run to prove the pipeline works
    python launch.py train         # real run; pushes the model to <you>/nllb-maninka-v1
    python launch.py eval          # score the trained model on the test set

Options:
    --flavor a10g-small            # override the GPU (list them: hf jobs hardware)
    --timeout 6h                   # override the time limit (jobs default to 30 min)
    --model-repo you/nllb-maninka-v2
    --dry-run                      # print the command without launching
"""
from __future__ import annotations

import argparse

from huggingface_hub import get_token, run_job, whoami

GIT_REPO = "https://github.com/icbrahimc/maninka-mt"
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"

# step -> (command, default GPU flavor, default timeout)
STEPS = {
    "baseline": ("python train_nllb.py --data out --eval-only --split test",
                 "l4x1", "30m"),
    "smoke": ("python train_nllb.py --data out --max-train 2000 --epochs 1 --output runs/smoke",
              "l4x1", "45m"),
    "train": ("python train_nllb.py --data out --output runs/v1 --hub-repo {model_repo}",
              "a100-large", "4h"),
    "eval": ("python train_nllb.py --data out --eval-only --model {model_repo} --split test",
             "l4x1", "30m"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--flavor")
    ap.add_argument("--timeout")
    ap.add_argument("--model-repo", help="default: <your HF username>/nllb-maninka-v1")
    ap.add_argument("--git-repo", default=GIT_REPO)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = get_token()
    if not token:
        raise SystemExit("Not logged in. Run: hf auth login")
    model_repo = args.model_repo or f"{whoami(token)['name']}/nllb-maninka-v1"

    cmd, flavor, timeout = STEPS[args.step]
    cmd = cmd.format(model_repo=model_repo)
    flavor = args.flavor or flavor
    timeout = args.timeout or timeout

    script = " && ".join([
        "apt-get update -qq",
        "apt-get install -y -qq git > /dev/null",
        f"git clone --depth 1 --branch {args.branch} {args.git_repo} repo",
        "cd repo",
        "pip install -q -r requirements.txt",
        cmd,
    ])

    print(f"step:    {args.step}\nflavor:  {flavor}\ntimeout: {timeout}\nrun:     {cmd}")
    if args.dry_run:
        return

    job = run_job(
        image=IMAGE,
        command=["bash", "-c", script],
        flavor=flavor,
        timeout=timeout,
        secrets={"HF_TOKEN": token},  # encrypted; lets the job push the model
    )
    print(f"\nlaunched job {job.id}\nlogs: {job.url}")


if __name__ == "__main__":
    main()

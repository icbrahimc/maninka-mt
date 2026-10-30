"""Upload the prepared splits in out/ to a private Hugging Face dataset repo.

    huggingface-cli login            # paste a token with *write* access
    python push_dataset.py you/maninka-mt-data

Then on any machine:  datasets.load_dataset("you/maninka-mt-data")
"""
import sys

from datasets import load_dataset

repo = sys.argv[1]
ds = load_dataset("json", data_files={s: f"out/{s}.jsonl" for s in ("train", "dev", "test")})
ds.push_to_hub(repo, private=True)
print(f"pushed {ds} to https://huggingface.co/datasets/{repo} (private)")

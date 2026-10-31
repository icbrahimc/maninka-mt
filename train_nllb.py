"""Fine-tune NLLB-200 for Maninka (Latin, via the bam_Latn token) -> English/French.

Baseline (untouched NLLB, no training) on the test set:
    python train_nllb.py --data out --eval-only --split test

Smoke test (tiny run to check the pipeline end to end, ~5 min on an L4):
    python train_nllb.py --data out --max-train 2000 --epochs 1 --output runs/smoke

Real run (A100 ~1-2 h, L4 ~4-6 h):
    python train_nllb.py --data out --output runs/v1

Evaluate a trained checkpoint:
    python train_nllb.py --data out --eval-only --model runs/v1/best --split test
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import sacrebleu
import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

SRC_LANG = "bam_Latn"   # stand-in for Maninka: closest language NLLB knows
EVAL_TGT = "eng_Latn"

# Rows per domain to keep (per target language). None = keep all.
# Dictionaries are huge and single-word; too much of them teaches the model
# to answer in glosses. Quran is already capped in build_dataset.py.
DOMAIN_CAPS = {"dictionary": 12000, "locales": None, "phrases": None,
               "prose": None, "quran": None}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def rebalance(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    seen, out = Counter(), []
    for r in rows:
        cap = DOMAIN_CAPS.get(r["domain"])
        key = (r["domain"], r["tgt_lang"])
        if cap is not None and seen[key] >= cap:
            continue
        seen[key] += 1
        out.append(r)
    print("training mix:", dict(sorted(seen.items())))
    return out


def chrf_pp(preds: list[str], refs: list[str]) -> float:
    return sacrebleu.corpus_chrf(preds, [refs], word_order=2).score


@torch.no_grad()
def translate(model, tok, texts: list[str], tgt_lang: str = EVAL_TGT,
              batch_size: int = 32, num_beams: int = 4) -> list[str]:
    tok.src_lang = SRC_LANG
    bos = tok.convert_tokens_to_ids(tgt_lang)
    out = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to(model.device)
        gen = model.generate(**enc, forced_bos_token_id=bos,
                             num_beams=num_beams, max_new_tokens=128)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def evaluate_split(model, tok, rows: list[dict], show: int = 15) -> float:
    preds = translate(model, tok, [r["src"] for r in rows])
    refs = [r["tgt"] for r in rows]
    score = chrf_pp(preds, refs)
    for r, p in list(zip(rows, preds))[:show]:
        print(f"  {r['src']}\n    ref: {r['tgt']}\n    out: {p}")
    print(f"chrF++ = {score:.2f}  (n={len(rows)})")
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("out"))
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--output", type=Path, default=Path("runs/v1"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--hub-repo", default=None,
                    help="push the best checkpoint to this private HF repo, e.g. you/nllb-maninka-v1")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, src_lang=SRC_LANG)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=dtype)
    model.generation_config.max_length = None  # we use max_new_tokens; avoids a warning per batch
    if torch.cuda.is_available():
        model.to("cuda")

    if args.eval_only:
        rows = load_jsonl(args.data / f"{args.split}.jsonl")
        evaluate_split(model, tok, rows)
        return

    model.to(torch.float32)  # train in fp32 weights + bf16 autocast (stable)
    train_rows = rebalance(load_jsonl(args.data / "train.jsonl"), args.seed)
    if args.max_train:
        train_rows = train_rows[:args.max_train]
    dev_rows = load_jsonl(args.data / "dev.jsonl")

    def encode(ex):
        tok.src_lang = SRC_LANG
        tok.tgt_lang = ex["tgt_lang"]      # sets the target-language prefix token
        return tok(ex["src"], text_target=ex["tgt"], max_length=128, truncation=True)

    cols = ["src", "tgt", "tgt_lang", "domain", "file"]
    train_ds = Dataset.from_list(train_rows).map(encode, remove_columns=cols)
    dev_ds = Dataset.from_list(dev_rows).map(encode, remove_columns=cols)

    # dev is all English, so generation during eval is forced to English
    model.generation_config.forced_bos_token_id = tok.convert_tokens_to_ids(EVAL_TGT)
    model.generation_config.max_new_tokens = 128
    model.generation_config.num_beams = 4

    def compute_metrics(p):
        preds, labels = p
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        dp = tok.batch_decode(preds, skip_special_tokens=True)
        dl = tok.batch_decode(labels, skip_special_tokens=True)
        return {"chrf++": chrf_pp(dp, dl)}

    targs = Seq2SeqTrainingArguments(
        output_dir=str(args.output),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        warmup_ratio=0.05,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        eval_strategy="steps", eval_steps=1000,
        save_strategy="steps", save_steps=1000, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="chrf++",
        greater_is_better=True,
        predict_with_generate=True,
        bf16=dtype == torch.bfloat16,
        logging_steps=100,
        report_to="none",
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        processing_class=tok,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    best = args.output / "best"
    trainer.save_model(str(best))
    tok.save_pretrained(str(best))
    print(f"saved best checkpoint to {best}")
    if args.hub_repo:
        trainer.model.push_to_hub(args.hub_repo, private=True)
        tok.push_to_hub(args.hub_repo, private=True)
        print(f"pushed to https://huggingface.co/{args.hub_repo} (private)")
    print("dev samples:")
    evaluate_split(trainer.model, tok, dev_rows)


if __name__ == "__main__":
    main()

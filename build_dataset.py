"""Build a clean Maninka(Latin) -> English/French training set from nicolingua-0005.

    python3 -I build_dataset.py data/nicolingua/data out/

Writes out/train.jsonl, out/dev.jsonl, out/test.jsonl and out/stats.txt.
Each row: {"src": maninka, "tgt": text, "tgt_lang": "eng_Latn"|"fra_Latn",
           "domain": ..., "file": ...}
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nko_translit import nko_to_latin  # noqa: E402

# file stem -> domain (README labels, corrected after inspecting the data)
DOMAINS = {
    "baba_mamadi_diane_parallel_001": "dictionary",   # N'Ko-French dictionary
    "baba_mamadi_diane_parallel_002": "quran",
    "baba_mamadi_diane_parallel_003": "quran",        # short Quran phrases
    "baba_mamadi_diane_parallel_004": "dictionary",   # README says "smartphone strings"; it's a word list
    "kalo_mory_diane_parallel_001": "phrases",
    "kalo_mory_diane_parallel_002": "phrases",
    "kalo_mory_diane_parallel_003": "phrases",
    "solo_farabado_cisse_parallel_001": "locales",
    "solo_farabado_cisse_parallel_002": "phrases",
    "djibrila_diane_parallel_001": "phrases",
    "djibrila_diane_parallel_002": "phrases",
    "djibrila_diane_parallel_003": "phrases",
    "nafadji_sory_conde_parallel_001": "prose",
    "nafadji_sory_conde_parallel_002": "prose",
    "nafadji_sory_conde_parallel_003": "prose",       # literary narrative
    "nafadji_sory_conde_parallel_004": "prose",
}

QURAN_CAP = 5000        # per target language; keeps the model from sounding scriptural
DEV_N, TEST_N = 500, 500
MAX_WORDS = 80
SEED = 13

LANG_EXT = {"eng_Latn": "eng_Latn", "fra_Latn": "fra_Latn"}


def clean_tgt(s: str) -> str:
    s = s.strip().lstrip("—–- ").strip()
    return re.sub(r"\s+", " ", s)


def ok_pair(src: str, tgt: str) -> bool:
    if not src or not tgt:
        return False
    ns, nt = len(src.split()), len(tgt.split())
    if ns > MAX_WORDS or nt > MAX_WORDS:
        return False
    if ns >= 4 and not (0.33 <= ns / max(nt, 1) <= 3.0):  # badly misaligned
        return False
    if re.search(r"[߀-߿]", src):                 # untransliterated residue
        return False
    return True


def main(data_dir: Path, out_dir: Path) -> None:
    rng = random.Random(SEED)
    rows, dropped = [], Counter()

    for nqo in sorted(data_dir.glob("*_parallel_*.nqo_Nkoo")):
        stem = nqo.name.split(".")[0]
        domain = DOMAINS.get(stem, "other")
        src_lines = nqo.read_text(encoding="utf-8").splitlines()
        for lang in LANG_EXT:
            tgt_path = data_dir / f"{stem}.{lang}"
            if not tgt_path.exists():
                continue
            tgt_lines = tgt_path.read_text(encoding="utf-8").splitlines()
            if len(tgt_lines) != len(src_lines):
                print(f"skip {stem}.{lang}: {len(src_lines)} vs {len(tgt_lines)} lines")
                continue
            for s, t in zip(src_lines, tgt_lines):
                src, tgt = nko_to_latin(s), clean_tgt(t)
                if ok_pair(src, tgt):
                    rows.append({"src": src, "tgt": tgt, "tgt_lang": lang,
                                 "domain": domain, "file": stem})
                else:
                    dropped[(stem, lang)] += 1

    # dedupe exact (src, tgt, lang)
    seen, uniq = set(), []
    for r in rows:
        key = (r["src"], r["tgt"].lower(), r["tgt_lang"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    n_dupes = len(rows) - len(uniq)
    rows = uniq

    # cap Quran
    rng.shuffle(rows)
    quran_count, capped = Counter(), []
    for r in rows:
        if r["domain"] == "quran":
            if quran_count[r["tgt_lang"]] >= QURAN_CAP:
                continue
            quran_count[r["tgt_lang"]] += 1
        capped.append(r)
    rows = capped

    # dev/test: English conversational phrases, split by English sentence so
    # variants like "alu ye X" / "i ye X" -> "Do X" never straddle splits
    by_tgt = defaultdict(list)
    for r in rows:
        if r["tgt_lang"] == "eng_Latn" and r["domain"] == "phrases":
            by_tgt[r["tgt"].lower()].append(r)
    keys = sorted(by_tgt)
    rng.shuffle(keys)
    dev, test, held = [], [], set()
    for k in keys:
        bucket = dev if len(dev) < DEV_N else test if len(test) < TEST_N else None
        if bucket is None:
            break
        bucket.append(by_tgt[k][0])
        held.add(k)
    held_src = {r["src"] for r in dev + test}
    train = [r for r in rows
             if not (r["tgt_lang"] == "eng_Latn" and r["tgt"].lower() in held)
             and r["src"] not in held_src]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("dev", dev), ("test", test)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = [f"train {len(train)}  dev {len(dev)}  test {len(test)}",
             f"exact duplicates removed: {n_dupes}",
             f"quran rows kept (cap {QURAN_CAP}/lang): {dict(quran_count)}", "",
             "train rows by domain x target:"]
    c = Counter((r["domain"], r["tgt_lang"]) for r in train)
    stats += [f"  {d:<11} {l}  {n}" for (d, l), n in sorted(c.items())]
    stats += ["", "dropped by filters:"]
    stats += [f"  {s}.{l}: {n}" for (s, l), n in sorted(dropped.items())]
    (out_dir / "stats.txt").write_text("\n".join(stats) + "\n", encoding="utf-8")
    print("\n".join(stats))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

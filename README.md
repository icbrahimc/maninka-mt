# maninka-mt

Fine-tune NLLB-200 to translate Maninka (Latin script) into English, as the
translation half of a song-lyrics pipeline (AudioShake stems → ASR → this).

## What's here

| File | What it does |
|---|---|
| `nko_translit.py` | N'Ko → Latin Maninka transliterator + `to_canonical()` normalizer for any Maninka text (N'Ko, Latin with ɛ/ɔ/ɲ, or loose lyric spellings). Use it on **every** input, training and inference, so spelling is consistent. |
| `build_dataset.py` | Turns the nicolingua-0005 corpus into `out/train.jsonl`, `dev.jsonl`, `test.jsonl`. Caps Quran text, dedupes, filters misaligned pairs, holds out dev/test by English sentence so variants don't leak. |
| `train_nllb.py` | Baseline eval, fine-tuning, and checkpoint eval (chrF++). |
| `out/` | The prepared dataset (already built). |
| `songs/song_test.template.jsonl` | Format for a song-lyric test set. Only use lyrics and translations you have the rights to share (your own, or a native speaker's who agrees to publish them). Never train on the test set. |

Current data (`out/stats.txt`): ~119k train pairs (Maninka → English and → French),
500 dev, 500 test. Training downsamples the dictionary word-lists to 12k per language.

## Run it on RunPod

1. Create a pod: **L4 (24GB)** to start (~$0.40/h), template *RunPod PyTorch 2.x*.
   Switch to an A100 once the smoke test passes.
2. Upload this folder (or `git push` it somewhere and clone), then:

```bash
cd maninka-mt
pip install -r requirements.txt

# 1. Baseline: how good is untouched NLLB (Bambara token) on Maninka?  ~2 min
python train_nllb.py --data out --eval-only --split test

# 2. Smoke test: tiny training run to prove the pipeline works.  ~5 min
python train_nllb.py --data out --max-train 2000 --epochs 1 --output runs/smoke

# 3. Real run.  A100 ~1-2 h, L4 ~4-6 h
python train_nllb.py --data out --output runs/v1

# 4. Score the trained model on the held-out test set
python train_nllb.py --data out --eval-only --model runs/v1/best --split test
```

3. **Stop the pod.** If you passed `--hub-repo`, the model is already on Hugging Face.

### With your Hugging Face account

On the pod, once: `huggingface-cli login` with a **write** token
(huggingface.co → Settings → Access Tokens). Then:

```bash
python push_dataset.py YOUR_USERNAME/maninka-mt-data        # optional backup of the dataset
python train_nllb.py --data out --output runs/v1 --hub-repo YOUR_USERNAME/nllb-maninka-v1
```

Both repos are created **private**. Later, load the model anywhere with
`AutoModelForSeq2SeqLM.from_pretrained("YOUR_USERNAME/nllb-maninka-v1")`.

Write down the baseline chrF++ from step 1. Step 4 should beat it clearly; if it
doesn't, suspect the data (transliteration) before the model.

## Translating new text

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nko_translit import to_canonical
from train_nllb import translate

tok = AutoTokenizer.from_pretrained("runs/v1/best", src_lang="bam_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("runs/v1/best")
print(translate(model, tok, [to_canonical("Fɛn tɛ n bolo")]))
```

## Known limitations

- **Transliteration is rule-based.** It strips tones, merges ɛ/e and ɔ/o
  (lyric transcriptions don't keep them), and restores vowels N'Ko omits
  between same-vowel syllables. Spot-check output on words you know.
- **Domain:** no song or conversational-narrative English data yet. Expect
  stiff, literal output on lyrics. Adding a few hundred native-speaker song
  translations is the single biggest improvement available.
- **Benchmark:** the dev/test sets are short phrases. Add FLoRes N'Ko
  (from the WMT 2023 N'Ko paper's release) for a harder general benchmark.

## License

| Part | License |
|---|---|
| Code (`*.py`) | [MIT](LICENSE) |
| Dataset in `out/` | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), inherited from nicolingua-0005 (see below) |
| Models fine-tuned from NLLB-200 | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), inherited from Meta's NLLB-200 weights: research and non-commercial use only |

## Data source

The dataset in `out/` is derived from
[nicolingua-0005-nqo-nmt-resources](https://github.com/mdoumbouya/nicolingua-0005-nqo-nmt-resources)
(Doumbouya et al.; data donated by Baba Mamadi Diané, Solo Farabado Cissé, Djibrila
Diané, Nafadji Sory Condé and Kalo Mory Diané), licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The files in
`out/` are transliterated and filtered versions of that data and are shared under
the same license. To rebuild them:

```bash
git clone https://github.com/mdoumbouya/nicolingua-0005-nqo-nmt-resources data/nicolingua
python3 build_dataset.py data/nicolingua/data out
```

No song lyrics or lyric translations are included; `songs/` holds only an empty
template. Published lyrics and their translations are usually copyrighted, so a
song test set should come from translations you have permission to release.

# multilingual-hplt-corpus

A pipeline to build a cleaned, deduplicated multilingual pretraining corpus from [HPLT Monolingual v3](https://hplt-project.org/datasets/v3.0). Works with any language combination available in HPLT v3.

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

## Datasets Created Using This Pipeline

| Dataset | Languages | HuggingFace |
|---------|-----------|-------------|
| Indic HPLT v1 | Indic languages | [AM0908/indic-hplt-v1](https://huggingface.co/datasets/AM0908/indic-hplt-v1) |

## Quick Start

```bash
git clone https://github.com/ashtok/multilingual-hplt-corpus.git
cd multilingual-hplt-corpus
uv sync
```

**Test run (1K lines):**
```bash
uv run python main.py --lines 1000
```

**Full run:**
```bash
uv run python main.py
```

**Run specific steps:**
```bash
uv run python main.py --steps download clean
uv run python main.py --skip stats
```

---

## Configuration

Edit `configs/corpus.yaml` to choose your languages and scale:

```yaml
source: https://data.hplt-project.org/three/sorted
total_lines: 10000000
min_wds_score: 0.5
download_buffer: 1.25

languages:
  hin_Deva:
    bcp47: hi
    fraction: 0.1
  ben_Beng:
    bcp47: bn
    fraction: 0.09
  tam_Taml:
    bcp47: ta
    fraction: 0.09
  # ... add/remove any language from HPLT v3

quality:
  min_chars: 50
  max_chars: 100000
  min_avg_word_len: 2.0
  max_non_alpha_ratio: 0.5

dedup:
  exact: true
  minhash: true
  minhash_threshold: 0.7
  num_perm: 128
  ngram: 5
```

**Fractions don't need to sum to 1** — the pipeline normalises them. To find available language codes, browse `https://data.hplt-project.org/three/sorted/`.

---

## Pipeline

```
download → clean → dedup → merge → stats
```

| Step | What it does |
|---|---|
| `download` | Streams `.jsonl.zst` shards from HPLT v3, pre-filters by WDS score, stops at per-language target |
| `clean` | Filters by character length, avg word length, and non-alphabetic ratio; caps each language at its quota |
| `dedup` | Exact SHA-256 dedup on all languages, then MinHash LSH (Jaccard ≥ 0.7) on languages not already globally deduped by HPLT |
| `merge` | Interleaves languages at configured fractions; redistributes quota from short languages to others |
| `stats` | Prints and saves per-language doc/token/word counts + quality distribution to `data/stats.json` |

Output: `data/merged/train.jsonl`

---

## Upload to Hugging Face (optional)

```bash
huggingface-cli login
uv run python src/upload.py
```

Set `REPO_ID` in [src/upload.py](src/upload.py) to your target dataset repo before running.

---

## License

[CC0 1.0 Universal](LICENSE) — inherited from HPLT v3.

Built on [HPLT Monolingual v3](https://hplt-project.org) by the HPLT consortium.

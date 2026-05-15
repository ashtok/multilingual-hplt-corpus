"""Exact + MinHash deduplication."""
import json, hashlib
from pathlib import Path
import yaml
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


CFG_D     = yaml.safe_load(open("configs/corpus.yaml", encoding="utf-8"))["dedup"]
CLEAN_DIR = Path("data/cleaned")
DEDUP_DIR = Path("data/deduped")
DEDUP_DIR.mkdir(parents=True, exist_ok=True)


# HPLT v3 already globally deduped these Indic languages
SKIP_MINHASH = {
    "hin_Deva", "ben_Beng", "tam_Taml", "tel_Telu",
    "mar_Deva", "guj_Gujr", "kan_Knda", "mal_Mlym",
    "pan_Guru", "urd_Arab"
}


def shingles(text, n=5):
    words = text.lower().split()
    return {" ".join(words[i:i+n]) for i in range(max(1, len(words)-n+1))}


def make_minhash(text):
    m = MinHash(num_perm=CFG_D["num_perm"])
    for s in shingles(text, CFG_D["ngram"]):
        m.update(s.encode("utf-8"))
    return m

def dedup_file(path: Path, run_minhash: bool):
    out = DEDUP_DIR / path.name

    if out.exists() and out.stat().st_size > 0:
        actual = sum(1 for _ in open(out, encoding="utf-8"))
        print(f"  ✓ {path.stem} already deduped ({actual:,} lines), skipping")
        return

    # Stage 1: exact SHA-256 dedup — streaming, no full file in RAM
    seen_hashes = set()
    stage1_path = DEDUP_DIR / f"_{path.stem}_stage1.jsonl"

    total = kept_exact = 0
    with open(path, encoding="utf-8") as fin, \
         open(stage1_path, "w", encoding="utf-8") as ftmp:
        for line in tqdm(fin, desc=f"  Exact {path.stem}", leave=False):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                text = json.loads(line)["text"].strip().lower()
            except (json.JSONDecodeError, KeyError):
                continue
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                ftmp.write(line + "\n")
                kept_exact += 1

    del seen_hashes  # free RAM before MinHash

    if not run_minhash:
        stage1_path.rename(out)
        print(f"{path.stem}: exact only → {kept_exact:,}/{total:,} kept")
        return

    # Stage 2: MinHash — stream stage1 temp file, keep LSH in RAM only
    lsh  = MinHashLSH(threshold=CFG_D["minhash_threshold"], num_perm=CFG_D["num_perm"])
    kept = 0
    with open(stage1_path, encoding="utf-8") as fin, \
         open(out, "w", encoding="utf-8") as fout:
        for i, line in tqdm(enumerate(fin), desc=f"  MinHash {path.stem}",
                            leave=False):
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError):
                continue
            m = make_minhash(text)
            if not lsh.query(m):
                lsh.insert(str(i), m)
                fout.write(line + "\n")
                kept += 1

    stage1_path.unlink()  # clean up temp file
    print(f"{path.stem}: {kept:,}/{total:,} kept after full dedup")
    
def dedup():
    for path in sorted(CLEAN_DIR.glob("*.jsonl")):
        run_minhash = path.stem not in SKIP_MINHASH
        dedup_file(path, run_minhash)


def main():
    dedup()


if __name__ == "__main__":
    main()
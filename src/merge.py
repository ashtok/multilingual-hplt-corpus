"""Adaptive quota merge into a single interleaved JSONL, then shuffle."""
import json
import random
from pathlib import Path
import yaml
from tqdm import tqdm

CFG        = yaml.safe_load(open("configs/corpus.yaml", encoding="utf-8"))
DEDUP_DIR  = Path("data/deduped")
MERGED_DIR = Path("data/merged")
MERGED_DIR.mkdir(parents=True, exist_ok=True)
TOTAL      = CFG["total_lines"]

DROP_KEYS  = {"f", "o", "rs", "de", "xml", "html_lang", "ts", "cluster_size"}


def get_quality_score(obj: dict) -> float:
    ds = obj.get("doc_scores")
    if ds is None:
        return 0.0
    if isinstance(ds, dict):
        return float(ds.get("wds", ds.get("avg", next(iter(ds.values()), 0.0))))
    if isinstance(ds, list):
        if not ds:
            return 0.0
        first = ds[0]
        if isinstance(first, dict):
            return float(first.get("wds", first.get("avg", next(iter(first.values()), 0.0))))
        return float(first)
    try:
        return float(ds)
    except (TypeError, ValueError):
        return 0.0


def count_available(langs):
    counts = {}
    for lang in langs:
        path = DEDUP_DIR / f"{lang}.jsonl"
        counts[lang] = 0 if not path.exists() else sum(
            1 for _ in open(path, encoding="utf-8"))
    return counts


def compute_quotas(fracs, available, total):
    norm  = sum(fracs.values())
    f     = {k: v / norm for k, v in fracs.items()}
    quotas, capped, pending = {}, {}, set(f)
    while pending:
        budget = total - sum(capped.values())
        fsum   = sum(f[l] for l in pending)
        new_caps = {l: available[l] for l in pending
                    if int(budget * f[l] / fsum) >= available[l]}
        if new_caps:
            capped.update(new_caps)
            pending -= set(new_caps)
        else:
            quotas = {l: int(budget * f[l] / fsum) for l in pending}
            pending.clear()
    quotas.update(capped)
    diff = total - sum(quotas.values())
    if diff:
        headroom = {l: available[l] - quotas[l] for l in quotas
                    if available[l] - quotas[l] > 0}
        quotas[max(headroom or quotas, key=(headroom or quotas).get)] += diff
    return quotas


def shuffle_file(path: Path, seed: int = 42):
    print(f"\nShuffling {path.name}...")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    random.seed(seed)
    random.shuffle(lines)
    path.write_text("".join(lines), encoding="utf-8")
    print(f"✓ Shuffled {len(lines):,} lines")

def merge():
    langs     = list(CFG["languages"])
    fracs     = {l: CFG["languages"][l]["fraction"] for l in langs}
    available = count_available(langs)

    for l, n in available.items():
        if n == 0:
            print(f"  ⚠ {l}: 0 lines available — skipped")

    quotas = compute_quotas(fracs, available, TOTAL)

    print("\nQuotas:")
    for l, q in quotas.items():
        print(f"  {l:<12} available: {available[l]:>10,}  assigned: {q:>10,}  ({100*q/TOTAL:.1f}%)")
    print(f"  {'TOTAL':<12} {'':10}   assigned: {sum(quotas.values()):>10,}")

    files = {
        l: open(DEDUP_DIR / f"{l}.jsonl", encoding="utf-8")
        for l in langs if quotas.get(l, 0) > 0
    }
    remaining     = {l: quotas[l] for l in files}
    total_written = 0
    active        = list(files.keys())
    doc_counters  = {l: 0 for l in active}

    with open(MERGED_DIR / "train.jsonl", "w", encoding="utf-8") as fout:
        bar = tqdm(total=sum(remaining.values()), desc="Merging", unit_scale=True)
        idx = 0
        while total_written < TOTAL:
            if not any(r > 0 for r in remaining.values()):
                print(f"\n  ⚠ All sources exhausted — wrote {total_written:,}/{TOTAL:,}")
                break

            lang = active[idx % len(active)]
            idx += 1

            if remaining.get(lang, 0) <= 0:
                continue

            line = files[lang].readline()
            if not line:
                remaining[lang] = 0
                continue

            obj   = json.loads(line)
            bcp47 = CFG["languages"][lang]["bcp47"]

            obj["url"]        = obj.pop("u", "")
            obj["collection"] = obj.pop("c", "unknown")
            obj["score"]      = get_quality_score(obj)

            obj["lang"]       = bcp47
            obj["char_count"] = len(obj["text"])
            obj["word_count"] = len(obj["text"].split())
            doc_counters[lang] += 1
            obj["doc_id"]     = f"{bcp47}_{doc_counters[lang]:07d}"

            for k in DROP_KEYS:
                obj.pop(k, None)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            remaining[lang]  -= 1
            total_written    += 1
            bar.update(1)

        bar.close()

    for f in files.values():
        f.close()

    print(f"\n✓ {MERGED_DIR}/train.jsonl — {total_written:,} lines written")
    shuffle_file(MERGED_DIR / "train.jsonl")


def main():
    merge()


if __name__ == "__main__":
    main()

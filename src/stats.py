"""Compute corpus statistics for README and dataset card."""
import json
import collections
from pathlib import Path

MERGED_PATH = Path("data/merged/train.jsonl")
STATS_OUT   = Path("data/stats.json")


def bucket_score(score) -> str:
    if score is None: return "unknown"
    score = float(score)
    if score >= 10: return "10+"
    if score >= 8:  return "8–10"
    if score >= 6:  return "6–8"
    if score >= 4:  return "4–6"
    return                  "<4"


def flatten_field(val, default="unknown") -> str:
    """Safely convert any HPLT complex field (dict/list/str/None) to a string label."""
    if val is None:
        return default
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return max(val, key=val.get) if val else default
    if isinstance(val, list):
        if not val:
            return default
        first = val[0]
        if isinstance(first, dict):
            return max(first, key=first.get) if first else default
        return str(first)
    return str(val)


def main():
    lang_docs        = collections.Counter()
    lang_chars       = collections.Counter()
    lang_words       = collections.Counter()
    score_buckets    = collections.Counter()
    collections_ctr  = collections.Counter()
    web_registers    = collections.Counter()
    word_lengths     = []
    total            = 0

    print("Reading corpus...")
    with open(MERGED_PATH, encoding="utf-8") as f:
        for line in f:
            obj  = json.loads(line)
            lang = obj.get("lang", "??")
            text = obj.get("text", "")

            score      = float(obj.get("score", 0.0))
            collection = flatten_field(obj.get("collection"), "unknown")
            web_reg    = flatten_field(obj.get("web-register"), "unknown")

            wc = obj.get("word_count") or len(text.split())
            cc = obj.get("char_count") or len(text)

            lang_docs[lang]         += 1
            lang_chars[lang]        += cc
            lang_words[lang]        += wc
            score_buckets[bucket_score(score)] += 1
            collections_ctr[collection]        += 1
            web_registers[web_reg]             += 1
            word_lengths.append(wc)
            total += 1

    total_chars = sum(lang_chars.values())
    total_words = sum(lang_words.values())
    est_tokens  = total_chars // 4
    sorted_wl   = sorted(word_lengths)
    median_wl   = sorted_wl[len(sorted_wl) // 2]
    mean_wl     = total_words / total if total else 0

    print(f"\n{'='*60}")
    print(f"  CORPUS STATISTICS")
    print(f"{'='*60}")
    print(f"  Total documents   : {total:>12,}")
    print(f"  Total characters  : {total_chars:>12,}")
    print(f"  Total words       : {total_words:>12,}")
    print(f"  Estimated tokens  : {est_tokens:>12,}  (~chars÷4)")
    print(f"  Avg doc length    : {mean_wl:>11.0f}  words")
    print(f"  Median doc length : {median_wl:>12,}  words")
    print(f"  Min doc length    : {sorted_wl[0]:>12,}  words")
    print(f"  Max doc length    : {sorted_wl[-1]:>12,}  words")

    print(f"\n  {'─'*60}")
    print(f"  {'LANGUAGE':<8} {'DOCS':>10} {'%':>6} {'EST. TOKENS':>14} {'AVG WORDS':>10}")
    print(f"  {'─'*60}")
    for lang, n in sorted(lang_docs.items(), key=lambda x: -x[1]):
        pct    = 100 * n / total
        tokens = lang_chars[lang] // 4
        avg_w  = lang_words[lang] / n
        print(f"  {lang:<8} {n:>10,} {pct:>5.1f}% {tokens:>14,} {avg_w:>10.0f}")
    print(f"  {'─'*60}")
    print(f"  {'TOTAL':<8} {total:>10,} {'100.0':>6}% {est_tokens:>14,}")

    print(f"\n  QUALITY SCORE DISTRIBUTION (raw WDS, higher = better)")
    print(f"  {'─'*40}")
    for bucket in ["10+", "8–10", "6–8", "4–6", "<4"]:
        n   = score_buckets.get(bucket, 0)
        pct = 100 * n / total if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {bucket}  {n:>8,}  {pct:>5.1f}%  {bar}")

    print(f"\n  TOP SOURCE COLLECTIONS")
    print(f"  {'─'*40}")
    for col, n in collections_ctr.most_common(10):
        print(f"  {col:<30} {n:>8,}")

    print(f"\n  WEB REGISTER (document type)")
    print(f"  {'─'*40}")
    for reg, n in web_registers.most_common(10):
        pct = 100 * n / total
        print(f"  {reg:<30} {n:>8,}  ({pct:.1f}%)")

    stats = {
        "total_documents":         total,
        "total_characters":        total_chars,
        "total_words":             total_words,
        "estimated_tokens":        est_tokens,
        "avg_doc_length_words":    round(mean_wl, 1),
        "median_doc_length_words": median_wl,
        "min_doc_length_words":    sorted_wl[0],
        "max_doc_length_words":    sorted_wl[-1],
        "languages": {
            lang: {
                "documents":        lang_docs[lang],
                "pct":              round(100 * lang_docs[lang] / total, 2),
                "estimated_tokens": lang_chars[lang] // 4,
                "avg_words":        round(lang_words[lang] / lang_docs[lang], 1),
            }
            for lang in sorted(lang_docs, key=lambda x: -lang_docs[x])
        },
        "quality_score_distribution_raw_wds": dict(score_buckets),
        "top_collections":            dict(collections_ctr.most_common(10)),
        "web_registers":              dict(web_registers.most_common(10)),
    }

    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
    STATS_OUT.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n✓ Stats saved to {STATS_OUT}")


if __name__ == "__main__":
    main()

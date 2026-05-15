"""Rule-based quality filtering — each language is capped at its clean target."""
import json
import math
from pathlib import Path
import yaml
from tqdm import tqdm

CFG_ALL   = yaml.safe_load(open("configs/corpus.yaml", encoding="utf-8"))
CFG       = CFG_ALL["quality"]
RAW_DIR   = Path("data/raw")
CLEAN_DIR = Path("data/cleaned")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def per_language_clean_targets(cfg):
    total = cfg["total_lines"]
    langs = cfg["languages"]
    fracs = {k: v["fraction"] for k, v in langs.items()}
    fsum  = sum(fracs.values())

    targets = {lang: int(total * fracs[lang] / fsum) for lang in langs}
    diff    = total - sum(targets.values())
    if diff:
        targets[max(targets, key=targets.get)] += diff
    return targets


CLEAN_TARGETS = per_language_clean_targets(CFG_ALL)


def is_quality(text: str) -> bool:
    if not (CFG["min_chars"] <= len(text) <= CFG["max_chars"]):
        return False
    words = text.split()
    if not words:
        return False
    if sum(len(w) for w in words) / len(words) < CFG["min_avg_word_len"]:
        return False
    non_alpha = sum(1 for c in text if not c.isalpha() and not c.isspace())
    if non_alpha / len(text) > CFG["max_non_alpha_ratio"]:
        return False
    return True

def clean():
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        lang   = path.stem
        target = CLEAN_TARGETS.get(lang)
        out    = CLEAN_DIR / path.name

        # ── Skip if already cleaned and at target ──────────────────────────
        if out.exists():
            actual = sum(1 for _ in open(out, encoding="utf-8"))
            if target is None or actual >= target:
                print(f"  ✓ {lang} already cleaned ({actual:,} lines), skipping")
                continue
            print(f"  ↺ {lang} cleaned file short ({actual:,}/{target:,}), re-cleaning")
            out.unlink()
        # ───────────────────────────────────────────────────────────────────

        kept = total = 0
        with open(path, encoding="utf-8") as fin, \
             open(out, "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc=f"Cleaning {lang}", leave=False):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if is_quality(obj.get("text", "")):
                    fout.write(line + "\n")
                    kept += 1
                    if target and kept >= target:
                        break

        pct = f"{100*kept/total:.1f}%" if total > 0 else "N/A"
        cap = f"/{target:,}" if target else ""
        print(f"{lang}: {kept:,}{cap} kept from {total:,} raw ({pct})")
        
def main():
    clean()


if __name__ == "__main__":
    main()

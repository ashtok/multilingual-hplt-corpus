"""Download raw HPLT v3 shards per language — stops at per-language raw target."""
import io
import json
import math
import requests
import zstandard as zstd
from pathlib import Path
import yaml
from tqdm import tqdm

CFG    = yaml.safe_load(open("configs/corpus.yaml", encoding="utf-8"))
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def per_language_targets(cfg):
    """Compute clean_target and raw_target (= clean * buffer) per language."""
    total  = cfg["total_lines"]
    langs  = cfg["languages"]
    fracs  = {k: v["fraction"] for k, v in langs.items()}
    fsum   = sum(fracs.values())
    buf    = cfg.get("download_buffer", 1.25)

    clean  = {lang: int(total * fracs[lang] / fsum) for lang in langs}
    # distribute any rounding remainder to the largest language
    diff   = total - sum(clean.values())
    if diff:
        clean[max(clean, key=clean.get)] += diff

    raw = {lang: math.ceil(clean[lang] * buf) for lang in langs}
    return clean, raw


CLEAN_TARGETS, RAW_TARGETS = per_language_targets(CFG)


def get_quality_score(obj: dict) -> float:
    """Return HPLT v3 WDS score from the 's' field (raw integer, higher = better).
    Falls back to doc_scores[0] if 's' is absent. Returns None if unavailable."""
    s = obj.get("s")
    if s is not None:
        try:
            return float(s)
        except (TypeError, ValueError):
            pass
    ds = obj.get("doc_scores")
    if not ds:
        return None
    first = ds[0]
    if isinstance(first, dict):
        v = first.get("wds", first.get("avg", next(iter(first.values()), None)))
    else:
        v = first
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def fetch_map(lang: str) -> list[str]:
    url = f"{CFG['source']}/{lang}.map"
    r   = requests.get(url, timeout=30)
    r.raise_for_status()
    return [l.strip() for l in r.text.splitlines() if l.strip()]


def stream_shard(url: str):
    dctx = zstd.ZstdDecompressor()
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with dctx.stream_reader(resp.raw, read_across_frames=True) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text_stream:
                yield line


def download():
    for lang in tqdm(CFG["languages"], desc="Languages"):
        raw_target   = RAW_TARGETS[lang]
        clean_target = CLEAN_TARGETS[lang]
        out = RAW_DIR / f"{lang}.jsonl"

        if out.exists():
            tqdm.write(f"  ✓ {lang} exists, skipping")
            continue

        shards = fetch_map(lang)
        tqdm.write(f"  {lang}: {len(shards)} shards | "
                   f"clean_target={clean_target:,} raw_target={raw_target:,}")

        collected = 0
        with open(out, "w", encoding="utf-8") as f:
            bar = tqdm(total=raw_target, desc=f"  {lang}", unit="lines",
                       unit_scale=True, leave=False)
            for shard_url in shards:
                tqdm.write(f"    fetching: {shard_url}")
                try:
                    for line in stream_shard(shard_url):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        f.write(line + "\n")
                        collected += 1
                        bar.update(1)
                        if collected >= raw_target:
                            break
                except Exception as e:
                    tqdm.write(f"    ⚠ shard error: {e}")
                    continue
                if collected >= raw_target:
                    break
            bar.close()

        tqdm.write(f"  ✓ {lang}: {collected:,}/{raw_target:,} raw lines saved")


def main():
    download()


if __name__ == "__main__":
    main()

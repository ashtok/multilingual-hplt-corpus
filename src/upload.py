"""Convert JSONL to Parquet shards and push to Hugging Face Hub."""
import json
import os
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from huggingface_hub import HfApi

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID     = "AM0908/indic-hplt-v1" # Hugging Face repo ID (username/repo-name)
MERGED      = Path("data/merged/train.jsonl")
PARQUET_DIR = Path("data/parquet")
SHARD_SIZE  = 100_000   # rows per shard → ~99 shards for 9.8M rows

KEEP = {
    "text", "lang", "url", "score", "collection",
    "char_count", "word_count", "doc_id",
    "web-register", "prob",
}


# ── Step 1: JSONL → Parquet shards (streaming, low RAM) ──────────────────────
def jsonl_to_parquet():
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(PARQUET_DIR.glob("train-*.parquet"))
    if existing:
        print(f"  ✓ {len(existing)} Parquet shards already exist, skipping conversion")
        return existing

    print(f"Converting {MERGED} → Parquet shards ({SHARD_SIZE:,} rows each)...")
    buffer, shard_idx, total = [], 0, 0
    paths = []

    def flush(buf, idx):
        table = pa.Table.from_pylist(buf)
        out   = PARQUET_DIR / f"train-{idx:05d}-of-XXXXX.parquet"
        pq.write_table(
            table, out,
            compression="snappy",
            row_group_size=5_000,   # small row groups = HF viewer can page without loading full shard
        )
        return out

    with open(MERGED, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = {k: obj.get(k) for k in KEEP if k in obj}
            buffer.append(row)
            total += 1
            if len(buffer) >= SHARD_SIZE:
                paths.append(flush(buffer, shard_idx))
                print(f"  wrote shard {shard_idx:05d}  ({total:,} rows total)")
                buffer  = []
                shard_idx += 1

    if buffer:
        paths.append(flush(buffer, shard_idx))
        shard_idx += 1

    # Rename placeholder XXXXX → actual total
    final = []
    for p in paths:
        new = p.parent / p.name.replace("XXXXX", f"{shard_idx:05d}")
        p.rename(new)
        final.append(new)

    print(f"\n✓ {total:,} rows → {shard_idx} shards in {PARQUET_DIR}/")
    return sorted(final)


# ── Step 2: Split shards 98 / 1 / 1 ─────────────────────────────────────────
def split_shards(paths):
    n   = len(paths)
    v   = max(1, int(n * 0.98))
    t   = max(v + 1, int(n * 0.99))
    return {
        "train":      paths[:v],
        "validation": paths[v:t],
        "test":       paths[t:],
    }


# ── Step 3: Upload ────────────────────────────────────────────────────────────
def upload(splits):
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, private=False)
    print(f"Repo: https://huggingface.co/datasets/{REPO_ID}\n")

    for split_name, paths in splits.items():
        print(f"Uploading {split_name}: {len(paths)} shards...")
        for p in paths:
            dest = f"data/{split_name}/{p.name}"
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=dest,
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Upload {split_name}/{p.name}",
            )
            print(f"  ✓ {p.name}")

    print(f"\n✅ Done: https://huggingface.co/datasets/{REPO_ID}")


def main():
    parquet_paths = jsonl_to_parquet()
    splits        = split_shards(parquet_paths)

    print("\nSplit summary:")
    for s, p in splits.items():
        print(f"  {s:<12} {len(p):>3} shards")
    print()

    upload(splits)


if __name__ == "__main__":
    main()

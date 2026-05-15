"""
indic-hplt-corpus — full pipeline runner.

Usage:
    uv run python main.py                        # run all steps
    uv run python main.py --steps download clean # run specific steps
    uv run python main.py --skip upload          # skip a step
    uv run python main.py --lines 1000           # override total_lines (quick test)
"""
import argparse
import importlib
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime

STEPS = ["download", "clean", "dedup", "merge", "stats"]


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def banner(msg, char="="):
    width = 60
    print(f"\n{'=' * width}")
    print(f"=== {ts()} | {msg}")
    print(f"{'=' * width}")


def run_step(name: str):
    banner(f"Starting: {name}")
    t0 = time.time()
    try:
        mod = importlib.import_module(f"src.{name}")
        # Each module exposes a main() function
        mod.main()
        elapsed = time.time() - t0
        banner(f"Done: {name}  [{elapsed/60:.1f} min]")
        return True
    except Exception as e:
        banner(f"FAILED: {name}")
        print(f"  Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the indic-hplt corpus pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # all steps
  python main.py --steps download clean   # only download + clean
  python main.py --skip upload            # all steps except upload
  python main.py --lines 1000             # test run with 1K lines
  python main.py --steps upload           # re-upload only
        """,
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEPS,
        metavar="STEP",
        help=f"Steps to run (default: all). Choices: {', '.join(STEPS)}",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=STEPS,
        metavar="STEP",
        default=[],
        help="Steps to skip.",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=None,
        metavar="N",
        help="Override total_lines in configs/corpus.yaml (useful for test runs).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/corpus.yaml",
        metavar="PATH",
        help="Path to corpus config (default: configs/corpus.yaml).",
    )
    return parser.parse_args()


def patch_config(config_path: str, lines: int):
    """Temporarily override total_lines in the config for this run."""
    cfg_path = Path(config_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    original = cfg["total_lines"]
    cfg["total_lines"] = lines
    cfg_path.write_text(
        yaml.dump(cfg, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"  ⚙  total_lines overridden: {original:,} → {lines:,}")
    return original


def restore_config(config_path: str, original_lines: int):
    cfg_path = Path(config_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["total_lines"] = original_lines
    cfg_path.write_text(
        yaml.dump(cfg, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"  ⚙  total_lines restored to {original_lines:,}")


def main():
    args = parse_args()

    steps_to_run = args.steps if args.steps else STEPS
    steps_to_run = [s for s in steps_to_run if s not in (args.skip or [])]

    if not steps_to_run:
        print("No steps to run after applying --skip. Exiting.")
        sys.exit(0)

    print(f"\n{'=' * 60}")
    print(f"  indic-hplt-corpus pipeline")
    print(f"  Config : {args.config}")
    print(f"  Steps  : {' → '.join(steps_to_run)}")
    if args.lines:
        print(f"  Lines  : {args.lines:,} (test override)")
    print(f"{'=' * 60}")

    # Override config if --lines passed
    original_lines = None
    if args.lines:
        original_lines = patch_config(args.config, args.lines)

    pipeline_start = time.time()
    results = {}

    try:
        for step in steps_to_run:
            ok = run_step(step)
            results[step] = ok
            if not ok:
                print(f"\n  ✗ Step '{step}' failed — stopping pipeline.", file=sys.stderr)
                break
    finally:
        # Always restore config even if pipeline crashes
        if args.lines and original_lines is not None:
            restore_config(args.config, original_lines)

    # Summary
    total_elapsed = time.time() - pipeline_start
    banner("Pipeline Summary")
    for step in steps_to_run:
        status = "✓" if results.get(step) else ("✗" if step in results else "–")
        print(f"  {status}  {step}")

    skipped = [s for s in STEPS if s not in steps_to_run]
    if skipped:
        for s in skipped:
            print(f"  –  {s}  (skipped)")

    print(f"\n  Total time: {total_elapsed/60:.1f} min")

    if all(results.values()):
        print(f"\n  ✅ All steps completed successfully.")
        print(f"  Output : data/merged/train.jsonl")
        print(f"  Stats  : data/stats.json")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

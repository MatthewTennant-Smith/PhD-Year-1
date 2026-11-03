#!/usr/bin/env python3
"""
Extract just the 'test' partition from the raw hh-rlhf clone and write it
out as readable, indented JSON (one file per subset, plus a combined file).
"""

import argparse
import gzip
import json
from pathlib import Path

SUBSETS = [
    "harmless-base",
    "helpful-base",
    "helpful-online",
    "helpful-rejection-sampled",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to raw hh-rlhf clone")
    parser.add_argument("--out", required=True, help="Output directory for readable JSON")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only write the first N rows per subset (for a quick peek)",
    )
    args = parser.parse_args()

    src = Path(args.src).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for subset in SUBSETS:
        fpath = src / subset / "test.jsonl.gz"

        if not fpath.exists():
            print(f"  [skip] {fpath} not found")
            continue

        print(f"  [read] {fpath}")

        rows = []
        with gzip.open(fpath, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.limit is not None and i >= args.limit:
                    break
                row = json.loads(line)
                row["_subset"] = subset
                rows.append(row)

        out_path = out / f"{subset}_test.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        print(f"    -> wrote {len(rows)} rows to {out_path}")
        all_rows.extend(rows)

    combined_path = out / "all_test_combined.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    print(f"\nCombined: {len(all_rows)} rows -> {combined_path}")


if __name__ == "__main__":
    main()

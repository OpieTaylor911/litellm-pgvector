#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample extracted metadata JSON files for manual QA")
    parser.add_argument("json_dir", help="Directory containing extracted metadata .json files")
    parser.add_argument("--count", type=int, default=10, help="How many files to sample")
    args = parser.parse_args()

    root = Path(args.json_dir)
    files = sorted(p for p in root.glob("*.json") if p.is_file())
    if not files:
        print("No JSON files found.")
        return 1

    sample = random.sample(files, min(args.count, len(files)))
    for path in sample:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"=== {path.name} ===")
        print(json.dumps({
            "genres": data.get("genres"),
            "tropes": data.get("tropes"),
            "military": data.get("military"),
            "kinks": data.get("kinks"),
            "heat_level": data.get("heat_level"),
            "tone": data.get("tone"),
            "explicitness": data.get("explicitness"),
            "ending": data.get("ending"),
            "search_keywords": data.get("search_keywords"),
        }, indent=2))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

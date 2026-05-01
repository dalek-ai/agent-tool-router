"""Aggregate all loaders into a single normalized JSONL.

Output: data/traces.jsonl  (one Trace per line)

Run: python3 router/index/aggregate.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Path-relative imports because all loaders live in router/index.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from trace_schema import Trace  # noqa: E402

LOADERS = [
    "load_swebench",
    "load_taubench",
    "load_osworld",
    "load_hermes",
    "load_toolace",
]
OUT = ROOT / "data" / "traces.jsonl"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    by_source: Counter[str] = Counter()
    total = 0
    with OUT.open("w", encoding="utf-8") as f:
        for mod_name in LOADERS:
            try:
                mod = __import__(mod_name)
            except ImportError as exc:
                print(f"[warn] cannot import {mod_name}: {exc}", file=sys.stderr)
                continue
            try:
                for trace in mod.iter_traces():
                    f.write(json.dumps(trace.to_jsonl(), ensure_ascii=False) + "\n")
                    by_source[trace.source_dataset] += 1
                    total += 1
            except FileNotFoundError as exc:
                print(f"[skip] {mod_name}: {exc}", file=sys.stderr)
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {mod_name} raised: {exc}", file=sys.stderr)
                continue
    print(f"[done] {total} traces -> {OUT}")
    for src, n in by_source.most_common():
        print(f"  {n:>6}  {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

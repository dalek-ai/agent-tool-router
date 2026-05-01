"""Loader for SWE-bench Verified.

Dataset: princeton-nlp/SWE-bench_Verified on HuggingFace (~500 tasks).
Each task has a problem_statement, gold patch, and test script. Agent traces
themselves are not in the dataset — we treat each task as a "task to be routed"
and use the gold patch's touched files as a weak label for which tools the
agent should call.

To download:
    pip3 install datasets
    python3 router/index/load_swebench.py --download

To iterate after download:
    python3 router/index/load_swebench.py | head
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

from trace_schema import Trace, ToolCall

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "swebench_verified.jsonl"


def download() -> None:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[error] pip3 install datasets first", file=sys.stderr)
        raise SystemExit(2)
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with CACHE.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"[done] cached {n} rows to {CACHE}")


def iter_traces() -> Iterator[Trace]:
    if not CACHE.exists():
        raise FileNotFoundError(
            f"{CACHE} missing — run with --download first"
        )
    with CACHE.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            instance_id = row.get("instance_id", "")
            problem = row.get("problem_statement", "")
            patch = row.get("patch", "") or ""
            # Weak label: extract files touched by the patch as the "tools sequence".
            # Each touched file becomes one synthetic ToolCall(name="edit_file", args={"path": <p>}).
            files = []
            for line2 in patch.splitlines():
                if line2.startswith("diff --git "):
                    parts = line2.split()
                    if len(parts) >= 4:
                        files.append(parts[2].removeprefix("a/"))
            tools = [ToolCall(name="edit_file", args={"path": p}) for p in files]
            yield Trace(
                trace_id=f"swebench-verified/{instance_id}",
                source_dataset="swe-bench-verified",
                task_text=problem,
                tools_called=tools,
                outcome="success",  # gold patches are by definition successful
                metadata={"repo": row.get("repo"), "base_commit": row.get("base_commit")},
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    p.add_argument("--count", action="store_true")
    args = p.parse_args()
    if args.download:
        download()
        return 0
    if args.count:
        n = sum(1 for _ in iter_traces())
        print(f"swe-bench-verified: {n} traces")
        return 0
    for trace in iter_traces():
        print(json.dumps(trace.to_jsonl(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

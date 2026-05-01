"""Loader for OSWorld.

Source: https://github.com/xlangai/OSWorld
OSWorld is a benchmark of real OS-level GUI tasks (Linux/Windows). Trajectories
are screenshots + actions (click, type, scroll, etc.). Useful for the "browser /
desktop" half of an Agent Tool Router's coverage.

Heads up — OSWorld is heavy (~8 GB with screenshots). Skip the screenshots and
load only the action trajectories for our purposes.

To clone (lightweight, no LFS pull of screenshots):
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/xlangai/OSWorld.git data/osworld

Each example lives at evaluation_examples/examples/<app>/<task_id>.json with
fields: id, instruction, expected, etc. Trajectories themselves are in trial
output dirs (results/) once you actually run the benchmark — without running
it, we only have *task definitions*, not realized agent traces.

We therefore emit one Trace per example with empty tools_called and outcome
"unknown" — useful as a *task corpus* for the routing classifier even without
realized trajectories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

from trace_schema import Trace, ToolCall

ROOT = Path(__file__).resolve().parents[2]
OSW_DIR = ROOT / "data" / "osworld"


def iter_traces() -> Iterator[Trace]:
    if not OSW_DIR.exists():
        raise FileNotFoundError(
            f"{OSW_DIR} missing — clone OSWorld first:\n"
            f"  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/xlangai/OSWorld.git {OSW_DIR}"
        )
    examples_root = OSW_DIR / "evaluation_examples" / "examples"
    if not examples_root.exists():
        # Fallback for newer layouts.
        examples_root = OSW_DIR / "evaluation_examples"
    for path in examples_root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        task_id = data.get("id") or path.stem
        instruction = data.get("instruction") or ""
        app = path.parent.name
        yield Trace(
            trace_id=f"osworld/{app}/{task_id}",
            source_dataset="osworld",
            task_text=instruction,
            tools_called=[],
            outcome="unknown",
            metadata={"app": app, "file": str(path.relative_to(ROOT))},
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--count", action="store_true")
    args = p.parse_args()
    if args.count:
        n = sum(1 for _ in iter_traces())
        print(f"osworld: {n} traces")
        return 0
    for trace in iter_traces():
        print(json.dumps(trace.to_jsonl(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

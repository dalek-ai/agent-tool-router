"""Loader for tau-bench (sierra-research/tau-bench).

We use the `historical_trajectories/` directory which ships with the repo —
these are realized rollouts of GPT-4o and Sonnet-3.5 on retail+airline domains.

Each file is a JSON list of entries shaped like:
    {
      "task_id": int,
      "reward": float,        # >0 if the task was completed correctly
      "info": {
        "task": {
          "user_id": "...",
          "actions": [{"name": "...", "kwargs": {...}}, ...],   # GOLD tool sequence
          "instruction": "..."                                  # NL task description
        }
      },
      "traj": [...],         # actual conversation (we ignore for now)
      "trial": int
    }

Each file's name encodes (model, domain): e.g. "sonnet-35-new-retail.json".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

from trace_schema import Trace, ToolCall

ROOT = Path(__file__).resolve().parents[2]
TAU_DIR = ROOT / "data" / "tau-bench"

_FILENAME_RE = re.compile(r"^(?P<model>.+?)-(?P<domain>retail|airline)\.json$")


def _parse_filename(stem: str) -> tuple[str, str]:
    """Return (model, domain) from a historical_trajectories filename."""
    m = _FILENAME_RE.match(stem + ".json")
    if not m:
        return ("unknown", "unknown")
    return m.group("model"), m.group("domain")


def iter_traces() -> Iterator[Trace]:
    if not TAU_DIR.exists():
        raise FileNotFoundError(
            f"{TAU_DIR} missing — clone tau-bench first."
        )
    hist = TAU_DIR / "historical_trajectories"
    if not hist.exists():
        raise FileNotFoundError(
            f"{hist} missing — repo layout may have changed; check tau-bench/"
        )
    for path in sorted(hist.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {path.name}: {exc}", file=sys.stderr)
            continue
        if not isinstance(data, list):
            continue
        model, domain = _parse_filename(path.name)
        for item in data:
            if not isinstance(item, dict):
                continue
            task_info = (item.get("info") or {}).get("task") or {}
            instruction = task_info.get("instruction") or ""
            gold_actions = task_info.get("actions") or []
            tools: list[ToolCall] = []
            for action in gold_actions:
                if not isinstance(action, dict):
                    continue
                tools.append(
                    ToolCall(
                        name=action.get("name") or "unknown",
                        args=action.get("kwargs") or {},
                        success=True,
                    )
                )
            reward = item.get("reward") or 0
            outcome = "success" if reward and reward > 0 else "failure"
            task_id = item.get("task_id", "?")
            trial = item.get("trial", 0)
            yield Trace(
                trace_id=f"tau-bench/{model}/{domain}/{task_id}/trial-{trial}",
                source_dataset="tau-bench",
                task_text=instruction,
                tools_called=tools,
                outcome=outcome,
                metadata={
                    "model": model,
                    "domain": domain,
                    "reward": reward,
                    "trial": trial,
                },
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--count", action="store_true", help="just count traces")
    args = p.parse_args()
    if args.count:
        n = sum(1 for _ in iter_traces())
        print(f"tau-bench: {n} traces")
        return 0
    for trace in iter_traces():
        print(json.dumps(trace.to_jsonl(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

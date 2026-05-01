"""Loader for NousResearch/hermes-function-calling-v1.

Each row ships:
  - tools: JSON string, list of OpenAI-style function specs
  - conversations: [{from: "system"|"human"|"gpt", value: str}, ...]

The "human" turn is the task text. The "gpt" turn embeds one or more
<tool_call>{...json...}</tool_call> blocks — the gold tool sequence.

Two configs we use:
  - func_calling_singleturn  (1893 rows)
  - func_calling             (1893 rows; superset / same content in this snapshot)

We dedupe by (config, id) and skip rows whose gpt turn has zero tool_calls
(those are pure-text replies, not useful for routing).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Iterator

from trace_schema import Trace, ToolCall

CONFIGS = ("func_calling_singleturn", "func_calling")
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _extract_tool_calls(gpt_value: str) -> list[ToolCall]:
    out: list[ToolCall] = []
    for raw in _TOOL_CALL_RE.findall(gpt_value or ""):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = parsed.get("arguments") or {}
        if not isinstance(args, dict):
            args = {"_raw": args}
        out.append(ToolCall(name=name, args=args, success=True))
    return out


def iter_traces() -> Iterator[Trace]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise FileNotFoundError(
            "datasets package not installed (pip install datasets)"
        ) from exc

    seen: set[tuple[str, str]] = set()
    for cfg in CONFIGS:
        try:
            ds = load_dataset(
                "NousResearch/hermes-function-calling-v1", cfg, split="train"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] hermes/{cfg}: {exc}", file=sys.stderr)
            continue

        for item in ds:
            row_id = str(item.get("id") or "")
            key = (cfg, row_id)
            if row_id and key in seen:
                continue
            seen.add(key)

            convs = item.get("conversations") or []
            human_text = ""
            gpt_text = ""
            for c in convs:
                if not isinstance(c, dict):
                    continue
                role = c.get("from")
                val = c.get("value") or ""
                if role == "human" and not human_text:
                    human_text = val
                elif role == "gpt" and not gpt_text:
                    gpt_text = val

            if not human_text or not gpt_text:
                continue
            tools = _extract_tool_calls(gpt_text)
            if not tools:
                continue

            yield Trace(
                trace_id=f"hermes/{cfg}/{row_id or len(seen)}",
                source_dataset="hermes-function-calling-v1",
                task_text=human_text,
                tools_called=tools,
                outcome="success",  # gold trajectories
                metadata={
                    "config": cfg,
                    "category": item.get("category"),
                    "subcategory": item.get("subcategory"),
                    "task": item.get("task"),
                },
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--count", action="store_true")
    args = p.parse_args()
    if args.count:
        n = sum(1 for _ in iter_traces())
        print(f"hermes: {n} traces")
        return 0
    for trace in iter_traces():
        print(json.dumps(trace.to_jsonl(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

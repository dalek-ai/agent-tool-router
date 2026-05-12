"""Extract (query, history, next_tool) triplets from multi-turn traces.

Iterates traces.jsonl. For each trace with >= 2 successful tool calls,
emits triplets at each position t > 0:
  (task_text, history=[tool_0..tool_{t-1}], next_tool=tool_t).

Triplets where next_tool.name is not present in the tool catalog
(tool_descriptions.jsonl) are skipped. Consecutive duplicate calls
(same tool name twice in a row) are collapsed before emission, so
tau-bench traces that hammer the same tool 5x produce useful
transitions rather than self-loops.

Output: data/next_tool_triplets.jsonl
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "data" / "traces.jsonl"
CATALOG = ROOT / "data" / "tool_descriptions.jsonl"
OUT = ROOT / "data" / "next_tool_triplets.jsonl"


def normalize(name: str) -> str:
    return (name or "").strip().lower()


def load_catalog_names() -> set[str]:
    names = set()
    with CATALOG.open() as f:
        for line in f:
            d = json.loads(line)
            n = d.get("name")
            if n:
                names.add(normalize(n))
    return names


def collapse_consecutive(tools: list[dict]) -> list[dict]:
    out = []
    last = None
    for t in tools:
        n = normalize(t.get("name", ""))
        if n and n != last:
            out.append(t)
            last = n
    return out


def main() -> None:
    catalog = load_catalog_names()
    print(f"Catalog: {len(catalog)} unique tool names")

    by_source = Counter()
    skipped_no_catalog = Counter()
    emitted = 0

    with TRACES.open() as f, OUT.open("w") as out:
        for line in f:
            d = json.loads(line)
            src = d.get("source_dataset", "?")
            tools = d.get("tools_called") or []
            tools = [t for t in tools if t.get("success") is not False]
            tools = collapse_consecutive(tools)
            if len(tools) < 2:
                continue

            task = d.get("task_text") or ""
            history = []
            for t in tools:
                name = normalize(t.get("name", ""))
                if not name:
                    history.append(None)
                    continue
                if history and any(h is not None for h in history):
                    if name not in catalog:
                        skipped_no_catalog[src] += 1
                    else:
                        rec = {
                            "task_text": task,
                            "history": [h for h in history if h],
                            "next_tool": t.get("name"),
                            "next_tool_norm": name,
                            "source": src,
                            "trace_id": d.get("trace_id"),
                            "position": len(history),
                        }
                        out.write(json.dumps(rec) + "\n")
                        emitted += 1
                        by_source[src] += 1
                history.append(name)

    print(f"Emitted {emitted} triplets")
    for s, n in by_source.most_common():
        print(f"  {s}: {n}  (skipped no-catalog: {skipped_no_catalog[s]})")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()

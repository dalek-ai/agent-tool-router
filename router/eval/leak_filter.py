"""Leakage filter: identify traces where the gold tool name(s) appear inside
the task text, either verbatim or as in-order subtokens within a small window.

Why this matters: a router that "predicts" a tool when the task literally
says "use get_weather to ..." is doing string matching, not learning. We
want to publish a cross-corpus number that survives the removal of those
rows, so that nobody can wave it away on HN.

Two modes:
  - verbatim: the tool name appears as a substring of the task (case-insensitive).
  - loose: the underscore/camelCase subtokens of the tool name appear in
    order within a 4-token window. Catches paraphrases like
    "use get weather" matching `get_weather`.

Default mode is "loose" — strictly tighter than "verbatim", so any number
produced under loose-filter is also defensible under verbatim-filter.

Usage as a library:
    from router.eval.leak_filter import is_leaky
    if is_leaky(task_text, tool_names, mode="loose"):
        skip
"""

from __future__ import annotations

import re

_SPLIT_RE = re.compile(r"[_\.\s\-]+")
_CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+")
_WORD_RE = re.compile(r"[a-z0-9]+")


def split_tokens(name: str) -> list[str]:
    """Split a tool name into lowercase subtokens length >= 2."""
    out: list[str] = []
    for part in _SPLIT_RE.split(name):
        out.extend(_CAMEL_RE.findall(part))
    return [t.lower() for t in out if len(t) >= 2]


def is_leaky(task_text: str, tool_names: list[str], *, mode: str = "loose",
             max_gap: int = 4) -> bool:
    """Return True if any tool_name leaks into the task_text.

    mode="verbatim": substring check.
    mode="loose":   substring OR subtokens in order within max_gap words.
    """
    task = (task_text or "").lower()
    if not task or not tool_names:
        return False
    # Verbatim first (cheap).
    for n in tool_names:
        if n and n.lower() in task:
            return True
    if mode == "verbatim":
        return False
    # Loose: subtokens in order, bounded gap.
    task_toks = _WORD_RE.findall(task)
    if not task_toks:
        return False
    pos: dict[str, list[int]] = {}
    for i, t in enumerate(task_toks):
        pos.setdefault(t, []).append(i)
    for n in tool_names:
        if not n:
            continue
        toks = split_tokens(n)
        if len(toks) < 2:
            continue
        if not all(t in pos for t in toks):
            continue
        for start in pos[toks[0]]:
            cur = start
            ok = True
            for t in toks[1:]:
                cand = [p for p in pos[t] if cur < p <= cur + max_gap]
                if not cand:
                    ok = False
                    break
                cur = cand[0]
            if ok:
                return True
    return False


def main() -> int:
    import argparse
    import json
    from collections import Counter
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="data/traces.jsonl")
    parser.add_argument("--mode", choices=("verbatim", "loose"), default="loose")
    parser.add_argument("--max-gap", type=int, default=4)
    args = parser.parse_args()

    by_src: Counter[str] = Counter()
    leak_by_src: Counter[str] = Counter()
    with Path(args.traces).open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            src = t.get("source_dataset", "")
            tools = [tc.get("name") for tc in (t.get("tools_called") or [])
                     if tc.get("name")]
            task = t.get("task_text", "") or ""
            by_src[src] += 1
            if is_leaky(task, tools, mode=args.mode, max_gap=args.max_gap):
                leak_by_src[src] += 1

    print(f"mode={args.mode} max_gap={args.max_gap}")
    for s in sorted(by_src):
        n = by_src[s]
        l = leak_by_src[s]
        pct = 100 * l / n if n else 0.0
        print(f"  {s:30s} {n:5d} rows  leaky={l:5d} ({pct:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

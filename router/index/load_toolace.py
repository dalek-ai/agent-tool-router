"""Loader for Team-ACE/ToolACE.

Schema:
  - system: expert prompt that includes the available tool catalog
  - conversations: [{from: "user"|"assistant"|"tool", value: str}, ...]

Assistant tool-call turns look like:
  [FuncName(arg=val, ...), OtherFunc(...)]
Plain assistant text is also possible (post-tool synthesis).

For routing we only care about the *gold* tool sequence: every assistant
turn that begins with "[" is a parallel/sequential tool call. We flatten
across the conversation so that one ToolACE row → one Trace whose
task_text = first user turn, tools_called = concatenation of all assistant
tool-call turns (in order). Dialogues with zero tool calls are skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Iterator

from trace_schema import Trace, ToolCall

# Match a single FuncName(...) at the start of a balanced-paren expression.
_FUNC_HEAD_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_\.\s]*?)\s*\(")


def _split_calls(body: str) -> list[tuple[str, str]]:
    """Split "[FuncA(...), FuncB(...)]" body into [(name, args_str), ...]
    using a paren-balanced scan (the args may themselves contain commas)."""
    out: list[tuple[str, str]] = []
    i = 0
    n = len(body)
    while i < n:
        m = _FUNC_HEAD_RE.match(body, i)
        if not m:
            break
        name = m.group(1).strip()
        args_start = m.end()  # one past the opening "("
        depth = 1
        j = args_start
        in_str: str | None = None
        while j < n and depth > 0:
            ch = body[j]
            if in_str:
                if ch == "\\" and j + 1 < n:
                    j += 2
                    continue
                if ch == in_str:
                    in_str = None
            else:
                if ch in ('"', "'"):
                    in_str = ch
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        args_str = body[args_start:j]
        out.append((name, args_str))
        # advance past ")" and an optional ","
        i = j + 1
        while i < n and body[i] in ", \n\t":
            i += 1
    return out


def _parse_assistant_call(value: str) -> list[ToolCall]:
    s = (value or "").strip()
    if not s.startswith("["):
        return []
    if not s.endswith("]"):
        # Some rows have trailing whitespace/newlines after the bracket — be lenient
        # by trimming to the last "]".
        last = s.rfind("]")
        if last == -1:
            return []
        s = s[: last + 1]
    inner = s[1:-1].strip()
    if not inner:
        return []
    calls = _split_calls(inner)
    out: list[ToolCall] = []
    for name, args_str in calls:
        if not name:
            continue
        out.append(
            ToolCall(name=name, args={"_raw_args": args_str.strip()}, success=True)
        )
    return out


def iter_traces() -> Iterator[Trace]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise FileNotFoundError(
            "datasets package not installed (pip install datasets)"
        ) from exc

    try:
        ds = load_dataset("Team-ACE/ToolACE", split="train")
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(f"toolace load failed: {exc}") from exc

    for idx, item in enumerate(ds):
        convs = item.get("conversations") or []
        first_user = ""
        tools: list[ToolCall] = []
        for c in convs:
            if not isinstance(c, dict):
                continue
            role = c.get("from")
            val = c.get("value") or ""
            if role == "user" and not first_user:
                first_user = val
            elif role == "assistant":
                tools.extend(_parse_assistant_call(val))

        if not first_user or not tools:
            continue
        yield Trace(
            trace_id=f"toolace/{idx}",
            source_dataset="toolace",
            task_text=first_user,
            tools_called=tools,
            outcome="success",
            metadata={"row_index": idx},
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--count", action="store_true")
    args = p.parse_args()
    if args.count:
        n = sum(1 for _ in iter_traces())
        print(f"toolace: {n} traces")
        return 0
    for trace in iter_traces():
        print(json.dumps(trace.to_jsonl(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a unified table of tool descriptions: (source, tool_name) -> description.

Sources covered:
  - tau-bench: import tool modules from data/tau-bench/.../envs/{retail,airline}/tools/*.py
    and call .get_info()['function']['description'].
  - hermes: parse the `tools` field of each row (OpenAI-style spec list).
  - toolace: regex out the JSON list from the system prompt of each row.

Output: data/tool_descriptions.jsonl with one row per (source, name) pair.

Why this matters: cross-source LOSO scored 0% because tool *names* don't
overlap. But two ecosystems can have a "search_flights" and a
"FlightSearchAPI" that both have the description "search for flights between
two airports on a given date". If we score by description similarity, that
0% becomes non-trivial. This file is the prerequisite for the description-
retrieval baseline.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "tool_descriptions.jsonl"
TAU_ROOT = DATA / "tau-bench" / "tau_bench" / "envs"


def _load_tau_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"_tau.{path.parent.parent.name}.{path.stem}", str(path)
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def iter_taubench() -> Iterable[dict]:
    """Yield {source, name, description} for tau-bench retail+airline tools."""
    # tau_bench imports `from tau_bench.envs.tool import Tool`. Add the package
    # root to sys.path so importlib can resolve that.
    sys.path.insert(0, str(DATA / "tau-bench"))
    for env in ("retail", "airline"):
        env_dir = TAU_ROOT / env / "tools"
        if not env_dir.is_dir():
            continue
        for f in sorted(env_dir.glob("*.py")):
            if f.name.startswith("_"):
                continue
            mod = _load_tau_module(f)
            if mod is None:
                continue
            # find the Tool subclass
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if not isinstance(obj, type):
                    continue
                if not hasattr(obj, "get_info"):
                    continue
                try:
                    info = obj.get_info()
                except Exception:
                    continue
                fn = (info or {}).get("function") or {}
                name = fn.get("name")
                desc = fn.get("description") or ""
                if not name:
                    continue
                yield {
                    "source": "tau-bench",
                    "env": env,
                    "name": name,
                    "description": desc.strip(),
                }
                break


def iter_hermes() -> Iterable[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return
    seen: set[tuple[str, str]] = set()
    for cfg in ("func_calling_singleturn", "func_calling"):
        try:
            ds = load_dataset(
                "NousResearch/hermes-function-calling-v1", cfg, split="train"
            )
        except Exception as exc:
            print(f"[warn] hermes/{cfg}: {exc}", file=sys.stderr)
            continue
        for item in ds:
            tools_raw = item.get("tools")
            if isinstance(tools_raw, str):
                try:
                    tools = json.loads(tools_raw)
                except json.JSONDecodeError:
                    continue
            else:
                tools = tools_raw
            if not isinstance(tools, list):
                continue
            for spec in tools:
                if not isinstance(spec, dict):
                    continue
                fn = spec.get("function") or spec
                name = fn.get("name")
                desc = (fn.get("description") or "").strip()
                if not name:
                    continue
                key = ("hermes-function-calling-v1", name)
                if key in seen:
                    continue
                seen.add(key)
                yield {
                    "source": "hermes-function-calling-v1",
                    "name": name,
                    "description": desc,
                }


_TOOLACE_HEADER = "Here is a list of functions in JSON format that you can invoke:"


def _extract_balanced_list(s: str, start: int) -> str | None:
    """Extract the JSON list starting at index `start` (must be '['). Tracks
    bracket depth, ignoring brackets inside string literals."""
    if start >= len(s) or s[start] != "[":
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def iter_toolace() -> Iterable[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return
    try:
        ds = load_dataset("Team-ACE/ToolACE", split="train")
    except Exception as exc:
        print(f"[warn] toolace: {exc}", file=sys.stderr)
        return
    seen: set[tuple[str, str]] = set()
    for item in ds:
        sysmsg = item.get("system") or ""
        idx = sysmsg.find(_TOOLACE_HEADER)
        if idx < 0:
            continue
        # find first '[' after the header
        bstart = sysmsg.find("[", idx + len(_TOOLACE_HEADER))
        if bstart < 0:
            continue
        raw = _extract_balanced_list(sysmsg, bstart)
        if raw is None:
            continue
        try:
            tools = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(tools, list):
            continue
        for spec in tools:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            desc = (spec.get("description") or "").strip()
            if not name:
                continue
            key = ("toolace", name)
            if key in seen:
                continue
            seen.add(key)
            yield {"source": "toolace", "name": name, "description": desc}


def main() -> int:
    DATA.mkdir(exist_ok=True)
    n = 0
    by_src: dict[str, int] = {}
    with OUT.open("w", encoding="utf-8") as f:
        for it in (iter_taubench(), iter_hermes(), iter_toolace()):
            for row in it:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                by_src[row["source"]] = by_src.get(row["source"], 0) + 1
    print(f"wrote {n} tool descriptions to {OUT}")
    for s in sorted(by_src):
        print(f"  {s:30s} {by_src[s]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Five mock tools for the research-helper dogfood.

Each tool has:
  - a `name` (what the router predicts)
  - a `describe` string (what the agent shows the user before invocation)
  - a `run(arg)` function that returns a string (mock execution — no network,
    no file IO outside this folder, no real interpreter).

The point of the dogfood isn't the tool implementations. It's to show the
shape of an agent that uses Router.from_examples() to pick which tool(s) to
call for a free-form natural-language task.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    describe: str
    run: Callable[[str], str]


def _web_search(query: str) -> str:
    return (
        f"[mock web_search] would query a search API for: {query!r}\n"
        f"  -> top result: 'A blog post that vaguely matches your query.'"
    )


_SAFE_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


def _calculator(expression: str) -> str:
    # Tiny safe evaluator: numbers and + - * / only. No eval(), no names.
    # Skip leading words: pick out the longest numeric+operator subsequence.
    cleaned = expression.replace(" ", "")
    tokens = re.findall(r"-?\d+(?:\.\d+)?|[+\-*/]", cleaned)
    # Trim leading/trailing operators so "find/open" doesn't yield ['/'].
    while tokens and tokens[0] in _SAFE_OPS:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _SAFE_OPS:
        tokens = tokens[:-1]
    has_number = any(t not in _SAFE_OPS for t in tokens)
    if not has_number:
        return f"[calculator] no numeric expression found in: {expression!r}"
    try:
        result = float(tokens[0])
        i = 1
        while i < len(tokens) - 1:
            op, rhs = tokens[i], float(tokens[i + 1])
            result = _SAFE_OPS[op](result, rhs)
            i += 2
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        return f"[calculator] error: {exc}"
    return f"[calculator] picked numbers from {expression!r} -> result = {result}"


def _file_read(path: str) -> str:
    return (
        f"[mock file_read] would open {path!r} and return its contents.\n"
        f"  -> first line (mocked): '# placeholder file content'"
    )


def _memory_lookup(topic: str) -> str:
    return (
        f"[mock memory_lookup] would search prior conversation/memory for: {topic!r}\n"
        f"  -> match (mocked): 'earlier you mentioned that {topic} mattered for the q3 review.'"
    )


def _python_exec(code: str) -> str:
    return (
        f"[mock python_exec] would execute in a sandbox:\n"
        f"  >>> {code.strip()[:80]}{'...' if len(code) > 80 else ''}\n"
        f"  -> stdout (mocked): '<result>'"
    )


TOOLS: dict[str, Tool] = {
    "web_search": Tool(
        name="web_search",
        describe="search the public web for up-to-date information",
        run=_web_search,
    ),
    "calculator": Tool(
        name="calculator",
        describe="evaluate a numeric expression",
        run=_calculator,
    ),
    "file_read": Tool(
        name="file_read",
        describe="read the contents of a local file",
        run=_file_read,
    ),
    "memory_lookup": Tool(
        name="memory_lookup",
        describe="recall something from earlier in the conversation",
        run=_memory_lookup,
    ),
    "python_exec": Tool(
        name="python_exec",
        describe="run a short python snippet in a sandbox",
        run=_python_exec,
    ),
}

"""Unified trace schema.

All benchmark loaders normalize to this shape. One trace = one agent task,
from prompt to outcome, with the sequence of tools the agent invoked.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ToolCall:
    name: str                 # e.g. "edit_file", "bash", "browser.click"
    args: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    duration_ms: int | None = None


@dataclass
class Trace:
    trace_id: str             # globally unique, e.g. "swebench-verified/django__django-12345"
    source_dataset: str       # "swe-bench-verified" | "tau-bench" | "osworld" | ...
    task_text: str            # natural-language description of the task
    tools_called: list[ToolCall] = field(default_factory=list)
    outcome: str = "unknown"  # "success" | "failure" | "partial" | "unknown"
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> dict[str, Any]:
        return asdict(self)

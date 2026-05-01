"""CLI: python -m agent_tool_router "your task" -k 3"""

from __future__ import annotations

import argparse
import sys

from .router import Router


def main() -> int:
    p = argparse.ArgumentParser(description="Route a task to its top-k tools.")
    p.add_argument("task", help="task description in natural language")
    p.add_argument("-k", type=int, default=3, help="how many tools to return")
    p.add_argument("--model", default="baseline-v0",
                   help="model name (in ./models/) or path")
    p.add_argument("--scores", action="store_true",
                   help="print scores alongside tool names")
    args = p.parse_args()

    try:
        r = Router.from_pretrained(args.model)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.scores:
        results = r.route(args.task, k=args.k, return_scores=True)
        for res in results:
            print(f"{res.score:.4f}\t{res.tool}")
    else:
        results = r.route(args.task, k=args.k)
        for tool in results:
            print(tool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

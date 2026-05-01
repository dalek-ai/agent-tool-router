"""A 70-line agent that uses Router.from_examples() to pick its tools.

Run a single task:
    python -m examples.research_helper.agent "find me the latest paper on RLHF"

Run a built-in demo of 6 tasks (no args):
    python -m examples.research_helper.agent

Run interactively:
    python -m examples.research_helper.agent -i

The agent itself is dumb on purpose. It does not call an LLM. It asks the
router for the top-k tools, then runs whichever ones the router returned
that score above a threshold. That is enough to show the value of routing:
the prompt to a downstream LLM only needs the tools that survived this step,
not all five.
"""

from __future__ import annotations

import argparse
import sys

from agent_tool_router import Router

from .seed_examples import SEED_EXAMPLES
from .tools import TOOLS

DEMO_TASKS = [
    "find me a recent survey paper on tool-using agents",
    "what is 19 * 23 + 7",
    "open ./README.md and tell me what is in it",
    "remind me what i said earlier about the migration plan",
    "run a python snippet that returns sorted([3,1,2])",
    "look up the population of japan online and divide it by 1000",
]


def _build_router() -> Router:
    return Router.from_examples(SEED_EXAMPLES)


def _route_and_run(router: Router, task: str, k: int = 2, threshold: float = 0.10) -> None:
    results = router.route(task, k=k, return_scores=True)
    print(f"\n>>> task: {task}")
    print("    router returned:")
    for r in results:
        marker = "[run] " if r.score >= threshold else "[skip]"
        print(f"      {marker} {r.tool:<14}  score={r.score:.3f}")
    chosen = [r.tool for r in results if r.score >= threshold]
    if not chosen:
        print("    no tool scored above threshold — falling back to top-1.")
        chosen = [results[0].tool]
    for tool_name in chosen:
        tool = TOOLS[tool_name]
        out = tool.run(task)
        print(f"    -- {tool_name} ({tool.describe}) --")
        for line in out.splitlines():
            print(f"       {line}")


def main() -> int:
    p = argparse.ArgumentParser(description="Research-helper dogfood for agent-tool-router.")
    p.add_argument("task", nargs="?", help="task in plain english (omit for demo).")
    p.add_argument("-k", type=int, default=2, help="how many tools to consider.")
    p.add_argument("-i", "--interactive", action="store_true",
                   help="prompt for tasks until empty line.")
    p.add_argument("--threshold", type=float, default=0.10,
                   help="minimum cosine score to actually run a tool.")
    args = p.parse_args()

    print(f"[router] training on {len(SEED_EXAMPLES)} seed examples...", file=sys.stderr)
    router = _build_router()
    print(f"[router] vocab = {router.vocab}", file=sys.stderr)

    if args.interactive:
        try:
            while True:
                line = input("task> ").strip()
                if not line:
                    break
                _route_and_run(router, line, k=args.k, threshold=args.threshold)
        except (EOFError, KeyboardInterrupt):
            print()
        return 0

    tasks = [args.task] if args.task else DEMO_TASKS
    for t in tasks:
        _route_and_run(router, t, k=args.k, threshold=args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate baseline-v1-desc against the full corpus.

Unlike baseline_cross_corpus.py and baseline_loso_descriptions.py, which
restrict the candidate catalog at eval time (to vocab freq>=3, or to the
held-out source's tools), this evaluates the *production* setup: load the
shipped baseline-v1-desc model, route every trace's task against the full
18 671-tool catalog, measure top-k accuracy per source.

This is the noisier number to report, because confounders from sibling
sources (e.g., ToolACE's 16K synthetic tools) compete for top slots even on
non-ToolACE tasks. It's also the number a user would actually observe when
they load the pretrained model and route a task.

Usage:
  python -m router.eval.eval_baseline_v1_desc
  python -m router.eval.eval_baseline_v1_desc --traces data/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_tool_router import Router  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="baseline-v1-desc")
    p.add_argument("--traces", default=str(ROOT / "data" / "traces.jsonl"))
    p.add_argument("--ks", default="1,3,5,10",
                   help="comma-separated list of k values to evaluate")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    k_max = max(ks)

    r = Router.from_pretrained(args.model)
    V = len(r.vocab)
    vocab_set = set(r.vocab)
    print(f"[load] model={args.model} vocab={V}", file=sys.stderr)

    rows: list[tuple[str, list[str], str]] = []
    with open(args.traces, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            task = (t.get("task_text") or "").strip()
            gold = [tc.get("name") for tc in (t.get("tools_called") or [])]
            gold = [g for g in gold if isinstance(g, str) and g and g in vocab_set]
            if not task or not gold:
                continue
            src = t.get("source_dataset", "unknown")
            rows.append((task, gold, src))
    print(f"[load] {len(rows)} traces with at least one gold tool in v1-desc vocab",
          file=sys.stderr)

    per_source_hits: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    per_source_total: dict[str, int] = defaultdict(int)
    overall_hits: dict[int, int] = defaultdict(int)
    overall_total = 0

    bs = args.batch_size
    for batch_start in range(0, len(rows), bs):
        batch = rows[batch_start:batch_start + bs]
        tasks = [row[0] for row in batch]
        topk_lists = r.route(tasks, k=k_max)
        for (task, gold, src), topk in zip(batch, topk_lists):
            for tool in gold:
                per_source_total[src] += 1
                overall_total += 1
                for k in ks:
                    if tool in topk[:k]:
                        per_source_hits[src][k] += 1
                        overall_hits[k] += 1

    print(f"[eval] V={V}, n_calls={overall_total}", file=sys.stderr)
    print()
    print("=== overall (full v1-desc catalog) ===")
    for k in ks:
        acc = overall_hits[k] / overall_total if overall_total else 0.0
        rnd = k / V
        print(f"  top-{k:>2} acc = {acc:.4f}  ({acc/rnd:.1f}x random, n={overall_total})")
    print()

    print("=== per source ===")
    for src in sorted(per_source_total):
        n = per_source_total[src]
        print(f"  {src} (n_calls={n})")
        for k in ks:
            acc = per_source_hits[src][k] / n if n else 0.0
            rnd = k / V
            print(f"    top-{k:>2} acc = {acc:.4f}  ({acc/rnd:.1f}x random)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

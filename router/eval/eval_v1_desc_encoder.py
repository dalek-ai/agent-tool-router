"""Evaluate v1-desc with the bi-encoder backend on the full 18K-tool catalog.

Companion to eval_baseline_v1_desc.py (which evaluates the shipped tfidf model
on the same setup). The motivation: in tfidf, tau-bench collapses from 19.8%
top-3 (LOSO restricted to its 23 tools) to 3.2% (full 18K-tool catalog).
The hypothesis is that the bi-encoder should be more robust to confounders
because it scores semantic similarity rather than lexical overlap, so its
tau-bench accuracy at full-catalog scale should not drop nearly as hard.

This script builds an in-memory Router with backend="encoder" from the same
data/tool_descriptions.jsonl, then evaluates against data/traces.jsonl on the
union of all sources, restricting gold tools to those present in the catalog
(same filtering rule as eval_baseline_v1_desc.py).

Reports overall and per-source top-k accuracy with random-baseline ratios.

Usage:
  python -m router.eval.eval_v1_desc_encoder
  python -m router.eval.eval_v1_desc_encoder --hybrid --alpha 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_tool_router import Router  # noqa: E402


def load_descriptions(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            name = d.get("name")
            desc = (d.get("description") or "").strip()
            if not isinstance(name, str) or not name or name in seen:
                continue
            if not desc:
                continue
            seen.add(name)
            rows.append((name, desc))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--descriptions",
                   default=str(ROOT / "data" / "tool_descriptions.jsonl"))
    p.add_argument("--traces", default=str(ROOT / "data" / "traces.jsonl"))
    p.add_argument("--ks", default="1,3,5,10")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hybrid", action="store_true",
                   help="also score with backend='hybrid' (alpha-weighted "
                        "tfidf + encoder); requires both backends to be fit.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="hybrid mixing weight on tfidf (default 0.5).")
    p.add_argument("--encoder-model",
                   default="sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    k_max = max(ks)

    descs = load_descriptions(Path(args.descriptions))
    print(f"[load] {len(descs)} (name, description) pairs", file=sys.stderr)

    backend = "hybrid" if args.hybrid else "encoder"
    t0 = time.time()
    r = Router.from_descriptions(
        descs,
        backend=backend,
        alpha=args.alpha,
        encoder_model_name=args.encoder_model,
        include_name=True,
    )
    print(f"[build] backend={backend} V={len(r.vocab)} "
          f"in {time.time()-t0:.1f}s", file=sys.stderr)

    vocab_set = set(r.vocab)
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
    print(f"[load] {len(rows)} traces with at least one gold tool in catalog",
          file=sys.stderr)

    per_source_hits: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    per_source_total: dict[str, int] = defaultdict(int)
    overall_hits: dict[int, int] = defaultdict(int)
    overall_total = 0

    bs = args.batch_size
    t0 = time.time()
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
        if (batch_start // bs) % 10 == 0:
            done = batch_start + len(batch)
            print(f"[eval] {done}/{len(rows)} ({time.time()-t0:.0f}s)",
                  file=sys.stderr)

    V = len(r.vocab)
    label = f"v1-desc backend={backend}"
    if backend == "hybrid":
        label += f" alpha={args.alpha}"
    print(f"[eval] V={V}, n_calls={overall_total}, took {time.time()-t0:.0f}s",
          file=sys.stderr)
    print()
    print(f"=== overall ({label}, full catalog) ===")
    for k in ks:
        acc = overall_hits[k] / overall_total if overall_total else 0.0
        rnd = k / V
        print(f"  top-{k:>2} acc = {acc:.4f}  ({acc/rnd:.1f}x random, "
              f"n={overall_total})")
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

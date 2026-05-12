"""Measure next-tool prediction accuracy of the shipped retrieval router.

For each triplet (task_text, history, next_tool) emitted by
build_next_tool_dataset.py, run `Router.from_pretrained(...).route(task, k=10)`
and check whether the gold next_tool is in the top-1/3/5/10.

Retrieval-only baseline: the router never sees `history`. This is the
control we will compare to history-aware models (Markov-1, rerank).

Stratifies by position (t=1, t=2, t>=3) and by source.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from time import time

from agent_tool_router import Router

ROOT = Path(__file__).resolve().parents[2]
TRIPLETS = ROOT / "data" / "next_tool_triplets.jsonl"
MODEL = "baseline-v1-desc-hybrid"
KS = (1, 3, 5, 10)
TOP_K = max(KS)
BATCH = 256


def normalize(name: str) -> str:
    return (name or "").strip().lower()


def main() -> None:
    print(f"Loading router '{MODEL}'...")
    router = Router.from_pretrained(MODEL)

    triplets = [json.loads(line) for line in TRIPLETS.open()]
    print(f"Triplets: {len(triplets)}")

    bucket_keys = lambda t: (
        t["source"],
        "t=1" if t["position"] == 1 else ("t=2" if t["position"] == 2 else "t>=3"),
    )

    hits = defaultdict(lambda: {k: 0 for k in KS})
    totals = defaultdict(int)
    overall_hits = {k: 0 for k in KS}
    overall_total = 0

    t0 = time()
    for start in range(0, len(triplets), BATCH):
        batch = triplets[start : start + BATCH]
        tasks = [t["task_text"] for t in batch]
        topk_lists = router.route(tasks, k=TOP_K)
        for trip, names in zip(batch, topk_lists):
            gold = normalize(trip["next_tool"])
            normed = [normalize(n) for n in names]
            for k in KS:
                if gold in normed[:k]:
                    overall_hits[k] += 1
                    for key in [bucket_keys(trip), ("ALL", bucket_keys(trip)[1])]:
                        hits[key][k] += 1
            overall_total += 1
            for key in [bucket_keys(trip), ("ALL", bucket_keys(trip)[1])]:
                totals[key] += 1
        if (start // BATCH) % 10 == 0:
            print(f"  {start + len(batch)}/{len(triplets)} ({time() - t0:.1f}s)")

    print(f"\nDone in {time() - t0:.1f}s\n")
    print(
        f"OVERALL (n={overall_total}): "
        + " | ".join(
            f"top-{k}={overall_hits[k] / overall_total:.1%}" for k in KS
        )
    )
    print()
    print(f"{'source':<32}{'pos':<8}{'n':>7}  " + "  ".join(f"top-{k:>2}" for k in KS))
    print("-" * 80)
    for (src, pos), n in sorted(totals.items()):
        h = hits[(src, pos)]
        row = f"{src:<32}{pos:<8}{n:>7}  " + "  ".join(
            f"{h[k] / n:>5.1%}" for k in KS
        )
        print(row)


if __name__ == "__main__":
    main()

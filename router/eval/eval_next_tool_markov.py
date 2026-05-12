"""Markov-1 history-aware rerank vs retrieval-only.

Splits triplets by trace_id into train/test (80/20). On train, counts
P(next_tool | last_history_tool) with add-one smoothing. On test, fetches
top-N candidates from `Router.from_pretrained(...)`, then reranks with:

    final = alpha * retrieval_norm + (1 - alpha) * markov_prior

Reports top-1/3/5/10 stratified by position and source. Sweeps alpha
over a small grid to find the sweet spot.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from time import time

import numpy as np

from agent_tool_router import Router

ROOT = Path(__file__).resolve().parents[2]
TRIPLETS = ROOT / "data" / "next_tool_triplets.jsonl"
MODEL = "baseline-v1-desc-hybrid"
RETRIEVAL_N = 50  # candidates to consider for rerank
KS = (1, 3, 5, 10)
ALPHAS = (1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0)
BATCH = 256
SEED = 17


def normalize(name: str) -> str:
    return (name or "").strip().lower()


def split_by_trace(triplets, frac=0.8):
    by_trace = defaultdict(list)
    for t in triplets:
        by_trace[t["trace_id"]].append(t)
    trace_ids = sorted(by_trace)
    rng = random.Random(SEED)
    rng.shuffle(trace_ids)
    cut = int(len(trace_ids) * frac)
    train = [t for tid in trace_ids[:cut] for t in by_trace[tid]]
    test = [t for tid in trace_ids[cut:] for t in by_trace[tid]]
    return train, test


def build_markov(train):
    """P(next | prev) with add-one smoothing on the seen-tool vocabulary."""
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    vocab = set()
    for t in train:
        if not t["history"]:
            continue
        prev = normalize(t["history"][-1])
        nxt = normalize(t["next_tool"])
        counts[prev][nxt] += 1
        totals[prev] += 1
        vocab.add(prev)
        vocab.add(nxt)
    V = len(vocab)
    return counts, totals, V


def markov_prob(counts, totals, V, prev: str, candidate: str) -> float:
    if prev not in totals:
        return 1.0 / max(V, 1)
    return (counts[prev].get(candidate, 0) + 1.0) / (totals[prev] + V)


def main() -> None:
    print(f"Loading router '{MODEL}'...")
    router = Router.from_pretrained(MODEL)

    triplets = [json.loads(line) for line in TRIPLETS.open()]
    train, test = split_by_trace(triplets)
    print(f"Triplets: train={len(train)} test={len(test)}")
    counts, totals, V = build_markov(train)
    print(f"Markov vocab (seen prev/next): {V}")

    print(f"\nRouting test set (k={RETRIEVAL_N})...")
    t0 = time()
    all_cands: list[list[tuple[str, float]]] = []
    for start in range(0, len(test), BATCH):
        batch = test[start : start + BATCH]
        tasks = [t["task_text"] for t in batch]
        results = router.route(tasks, k=RETRIEVAL_N, return_scores=True)
        for res in results:
            all_cands.append([(r.tool, r.score) for r in res])
        if (start // BATCH) % 10 == 0:
            print(f"  {start + len(batch)}/{len(test)} ({time() - t0:.1f}s)")
    print(f"Routed in {time() - t0:.1f}s")

    def bucket(t):
        pos = "t=1" if t["position"] == 1 else ("t=2" if t["position"] == 2 else "t>=3")
        return (t["source"], pos)

    results_by_alpha: dict[float, dict] = {}
    for alpha in ALPHAS:
        hits = defaultdict(lambda: {k: 0 for k in KS})
        totals_b = defaultdict(int)
        overall_hits = {k: 0 for k in KS}
        for trip, cands in zip(test, all_cands):
            prev = normalize(trip["history"][-1]) if trip["history"] else ""
            gold = normalize(trip["next_tool"])
            ret_scores = np.array([c[1] for c in cands], dtype=float)
            if len(ret_scores) > 1:
                ret_norm = (ret_scores - ret_scores.min()) / (
                    ret_scores.max() - ret_scores.min() + 1e-9
                )
            else:
                ret_norm = ret_scores
            mk_scores = np.array(
                [markov_prob(counts, totals, V, prev, normalize(c[0])) for c in cands]
            )
            mk_norm = (mk_scores - mk_scores.min()) / (
                mk_scores.max() - mk_scores.min() + 1e-9
            )
            final = alpha * ret_norm + (1.0 - alpha) * mk_norm
            order = np.argsort(-final)
            ranked = [normalize(cands[i][0]) for i in order]
            for k in KS:
                if gold in ranked[:k]:
                    overall_hits[k] += 1
                    for key in [bucket(trip), ("ALL", bucket(trip)[1])]:
                        hits[key][k] += 1
            for key in [bucket(trip), ("ALL", bucket(trip)[1])]:
                totals_b[key] += 1
        results_by_alpha[alpha] = (overall_hits, hits, totals_b)

    n = len(test)
    print(f"\nOVERALL top-3 by alpha (n={n}):")
    print(f"  {'alpha':>6}  {'top-1':>6}  {'top-3':>6}  {'top-5':>6}  {'top-10':>7}")
    for alpha in ALPHAS:
        oh = results_by_alpha[alpha][0]
        print(
            f"  {alpha:>6.2f}  "
            + "  ".join(f"{oh[k] / n:>5.1%} " for k in (1, 3, 5))
            + f"  {oh[10] / n:>6.1%}"
        )

    # Show the best alpha in detail
    best_alpha = max(ALPHAS, key=lambda a: results_by_alpha[a][0][3])
    print(f"\nBest alpha by top-3: {best_alpha}")
    oh, hits, totals_b = results_by_alpha[best_alpha]
    print(f"\n{'source':<32}{'pos':<8}{'n':>7}  " + "  ".join(f"top-{k:>2}" for k in KS))
    print("-" * 80)
    for (src, pos), cnt in sorted(totals_b.items()):
        h = hits[(src, pos)]
        print(
            f"{src:<32}{pos:<8}{cnt:>7}  "
            + "  ".join(f"{h[k] / cnt:>5.1%}" for k in KS)
        )


if __name__ == "__main__":
    main()

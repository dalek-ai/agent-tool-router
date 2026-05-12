"""Bucket-size sweep: does widening the retrieval top-N break the Markov-1
ceiling found in session 22-24?

Session 22 finding (eval_next_tool_markov.py): with top-50 retrieval on
the held-out test (n=2094), a Markov-1 rerank trained on the 80% train
split reaches 48.0% top-3 — that's 99.5% of recall@50 = 48.2% in the
held-out setting. Anything stacked on top-50 is bounded by that ceiling.

This script re-runs the same Markov-1 rerank logic on widened buckets
(top-K for K ∈ {50, 100, 150, 200}). The cache built by
`build_next_tool_cache.py` now holds top-200 candidates per triplet,
so we just slice [:K]. For each K we report:
  - recall@K (the ceiling for any rerank in that bucket)
  - Markov-1 top-1/3/5/10/K (so we can see how close it gets to recall@K)
  - per-source / per-position breakdown at the best alpha

**The Markov-1 table is rebuilt from the 80% train split here**, not loaded
from `markov_counts.npz` (which the SDK ships as full-dataset — that
would leak the 20% test triplets into the prior and inflate the numbers).

Decision rule (matches what STATE.md set as N63 criterion):
  ship the widening if recall@200 > 70% AND Markov top-3 > 62%.
Otherwise log it as a negative finding.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from time import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRIPLETS = ROOT / "data" / "next_tool_triplets.jsonl"
CACHE = ROOT / "data" / "cache" / "next_tool"

SEED = 17
BUCKET_SIZES = (50, 100, 150, 200)
ALPHAS = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
KS = (1, 3, 5, 10)


def normalize(name: str) -> str:
    return (name or "").strip().lower()


def split_by_trace(triplets, frac=0.8):
    by_trace = defaultdict(list)
    for i, t in enumerate(triplets):
        by_trace[t["trace_id"]].append(i)
    trace_ids = sorted(by_trace)
    rng = random.Random(SEED)
    rng.shuffle(trace_ids)
    cut = int(len(trace_ids) * frac)
    return (
        [i for tid in trace_ids[:cut] for i in by_trace[tid]],
        [i for tid in trace_ids[cut:] for i in by_trace[tid]],
    )


def build_markov_train(train_triplets):
    """Honest train-only Markov-1 with add-one smoothing on the seen vocab.

    Same rule as `eval_next_tool_markov.py` from session 22, so the K=50
    numbers should reproduce.
    """
    from collections import defaultdict as dd

    counts = dd(lambda: dd(int))
    totals = dd(int)
    vocab = set()
    for t in train_triplets:
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


def markov_prob(counts, totals, V, prev_name, cand_name):
    if prev_name not in totals:
        return 1.0 / max(V, 1)
    return (counts[prev_name].get(cand_name, 0) + 1.0) / (totals[prev_name] + V)


def markov_score_row(counts, totals, V, prev_name, cand_names):
    out = np.empty(len(cand_names), dtype=np.float32)
    for j, cn in enumerate(cand_names):
        out[j] = markov_prob(counts, totals, V, prev_name, cn)
    return out


def minmax(x):
    if x.size == 0:
        return x
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


def bucket_pos(t):
    p = t["position"]
    return "t=1" if p == 1 else ("t=2" if p == 2 else "t>=3")


def main():
    triplets = [json.loads(line) for line in TRIPLETS.open()]
    cand_idx = np.load(CACHE / "cand_idx.npy")
    cand_scores = np.load(CACHE / "cand_scores.npy")
    prev_idx = np.load(CACHE / "prev_idx.npy")
    gold_idx = np.load(CACHE / "gold_idx.npy")

    print(f"Cache: cand_idx {cand_idx.shape}, cand_scores {cand_scores.shape}")
    assert cand_idx.shape[1] >= max(BUCKET_SIZES), (
        f"cache has only top-{cand_idx.shape[1]}; rebuild with TOP_N>={max(BUCKET_SIZES)}"
    )

    from agent_tool_router import Router

    router = Router.from_pretrained("baseline-v1-desc-hybrid")
    vocab = [normalize(v) for v in router.vocab]

    train_idx, test_idx = split_by_trace(triplets)
    n = len(test_idx)
    print(f"Triplets: train={len(train_idx)} test={n}")

    train_triplets = [triplets[i] for i in train_idx]
    counts, totals, V = build_markov_train(train_triplets)
    print(f"Markov-1 vocab (seen prev/next, train-only): {V}")

    print("\n=== Recall@K by retrieval bucket size ===")
    print(f"  {'K':>4}  recall@K   t=1        t=2        t>=3")
    recall_by_K = {}
    for K in BUCKET_SIZES:
        hits = 0
        pos_hits = defaultdict(int)
        pos_n = defaultdict(int)
        for i in test_idx:
            g = gold_idx[i]
            present = g in cand_idx[i, :K]
            if present:
                hits += 1
            b = bucket_pos(triplets[i])
            pos_n[b] += 1
            if present:
                pos_hits[b] += 1
        rec = hits / n
        recall_by_K[K] = rec
        print(
            f"  {K:>4}  {rec:>7.1%}   "
            + f"{pos_hits['t=1']/max(pos_n['t=1'],1):>7.1%}    "
            + f"{pos_hits['t=2']/max(pos_n['t=2'],1):>7.1%}    "
            + f"{pos_hits['t>=3']/max(pos_n['t>=3'],1):>7.1%}"
        )

    print("\n=== Markov-1 rerank by bucket size (best alpha per bucket) ===")
    print(
        f"  {'K':>4}  {'best alpha':>10}  {'top-1':>6}  {'top-3':>6}  {'top-5':>6}  {'top-10':>7}  {'recall@K':>9}"
    )
    best_by_K = {}
    detailed_by_K = {}
    t0 = time()
    for K in BUCKET_SIZES:
        per_alpha = {}
        for alpha in ALPHAS:
            hits = {k: 0 for k in KS}
            for i in test_idx:
                cids = cand_idx[i, :K]
                mask = cids >= 0
                cids_v = cids[mask]
                if len(cids_v) == 0:
                    continue
                cnames = [vocab[c] for c in cids_v]
                prev_name = vocab[prev_idx[i]] if prev_idx[i] >= 0 else ""
                gold = vocab[gold_idx[i]]
                ret = cand_scores[i, : len(cids_v)]
                ret_n = minmax(ret)
                mk = markov_score_row(counts, totals, V, prev_name, cnames)
                mk_n = minmax(mk)
                final = alpha * ret_n + (1 - alpha) * mk_n
                order = np.argsort(-final)
                ranked = [cnames[j] for j in order]
                for k in KS:
                    if gold in ranked[:k]:
                        hits[k] += 1
            per_alpha[alpha] = hits
        # pick the alpha with the highest top-3
        best_alpha = max(ALPHAS, key=lambda a: per_alpha[a][3])
        h = per_alpha[best_alpha]
        best_by_K[K] = (best_alpha, h)
        detailed_by_K[K] = per_alpha
        print(
            f"  {K:>4}  {best_alpha:>10.2f}  "
            + "  ".join(f"{h[k]/n:>5.1%}" for k in KS)
            + f"  {recall_by_K[K]:>9.1%}"
        )
    print(f"\n(sweep took {time()-t0:.1f}s)")

    # Detailed table at top-200 best alpha — show source × position
    K_widest = max(BUCKET_SIZES)
    ba, _ = best_by_K[K_widest]
    print(f"\n=== Markov-1 rerank at K={K_widest}, alpha={ba:.2f}: source × position ===")
    hits_b = defaultdict(lambda: {k: 0 for k in KS})
    tot_b = defaultdict(int)
    for i in test_idx:
        cids = cand_idx[i, :K_widest]
        mask = cids >= 0
        cids_v = cids[mask]
        if len(cids_v) == 0:
            continue
        cnames = [vocab[c] for c in cids_v]
        prev_name = vocab[prev_idx[i]] if prev_idx[i] >= 0 else ""
        gold = vocab[gold_idx[i]]
        ret = cand_scores[i, : len(cids_v)]
        ret_n = minmax(ret)
        mk = markov_score_row(counts, totals, V, prev_name, cnames)
        mk_n = minmax(mk)
        final = ba * ret_n + (1 - ba) * mk_n
        order = np.argsort(-final)
        ranked = [cnames[j] for j in order]
        t = triplets[i]
        keys = [(t["source"], bucket_pos(t)), ("ALL", bucket_pos(t))]
        for k in KS:
            if gold in ranked[:k]:
                for key in keys:
                    hits_b[key][k] += 1
        for key in keys:
            tot_b[key] += 1
    print(f"\n{'source':<32}{'pos':<8}{'n':>7}  " + "  ".join(f"top-{k:>2}" for k in KS))
    for (src, pos), cnt in sorted(tot_b.items()):
        if src == "ALL":
            continue
        h = hits_b[(src, pos)]
        print(f"{src:<32}{pos:<8}{cnt:>7}  " + "  ".join(f"{h[k]/cnt:>5.1%}" for k in KS))
    print(f"\n{'ALL':<32}{'pos':<8}{'n':>7}  " + "  ".join(f"top-{k:>2}" for k in KS))
    for (src, pos), cnt in sorted(tot_b.items()):
        if src != "ALL":
            continue
        h = hits_b[(src, pos)]
        print(f"{src:<32}{pos:<8}{cnt:>7}  " + "  ".join(f"{h[k]/cnt:>5.1%}" for k in KS))

    # Decision summary against N63 criterion (recall@200 > 70% AND markov top-3 > 62%)
    K = max(BUCKET_SIZES)
    rec_K = recall_by_K[K]
    _, h_K = best_by_K[K]
    mk_top3_K = h_K[3] / n
    print(
        f"\n=== Decision (criterion: recall@{K} > 70% AND Markov-1 top-3 > 62%) ==="
    )
    print(f"  recall@{K}  = {rec_K:.1%}  (criterion: > 70.0%)  {'PASS' if rec_K > 0.70 else 'FAIL'}")
    print(
        f"  markov top-3 @ K={K}, alpha={best_by_K[K][0]:.2f} = {mk_top3_K:.1%}  (criterion: > 62.0%)  {'PASS' if mk_top3_K > 0.62 else 'FAIL'}"
    )

    # also show top-3 lift vs top-50 baseline (58.7% from session 22-24)
    baseline_top3 = best_by_K[50][1][3] / n
    print(f"\n  top-3 by K (vs baseline K=50 = {baseline_top3:.1%}):")
    for K in BUCKET_SIZES:
        v = best_by_K[K][1][3] / n
        delta = v - baseline_top3
        print(f"    K={K:>4}: {v:.1%}  ({delta:+.1%}pp vs K=50)")


if __name__ == "__main__":
    main()

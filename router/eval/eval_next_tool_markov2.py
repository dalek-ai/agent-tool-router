"""Markov-2 (history bigram) rerank with stupid backoff to Markov-1.

Session 22-26 shipped Markov-1: P(next | prev[-1]) reranks the top-K retrieval
candidates and lifts next-tool top-3 by ~20pp on the held-out 20% split. But the
source x position breakdown shows the gain plateaus on long-horizon agents
(tau-bench t>=3 = 84.8% top-3 with next-v1 encoder), where the previous tool
alone doesn't fully condition the choice — you need at least the prev-prev too
to disambiguate (e.g. update_reservation_flights -> update_reservation_passengers
-> update_reservation_baggages).

5412/10480 next-tool triplets in `data/next_tool_triplets.jsonl` have history
length >= 2, so the bigram has real coverage. This script tests whether a
Markov-2 with stupid backoff to Markov-1 lifts the rerank ceiling.

Stupid backoff (Brants 2007):
  score(c | p2, p1) = count(p2, p1, c) / count(p2, p1)            if seen
                   = lambda * count(p1, c) / count(p1)             if backoff to M1
                   = lambda^2 / V                                  if backoff to uniform

lambda=0.4 follows the original paper. We minmax-normalize the score row before
blending with retrieval, so the multiplicative penalty between tiers doesn't
matter for ordering — what matters is that seen bigrams rank above seen unigrams
which rank above unseen-everything within a single candidate row.

The Markov-1 table is rebuilt from the 80% train split so we don't leak the
20% test prior (same protocol as the session 25 leak fix).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from time import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRIPLETS = ROOT / "data" / "next_tool_triplets.jsonl"
DEFAULT_CACHE = ROOT / "data" / "cache" / "next_tool_v1"
DEFAULT_MODEL = "models/baseline-v1-desc-hybrid-next-v1"

SEED = 17
K_BUCKET = 200
ALPHAS = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
KS = (1, 3, 5, 10)
LAMBDA_BACKOFF = 0.4


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


def build_markov12_train(train_triplets):
    c12 = defaultdict(lambda: defaultdict(int))
    t12 = defaultdict(int)
    c1 = defaultdict(lambda: defaultdict(int))
    t1 = defaultdict(int)
    vocab = set()
    for t in train_triplets:
        nxt = normalize(t["next_tool"])
        vocab.add(nxt)
        if not t["history"]:
            continue
        h = [normalize(x) for x in t["history"]]
        vocab.update(h)
        p1 = h[-1]
        c1[p1][nxt] += 1
        t1[p1] += 1
        if len(h) >= 2:
            p2 = h[-2]
            key = (p2, p1)
            c12[key][nxt] += 1
            t12[key] += 1
    return c12, t12, c1, t1, len(vocab)


def markov1_score(c1, t1, V, p1, cand_names):
    out = np.empty(len(cand_names), dtype=np.float32)
    if p1 in t1:
        total = t1[p1]
        for j, c in enumerate(cand_names):
            out[j] = (c1[p1].get(c, 0) + 1.0) / (total + V)
    else:
        out[:] = 1.0 / max(V, 1)
    return out


def markov2_backoff_score(c12, t12, c1, t1, V, p2, p1, cand_names, lam=LAMBDA_BACKOFF):
    out = np.empty(len(cand_names), dtype=np.float32)
    bigram_key = (p2, p1) if p2 is not None else None
    bigram_total = t12.get(bigram_key, 0) if bigram_key is not None else 0
    unigram_total = t1.get(p1, 0)
    for j, c in enumerate(cand_names):
        if bigram_total > 0:
            cnt = c12[bigram_key].get(c, 0)
            if cnt > 0:
                out[j] = cnt / bigram_total
                continue
        if unigram_total > 0:
            cnt1 = c1[p1].get(c, 0)
            if cnt1 > 0:
                out[j] = lam * cnt1 / unigram_total
                continue
        out[j] = (lam * lam) / max(V, 1)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--lambda-backoff", type=float, default=LAMBDA_BACKOFF)
    args = ap.parse_args()
    cache = Path(args.cache_dir)

    triplets = [json.loads(line) for line in TRIPLETS.open()]
    cand_idx = np.load(cache / "cand_idx.npy")
    cand_scores = np.load(cache / "cand_scores.npy")
    gold_idx = np.load(cache / "gold_idx.npy")

    print(f"Cache: cand_idx {cand_idx.shape}, model {args.model}, lambda={args.lambda_backoff}")
    K = K_BUCKET

    from agent_tool_router import Router
    router = Router.from_pretrained(args.model)
    vocab = [normalize(v) for v in router.vocab]

    train_idx, test_idx = split_by_trace(triplets)
    n = len(test_idx)
    print(f"Train={len(train_idx)} test={n}")

    train_triplets = [triplets[i] for i in train_idx]
    t0 = time()
    c12, t12, c1, t1, V = build_markov12_train(train_triplets)
    print(f"Train tables built in {time()-t0:.1f}s")
    print(f"  bigram keys (p2,p1): {len(t12)}")
    print(f"  unigram keys p1    : {len(t1)}")
    print(f"  vocab seen         : {V}")

    have_bigram_ctx = 0
    have_bigram_match = 0
    for i in test_idx:
        t = triplets[i]
        h = [normalize(x) for x in t["history"]]
        if len(h) >= 2:
            have_bigram_ctx += 1
            if (h[-2], h[-1]) in t12:
                have_bigram_match += 1
    print(f"\nTest rows with history>=2 : {have_bigram_ctx}/{n} ({have_bigram_ctx/n:.1%})")
    print(f"  ... whose (p2,p1) in train: {have_bigram_match} ({have_bigram_match/max(have_bigram_ctx,1):.1%})")

    print(f"\n=== Rerank top-3 vs alpha (K={K}) ===")
    print(f"  {'alpha':>5}  {'M1 top-3':>9}  {'M2 top-3':>9}  {'delta':>7}")
    best = {"M1": (None, 0.0, None), "M2": (None, 0.0, None)}
    for alpha in ALPHAS:
        h_m1 = {k: 0 for k in KS}
        h_m2 = {k: 0 for k in KS}
        for i in test_idx:
            cids = cand_idx[i, :K]
            mask = cids >= 0
            cids_v = cids[mask]
            if len(cids_v) == 0:
                continue
            cnames = [vocab[c] for c in cids_v]
            gold = vocab[gold_idx[i]]
            ret = cand_scores[i, : len(cids_v)]
            ret_n = minmax(ret)

            hist = [normalize(x) for x in triplets[i]["history"]]
            if not hist:
                final_m1 = ret_n.copy()
                final_m2 = ret_n.copy()
            else:
                p1 = hist[-1]
                p2 = hist[-2] if len(hist) >= 2 else None
                mk1 = markov1_score(c1, t1, V, p1, cnames)
                mk2 = markov2_backoff_score(c12, t12, c1, t1, V, p2, p1,
                                            cnames, lam=args.lambda_backoff)
                final_m1 = alpha * ret_n + (1 - alpha) * minmax(mk1)
                final_m2 = alpha * ret_n + (1 - alpha) * minmax(mk2)
            order1 = np.argsort(-final_m1)
            order2 = np.argsort(-final_m2)
            ranked1 = [cnames[j] for j in order1]
            ranked2 = [cnames[j] for j in order2]
            for k in KS:
                if gold in ranked1[:k]:
                    h_m1[k] += 1
                if gold in ranked2[:k]:
                    h_m2[k] += 1
        top3_m1 = h_m1[3] / n
        top3_m2 = h_m2[3] / n
        delta = top3_m2 - top3_m1
        print(f"  {alpha:>5.2f}  {top3_m1:>8.1%}   {top3_m2:>8.1%}   {delta:+6.2%}")
        if top3_m1 > best["M1"][1]:
            best["M1"] = (alpha, top3_m1, h_m1)
        if top3_m2 > best["M2"][1]:
            best["M2"] = (alpha, top3_m2, h_m2)

    print(f"\nBest M1: alpha={best['M1'][0]:.2f}, top-3={best['M1'][1]:.1%}")
    h = best["M1"][2]
    print(f"        top-1={h[1]/n:.1%}  top-5={h[5]/n:.1%}  top-10={h[10]/n:.1%}")
    print(f"Best M2: alpha={best['M2'][0]:.2f}, top-3={best['M2'][1]:.1%}")
    h = best["M2"][2]
    print(f"        top-1={h[1]/n:.1%}  top-5={h[5]/n:.1%}  top-10={h[10]/n:.1%}")
    print(f"Delta (M2-M1): {(best['M2'][1] - best['M1'][1])*100:+.2f}pp top-3")

    # source x position table at best M2 alpha — to see where the gain lands.
    ba = best["M2"][0]
    print(f"\n=== M2 backoff at alpha={ba:.2f}, K={K}: source × position ===")
    hits = defaultdict(lambda: {k: 0 for k in KS})
    tot = defaultdict(int)
    for i in test_idx:
        cids = cand_idx[i, :K]
        mask = cids >= 0
        cids_v = cids[mask]
        if len(cids_v) == 0:
            continue
        cnames = [vocab[c] for c in cids_v]
        gold = vocab[gold_idx[i]]
        ret = cand_scores[i, : len(cids_v)]
        ret_n = minmax(ret)
        hist = [normalize(x) for x in triplets[i]["history"]]
        if not hist:
            final = ret_n.copy()
        else:
            p1 = hist[-1]
            p2 = hist[-2] if len(hist) >= 2 else None
            mk2 = markov2_backoff_score(c12, t12, c1, t1, V, p2, p1,
                                        cnames, lam=args.lambda_backoff)
            final = ba * ret_n + (1 - ba) * minmax(mk2)
        order = np.argsort(-final)
        ranked = [cnames[j] for j in order]
        tt = triplets[i]
        keys = [(tt["source"], bucket_pos(tt)), ("ALL", bucket_pos(tt))]
        for k in KS:
            if gold in ranked[:k]:
                for key in keys:
                    hits[key][k] += 1
        for key in keys:
            tot[key] += 1
    print(f"\n{'source':<32}{'pos':<8}{'n':>7}  " + "  ".join(f"top-{k:>2}" for k in KS))
    for (src, pos), cnt in sorted(tot.items()):
        if src == "ALL":
            continue
        h = hits[(src, pos)]
        print(f"{src:<32}{pos:<8}{cnt:>7}  " + "  ".join(f"{h[k]/cnt:>5.1%}" for k in KS))
    print(f"\n{'ALL':<32}{'pos':<8}{'n':>7}  " + "  ".join(f"top-{k:>2}" for k in KS))
    for (src, pos), cnt in sorted(tot.items()):
        if src != "ALL":
            continue
        h = hits[(src, pos)]
        print(f"{src:<32}{pos:<8}{cnt:>7}  " + "  ".join(f"{h[k]/cnt:>5.1%}" for k in KS))

    # Subset eval: only on rows where history>=2 AND (p2,p1) seen in train
    print(f"\n=== Subset: only test rows with bigram seen in train (n={have_bigram_match}) ===")
    sub_test = []
    for i in test_idx:
        h = [normalize(x) for x in triplets[i]["history"]]
        if len(h) >= 2 and (h[-2], h[-1]) in t12:
            sub_test.append(i)
    sub_n = len(sub_test)
    if sub_n > 0:
        for alpha in (best["M1"][0], best["M2"][0]):
            for system in ("M1", "M2"):
                h_sys = {k: 0 for k in KS}
                for i in sub_test:
                    cids = cand_idx[i, :K]
                    mask = cids >= 0
                    cids_v = cids[mask]
                    if len(cids_v) == 0:
                        continue
                    cnames = [vocab[c] for c in cids_v]
                    gold = vocab[gold_idx[i]]
                    ret = cand_scores[i, : len(cids_v)]
                    ret_n = minmax(ret)
                    hist = [normalize(x) for x in triplets[i]["history"]]
                    p1, p2 = hist[-1], hist[-2]
                    if system == "M1":
                        mk = markov1_score(c1, t1, V, p1, cnames)
                    else:
                        mk = markov2_backoff_score(c12, t12, c1, t1, V, p2, p1,
                                                   cnames, lam=args.lambda_backoff)
                    final = alpha * ret_n + (1 - alpha) * minmax(mk)
                    order = np.argsort(-final)
                    ranked = [cnames[j] for j in order]
                    for k in KS:
                        if gold in ranked[:k]:
                            h_sys[k] += 1
                row = "  ".join(f"top-{k}={h_sys[k]/sub_n:.1%}" for k in KS)
                print(f"  alpha={alpha:.2f} {system}: {row}")


if __name__ == "__main__":
    main()

"""Eval the MLP rerank on the same test set as `eval_next_tool_markov.py`.

Loads the MLP weights saved by `train_next_tool_mlp.py`, scores each of
the cached top-50 candidates per test triplet, and reports top-1/3/5/10
overall and stratified by (source, position). Includes head-to-head
comparison against:
  - retrieval-only (alpha=1.0 in the Markov script)
  - Markov-1 best alpha=0.4 (from session 22)
  - MLP+Markov mix at a few blend weights

Pure numpy inference — no torch needed at this stage.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRIPLETS = ROOT / "data" / "next_tool_triplets.jsonl"
CACHE = ROOT / "data" / "cache" / "next_tool"
HYBRID_DIR = ROOT / "models" / "baseline-v1-desc-hybrid"
ENC_CENTROIDS = HYBRID_DIR / "encoder_centroids.npy"
MARKOV_NPZ = HYBRID_DIR / "markov_counts.npz"
MARKOV_VOCAB = HYBRID_DIR / "markov_vocab.txt"
MLP_NPZ = HYBRID_DIR / "mlp_rerank.npz"

SEED = 17
KS = (1, 3, 5, 10)
BLEND_GRID = (1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0)


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


def load_mlp(path: Path):
    npz = np.load(path)
    return {
        "W1": npz["fc1_weight"].astype(np.float32),  # [H, in]
        "b1": npz["fc1_bias"].astype(np.float32),  # [H]
        "W2": npz["fc2_weight"].astype(np.float32),  # [1, H]
        "b2": npz["fc2_bias"].astype(np.float32),  # [1]
    }


def mlp_score(X: np.ndarray, w) -> np.ndarray:
    """X: [B, in] -> [B] logit. Pure numpy ReLU MLP."""
    h = X @ w["W1"].T + w["b1"]
    np.maximum(h, 0, out=h)
    return (h @ w["W2"].T + w["b2"]).reshape(-1)


def load_markov():
    """Same loader pattern as `Router.from_pretrained` for the markov table."""
    import scipy.sparse as sp

    counts = sp.load_npz(MARKOV_NPZ).tocsr()
    vocab = MARKOV_VOCAB.read_text().splitlines()
    name_to_mk_idx = {n: i for i, n in enumerate(vocab)}
    totals = np.asarray(counts.sum(axis=1)).reshape(-1)
    V = counts.shape[0]
    return counts, totals, V, name_to_mk_idx


def markov_score_row(counts, totals, V, name_to_mk, prev_name: str, cand_names: list[str]):
    """add-one smoothed P(cand | prev) for a list of candidates."""
    prev_idx = name_to_mk.get(prev_name)
    out = np.empty(len(cand_names), dtype=np.float32)
    if prev_idx is None or totals[prev_idx] == 0:
        out[:] = 1.0 / max(V, 1)
        return out
    row = counts.getrow(prev_idx)
    row_dense = np.asarray(row.todense()).reshape(-1)
    denom = totals[prev_idx] + V
    for j, cn in enumerate(cand_names):
        ci = name_to_mk.get(cn)
        c = row_dense[ci] if ci is not None else 0
        out[j] = (c + 1.0) / denom
    return out


def minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


def bucket(t):
    p = t["position"]
    pos = "t=1" if p == 1 else ("t=2" if p == 2 else "t>=3")
    return (t["source"], pos)


def build_features_batch(
    query_emb: np.ndarray,  # [D]
    prev_emb: np.ndarray,  # [D]
    cand_embs: np.ndarray,  # [K, D]
    ret_scores: np.ndarray,  # [K]
) -> np.ndarray:
    K = cand_embs.shape[0]
    D = cand_embs.shape[1]
    q_dot_c = (query_emb[None, :] * cand_embs).sum(axis=1, keepdims=True)
    p_dot_c = (prev_emb[None, :] * cand_embs).sum(axis=1, keepdims=True)
    return np.concatenate(
        [
            np.tile(query_emb[None, :], (K, 1)),
            np.tile(prev_emb[None, :], (K, 1)),
            cand_embs,
            q_dot_c,
            p_dot_c,
            ret_scores.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)


def print_table(name, hits, totals_b, n):
    print(f"\n=== {name} ===")
    overall = hits[("ALL", "overall")]
    print(f"OVERALL n={n}  " + "  ".join(f"top-{k}={overall[k]/n:.1%}" for k in KS))
    print(f"\n{'source':<32}{'pos':<8}{'n':>7}  " + "  ".join(f"top-{k:>2}" for k in KS))
    for (src, pos), cnt in sorted(totals_b.items()):
        if src == "ALL":
            continue
        h = hits[(src, pos)]
        print(f"{src:<32}{pos:<8}{cnt:>7}  " + "  ".join(f"{h[k]/cnt:>5.1%}" for k in KS))
    print(f"\n{'ALL':<32}{'pos':<8}{'n':>7}  " + "  ".join(f"top-{k:>2}" for k in KS))
    for (src, pos), cnt in sorted(totals_b.items()):
        if src != "ALL":
            continue
        h = hits[(src, pos)]
        print(f"{src:<32}{pos:<8}{cnt:>7}  " + "  ".join(f"{h[k]/cnt:>5.1%}" for k in KS))


def main() -> None:
    triplets = [json.loads(line) for line in TRIPLETS.open()]
    query_embs = np.load(CACHE / "query_embs.npy")
    cand_idx = np.load(CACHE / "cand_idx.npy")
    cand_scores = np.load(CACHE / "cand_scores.npy")
    prev_idx = np.load(CACHE / "prev_idx.npy")
    gold_idx = np.load(CACHE / "gold_idx.npy")
    enc_centroids = np.load(ENC_CENTROIDS)
    D = enc_centroids.shape[1]
    zero_emb = np.zeros((D,), dtype=np.float32)

    # Catalog vocab for name lookup (router.vocab order matches cand_idx)
    from agent_tool_router import Router

    router = Router.from_pretrained("baseline-v1-desc-hybrid")
    vocab = [normalize(v) for v in router.vocab]

    _, test_idx = split_by_trace(triplets)
    n = len(test_idx)
    print(f"Test triplets: {n}")

    mlp = load_mlp(MLP_NPZ)
    print(f"MLP loaded: W1{mlp['W1'].shape}, W2{mlp['W2'].shape}")

    counts, totals, V, name_to_mk = load_markov()
    print(f"Markov-1 loaded: V={V}")

    def eval_scoring(score_fn, label: str):
        hits = defaultdict(lambda: {k: 0 for k in KS})
        totals_b = defaultdict(int)
        for i in test_idx:
            cids = cand_idx[i]  # [50]
            mask = cids >= 0
            if not mask.any():
                continue
            cids_v = cids[mask]
            cnames = [vocab[c] for c in cids_v]
            prev_name = vocab[prev_idx[i]] if prev_idx[i] >= 0 else ""
            gold = vocab[gold_idx[i]]
            scores = score_fn(i, cids_v, cnames, prev_name)
            order = np.argsort(-scores)
            ranked = [cnames[j] for j in order]
            t = triplets[i]
            keys = [bucket(t), ("ALL", bucket(t)[1]), ("ALL", "overall")]
            for k in KS:
                if gold in ranked[:k]:
                    for key in keys:
                        hits[key][k] += 1
            for key in keys:
                totals_b[key] += 1
        return hits, totals_b

    # --- 1. retrieval-only baseline ---
    def s_retrieval(i, cids_v, cnames, prev_name):
        return cand_scores[i, : len(cids_v)]

    hits_r, totals_r = eval_scoring(s_retrieval, "retrieval-only")
    print_table("Retrieval-only (alpha=1.0)", hits_r, totals_r, n)

    # --- 2. Markov-1 best alpha=0.4 (session 22) ---
    MK_ALPHA = 0.4

    def s_markov(i, cids_v, cnames, prev_name):
        ret = cand_scores[i, : len(cids_v)]
        ret_n = minmax(ret)
        mk = markov_score_row(counts, totals, V, name_to_mk, prev_name, cnames)
        mk_n = minmax(mk)
        return MK_ALPHA * ret_n + (1 - MK_ALPHA) * mk_n

    hits_m, totals_m = eval_scoring(s_markov, "markov-1")
    print_table(f"Markov-1 rerank (alpha={MK_ALPHA})", hits_m, totals_m, n)

    # --- 3. MLP rerank ---
    def s_mlp(i, cids_v, cnames, prev_name):
        q = query_embs[i]
        p = enc_centroids[prev_idx[i]] if prev_idx[i] >= 0 else zero_emb
        ce = enc_centroids[cids_v]
        rs = cand_scores[i, : len(cids_v)]
        X = build_features_batch(q, p, ce, rs)
        return mlp_score(X, mlp)

    hits_x, totals_x = eval_scoring(s_mlp, "mlp")
    print_table("MLP rerank (pure)", hits_x, totals_x, n)

    # --- 4. MLP + Markov-1 blend ---
    print("\n=== MLP+Markov-1 blend sweep (top-3 overall) ===")
    print(f"  {'beta_mlp':>8}  top-1   top-3   top-5   top-10")
    for beta in BLEND_GRID:
        hits_b = defaultdict(lambda: {k: 0 for k in KS})
        totals_bx = defaultdict(int)
        for i in test_idx:
            cids = cand_idx[i]
            mask = cids >= 0
            if not mask.any():
                continue
            cids_v = cids[mask]
            cnames = [vocab[c] for c in cids_v]
            prev_name = vocab[prev_idx[i]] if prev_idx[i] >= 0 else ""
            gold = vocab[gold_idx[i]]
            # MLP score (already a logit)
            q = query_embs[i]
            p = enc_centroids[prev_idx[i]] if prev_idx[i] >= 0 else zero_emb
            ce = enc_centroids[cids_v]
            rs = cand_scores[i, : len(cids_v)]
            X = build_features_batch(q, p, ce, rs)
            mlp_s = mlp_score(X, mlp)
            mlp_n = minmax(mlp_s)
            mk = markov_score_row(counts, totals, V, name_to_mk, prev_name, cnames)
            mk_n = minmax(mk)
            final = beta * mlp_n + (1 - beta) * mk_n
            order = np.argsort(-final)
            ranked = [cnames[j] for j in order]
            t = triplets[i]
            keys = [bucket(t), ("ALL", bucket(t)[1]), ("ALL", "overall")]
            for k in KS:
                if gold in ranked[:k]:
                    for key in keys:
                        hits_b[key][k] += 1
            for key in keys:
                totals_bx[key] += 1
        ov = hits_b[("ALL", "overall")]
        print(
            f"  {beta:>8.2f}  "
            + "  ".join(f"{ov[k]/n:>5.1%}" for k in KS)
        )

    # --- summary head-to-head ---
    print("\n=== Head-to-head overall top-K (n={}) ===".format(n))
    print(f"  {'system':<28}  top-1   top-3   top-5   top-10")
    for name, h in [
        ("retrieval-only", hits_r),
        ("markov-1 alpha=0.4", hits_m),
        ("mlp (pure)", hits_x),
    ]:
        ov = h[("ALL", "overall")]
        print(f"  {name:<28}  " + "  ".join(f"{ov[k]/n:>5.1%}" for k in KS))


if __name__ == "__main__":
    main()

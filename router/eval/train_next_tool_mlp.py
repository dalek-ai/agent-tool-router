"""Learned MLP rerank — ceiling exploration vs Markov-1.

Trains a small MLP that scores a candidate tool given (query_emb,
prev_history_emb, candidate_emb, retrieval_score, q.cand_cos, prev.cand_cos).
Negatives are sampled from the same top-50 retrieval list as the gold
positive; if the gold isn't in top-50, the triplet is skipped at training
time (it would be unrankable anyway).

Output: `models/baseline-v1-desc-hybrid/mlp_rerank.npz` — numpy-only weights
so the rerank can run at inference without torch.

Same split as `eval_next_tool_markov.py`: trace_id 80/20, seed=17.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
TRIPLETS = ROOT / "data" / "next_tool_triplets.jsonl"
CACHE = ROOT / "data" / "cache" / "next_tool"
ENC_CENTROIDS = ROOT / "models" / "baseline-v1-desc-hybrid" / "encoder_centroids.npy"
OUT = ROOT / "models" / "baseline-v1-desc-hybrid" / "mlp_rerank.npz"

SEED = 17
HIDDEN = 128
NEG_PER_POS = 4
EPOCHS = 8
BATCH = 256
LR = 1e-3
DROPOUT = 0.1


def split_by_trace(triplets, frac=0.8):
    by_trace = defaultdict(list)
    for i, t in enumerate(triplets):
        by_trace[t["trace_id"]].append(i)
    trace_ids = sorted(by_trace)
    rng = random.Random(SEED)
    rng.shuffle(trace_ids)
    cut = int(len(trace_ids) * frac)
    train_idx = [i for tid in trace_ids[:cut] for i in by_trace[tid]]
    test_idx = [i for tid in trace_ids[cut:] for i in by_trace[tid]]
    return train_idx, test_idx


class RerankMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x)))).squeeze(-1)


def build_features(
    query_emb: np.ndarray,
    prev_emb: np.ndarray,
    cand_emb: np.ndarray,
    retrieval_score: np.ndarray,
) -> np.ndarray:
    """All inputs are (B, D) or (B,) scalars. Returns (B, 3*D + 3)."""
    q_dot_c = (query_emb * cand_emb).sum(axis=1, keepdims=True)
    p_dot_c = (prev_emb * cand_emb).sum(axis=1, keepdims=True)
    rs = retrieval_score.reshape(-1, 1).astype(np.float32)
    return np.concatenate(
        [query_emb, prev_emb, cand_emb, q_dot_c, p_dot_c, rs], axis=1
    ).astype(np.float32)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    triplets = [json.loads(line) for line in TRIPLETS.open()]
    query_embs = np.load(CACHE / "query_embs.npy")  # [N, 384]
    cand_idx = np.load(CACHE / "cand_idx.npy")  # [N, 50]
    cand_scores = np.load(CACHE / "cand_scores.npy")  # [N, 50]
    prev_idx = np.load(CACHE / "prev_idx.npy")  # [N]
    gold_idx = np.load(CACHE / "gold_idx.npy")  # [N]
    enc_centroids = np.load(ENC_CENTROIDS)  # [V, 384] L2-normalized

    D = query_embs.shape[1]
    zero_emb = np.zeros((D,), dtype=np.float32)

    train_idx, test_idx = split_by_trace(triplets)
    print(f"Triplets: train={len(train_idx)} test={len(test_idx)}, dim={D}")

    # Keep only train triplets where the gold IS in the top-50 retrieval list
    # (otherwise the rerank has no chance — the candidate isn't there).
    keep_train = []
    gold_positions_in_top50 = []  # for diagnostic
    for i in train_idx:
        match = np.where(cand_idx[i] == gold_idx[i])[0]
        if len(match) > 0:
            keep_train.append(i)
            gold_positions_in_top50.append(int(match[0]))
    print(f"Trainable train triplets (gold in top-50): {len(keep_train)}/{len(train_idx)}")
    print(
        "Gold position distribution in top-50 (train, hit only): "
        f"mean={np.mean(gold_positions_in_top50):.1f} median={np.median(gold_positions_in_top50):.1f}"
    )

    # Materialize the training pairs in memory: each is one row (pos or neg)
    # with a binary label. Roughly (NEG_PER_POS+1)*len(keep_train) ~ 25K rows
    # times (3*384+3) floats = ~120 MB, fine.
    rng = np.random.RandomState(SEED)
    rows_X = []
    rows_y = []
    for i in keep_train:
        q = query_embs[i]
        p = enc_centroids[prev_idx[i]] if prev_idx[i] >= 0 else zero_emb
        g = gold_idx[i]
        # find gold position in top-50
        g_pos = np.where(cand_idx[i] == g)[0][0]
        g_score = cand_scores[i, g_pos]
        # positive: gold
        rows_X.append(build_features(q[None], p[None], enc_centroids[g][None], np.array([g_score])))
        rows_y.append(1.0)
        # negatives: random from top-50 minus gold
        neg_pool = [j for j in range(cand_idx.shape[1]) if cand_idx[i, j] != g and cand_idx[i, j] >= 0]
        if len(neg_pool) < NEG_PER_POS:
            continue
        chosen = rng.choice(neg_pool, size=NEG_PER_POS, replace=False)
        for j in chosen:
            c_idx = cand_idx[i, j]
            rows_X.append(
                build_features(q[None], p[None], enc_centroids[c_idx][None], np.array([cand_scores[i, j]]))
            )
            rows_y.append(0.0)

    X = np.concatenate(rows_X, axis=0)
    y = np.array(rows_y, dtype=np.float32)
    print(f"Training matrix: X={X.shape} y={y.shape} (pos rate={y.mean():.3f})")

    # Train
    model = RerankMLP(in_dim=X.shape[1]).cpu()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    N = X.shape[0]

    for ep in range(EPOCHS):
        perm = torch.randperm(N)
        total = 0.0
        nb = 0
        t0 = time()
        model.train()
        for s in range(0, N, BATCH):
            idx = perm[s : s + BATCH]
            xb = Xt[idx]
            yb = yt[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            nb += xb.size(0)
        print(f"  ep {ep + 1}/{EPOCHS}  loss={total/nb:.4f}  ({time()-t0:.1f}s)")

    # Export to numpy
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        fc1_weight=state["fc1.weight"],
        fc1_bias=state["fc1.bias"],
        fc2_weight=state["fc2.weight"],
        fc2_bias=state["fc2.bias"],
        in_dim=np.int32(X.shape[1]),
        hidden=np.int32(HIDDEN),
    )
    print(f"\nSaved {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()

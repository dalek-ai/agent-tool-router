"""Serialize a Markov-2 (history bigram) transition table into a pretrained
model directory.

Reads ``data/next_tool_triplets.jsonl`` and writes, into the target model
directory:

  - ``markov2_counts.npz`` : CSR sparse [N_bigrams, V_markov] of next counts
                             per bigram key. Row i is the next-tool count
                             vector for the (p2, p1) pair stored at
                             ``markov2_keys[i]``.
  - ``markov2_keys.npy``   : int32 array [N_bigrams, 2] of
                             (p2_idx, p1_idx) using the existing
                             ``markov_vocab.txt`` index.

Reuses ``markov_vocab.txt`` from the Markov-1 table that should already be
written into the same directory (run ``build_markov_table.py`` first). At
query time the Router applies stupid backoff (Brants 2007):

  score(c | p2, p1) = count(p2, p1, c) / count(p2, p1)             if seen
                    = lambda * count(p1, c) / count(p1)            backoff to M1
                    = lambda^2 / V                                 backoff to uniform

The bigram table is built over the full dataset (no train/test split): for
a shipped model the user wants the richest prior available, not a held-out
estimate. The 80/20 split in ``eval_next_tool_markov2.py`` exists to prove
the lift on unseen traces, which justifies shipping the table at all.

Usage:
  python -m router.eval.build_markov2_table \\
      --triplets data/next_tool_triplets.jsonl \\
      --out models/baseline-v1-desc-hybrid-next-v1
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def normalize(name: str) -> str:
    return (name or "").strip().lower()


def build(triplets_path: Path, vocab_path: Path) -> tuple[sp.csr_matrix, np.ndarray, int]:
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"{vocab_path} not found — run build_markov_table.py first so the "
            f"Markov-2 bigram keys can reuse the same vocab indices."
        )
    vocab = vocab_path.read_text(encoding="utf-8").splitlines()
    idx = {name: i for i, name in enumerate(vocab)}
    V = len(vocab)

    bigram_counts: dict[tuple[int, int], dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    n_used = 0
    n_total = 0
    n_skipped_oov = 0
    with triplets_path.open() as f:
        for line in f:
            t = json.loads(line)
            n_total += 1
            h = [normalize(x) for x in (t.get("history") or [])]
            if len(h) < 2:
                continue
            nxt = normalize(t["next_tool"])
            p2, p1 = h[-2], h[-1]
            p2_idx = idx.get(p2)
            p1_idx = idx.get(p1)
            nxt_idx = idx.get(nxt)
            if p2_idx is None or p1_idx is None or nxt_idx is None:
                n_skipped_oov += 1
                continue
            bigram_counts[(p2_idx, p1_idx)][nxt_idx] += 1
            n_used += 1

    keys = sorted(bigram_counts)
    N = len(keys)
    rows, cols, data = [], [], []
    for row_i, key in enumerate(keys):
        for nxt_idx, c in bigram_counts[key].items():
            rows.append(row_i)
            cols.append(nxt_idx)
            data.append(c)
    mat = sp.csr_matrix(
        (np.array(data, dtype=np.int32),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(N, V),
    )
    keys_arr = np.asarray(keys, dtype=np.int32) if N > 0 else np.zeros((0, 2), dtype=np.int32)
    print(f"Triplets total       : {n_total}")
    print(f"Triplets with len>=2 : {n_used}  ({n_used/max(n_total,1):.1%})")
    print(f"Skipped (OOV vocab)  : {n_skipped_oov}")
    print(f"Unique (p2, p1) keys : {N}")
    print(f"Non-zero next cells  : {mat.nnz}")
    return mat, keys_arr, V


def patch_config(out_dir: Path, lam: float) -> None:
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg["markov2_lambda"] = float(lam)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Patched config.json: markov2_lambda={lam}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--triplets",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "next_tool_triplets.jsonl",
    )
    parser.add_argument("--out", type=Path, required=True,
                        help="Pretrained model directory to write into.")
    parser.add_argument("--lambda-backoff", type=float, default=0.4,
                        help="Stupid backoff penalty (Brants 2007 = 0.4).")
    args = parser.parse_args()

    mat, keys, V = build(args.triplets, args.out / "markov_vocab.txt")
    args.out.mkdir(parents=True, exist_ok=True)
    sp.save_npz(args.out / "markov2_counts.npz", mat)
    np.save(args.out / "markov2_keys.npy", keys)
    print(f"Wrote {args.out / 'markov2_counts.npz'}  ({mat.shape}, nnz={mat.nnz})")
    print(f"Wrote {args.out / 'markov2_keys.npy'}    ({keys.shape})")
    patch_config(args.out, args.lambda_backoff)


if __name__ == "__main__":
    main()

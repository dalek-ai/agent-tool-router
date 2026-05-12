"""Serialize a Markov-1 transition table into a pretrained model directory.

Reads ``data/next_tool_triplets.jsonl`` and writes, into the target model
directory:

  - ``markov_counts.npz``  : CSR sparse [V, V] of co-occurrence counts
                             (rows: prev tool, cols: next tool)
  - ``markov_vocab.txt``   : ``V`` normalized tool names, row/col aligned

The Router loads these alongside the centroid artifacts; the smoothing
formula ``(counts[prev][next] + 1) / (totals[prev] + V)`` is applied at
query time, so we only need raw counts on disk.

Counts are built over the full dataset (no train/test split): for a
shipped model the user wants the richest prior available, not a held-out
estimate. The 80/20 split in eval_next_tool_markov.py exists to prove
the lift on unseen traces, which justifies shipping the table at all.

Usage:
  python -m router.eval.build_markov_table \\
      --triplets data/next_tool_triplets.jsonl \\
      --out models/baseline-v1-desc-hybrid
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


def build(triplets_path: Path) -> tuple[sp.csr_matrix, list[str]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    vocab: set[str] = set()
    n = 0
    with triplets_path.open() as f:
        for line in f:
            t = json.loads(line)
            if not t["history"]:
                continue
            prev = normalize(t["history"][-1])
            nxt = normalize(t["next_tool"])
            if not prev or not nxt:
                continue
            counts[prev][nxt] += 1
            vocab.add(prev)
            vocab.add(nxt)
            n += 1
    vocab_list = sorted(vocab)
    idx = {name: i for i, name in enumerate(vocab_list)}
    V = len(vocab_list)
    rows, cols, data = [], [], []
    for prev, row in counts.items():
        i = idx[prev]
        for nxt, c in row.items():
            rows.append(i)
            cols.append(idx[nxt])
            data.append(c)
    mat = sp.csr_matrix(
        (np.array(data, dtype=np.int32),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(V, V),
    )
    print(f"Triplets used: {n}")
    print(f"Markov vocab : {V}")
    print(f"Non-zero cells: {mat.nnz} (density {mat.nnz / max(V * V, 1):.4%})")
    return mat, vocab_list


def patch_config(out_dir: Path, default_alpha: float) -> None:
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg["markov_alpha"] = float(default_alpha)
    cfg["markov_rerank_n"] = 50
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Patched config.json: markov_alpha={default_alpha}, markov_rerank_n=50")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--triplets",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "next_tool_triplets.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Pretrained model directory to write into.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Default markov_alpha (best from sweep, 2026-05-12).",
    )
    args = parser.parse_args()

    mat, vocab = build(args.triplets)
    args.out.mkdir(parents=True, exist_ok=True)
    sp.save_npz(args.out / "markov_counts.npz", mat)
    (args.out / "markov_vocab.txt").write_text(
        "\n".join(vocab), encoding="utf-8",
    )
    print(f"Wrote {args.out / 'markov_counts.npz'}")
    print(f"Wrote {args.out / 'markov_vocab.txt'}")
    patch_config(args.out, args.alpha)


if __name__ == "__main__":
    main()

"""Fine-tune MiniLM-L6 on next-tool triplets to push the recall@K ceiling.

Session 25 finding: recall@200 = 69.6% is the mechanical ceiling for any
top-200 rerank. Markov-1 already saturates 99% of it. To go past 54.9%
top-3, the retriever itself must improve — that means moving the encoder.

This script fine-tunes `sentence-transformers/all-MiniLM-L6-v2` with
MultipleNegativesRankingLoss on (task_text, gold_description, hard_neg)
triples drawn from the 8386 train triplets (trace_id split, seed=17,
identical to all other next-tool eval scripts).

Hard negatives are sampled from the cached top-50 retrieval of the
**current** baseline-v1-desc-hybrid hybrid encoder, excluding the gold.
This is the standard ANCE-style hard-negative loop applied once.

Output: a fine-tuned encoder under `models/_finetune/minilm-next-v1/`,
ready to be plugged into train_descriptions.py via --encoder-model to
produce a new hybrid pretrained.
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
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRIPLETS = ROOT / "data" / "next_tool_triplets.jsonl"
DESCS = ROOT / "data" / "tool_descriptions.jsonl"
CACHE = ROOT / "data" / "cache" / "next_tool"

DEFAULT_BASE = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUT = ROOT / "models" / "_finetune" / "minilm-next-v1"
SEED = 17
HARD_NEGS_PER_POS = 1
EPOCHS = 2
BATCH = 64
LR = 2e-5
MAX_LEN = 128
DESC_MAX_CHARS = 600  # truncate very long descriptions so the encoder
                     # spends its budget on the discriminative prefix.


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-model",
        default=DEFAULT_BASE,
        help="HF id or local path of the encoder to fine-tune (default: MiniLM-L6).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Where to save the fine-tuned encoder (default: models/_finetune/minilm-next-v1).",
    )
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    args = ap.parse_args()

    OUT = args.out
    BASE = args.base_model
    max_len = args.max_len
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    triplets = [json.loads(line) for line in TRIPLETS.open()]
    descs_by_name: dict[str, str] = {}
    for line in DESCS.open():
        d = json.loads(line)
        descs_by_name[normalize(d["name"])] = (d.get("description") or "")[:DESC_MAX_CHARS]
    print(f"  triplets: {len(triplets)}, descriptions: {len(descs_by_name)}")

    cand_idx = np.load(CACHE / "cand_idx.npy")
    gold_idx = np.load(CACHE / "gold_idx.npy")
    meta = json.loads((CACHE / "meta.json").read_text())

    # Cache vocab from the current router so cand_idx maps to names.
    from agent_tool_router import Router
    router = Router.from_pretrained("baseline-v1-desc-hybrid")
    vocab = [normalize(v) for v in router.vocab]
    assert len(vocab) == meta["vocab_size"]

    train_idx, test_idx = split_by_trace(triplets)
    print(f"  train: {len(train_idx)}  test: {len(test_idx)}")

    # Build training examples: (task_text, gold_desc, hard_neg_desc)
    # MultipleNegativesRankingLoss expects [anchor, pos, neg], and adds
    # in-batch negatives on top of the explicit hard neg.
    rng = random.Random(SEED)
    examples: list[InputExample] = []
    n_skipped_no_gold_desc = 0
    n_skipped_no_neg = 0
    for i in train_idx:
        t = triplets[i]
        gold_name = normalize(t["next_tool"])
        gold_desc = descs_by_name.get(gold_name)
        if not gold_desc:
            n_skipped_no_gold_desc += 1
            continue
        task = (t["task_text"] or "").strip()
        if not task:
            continue
        cids = cand_idx[i]
        cids = cids[cids >= 0]
        # Pick hard negs from top-50 candidates that aren't the gold.
        # If gold is in top-50, drop it; if not, use the top-50 freely.
        candidates = []
        for c in cids[:50]:
            cname = vocab[c]
            if cname == gold_name:
                continue
            cdesc = descs_by_name.get(cname)
            if cdesc:
                candidates.append(cdesc)
        if not candidates:
            n_skipped_no_neg += 1
            continue
        for _ in range(HARD_NEGS_PER_POS):
            neg_desc = rng.choice(candidates)
            examples.append(InputExample(texts=[task, gold_desc, neg_desc]))

    print(
        f"\nTrain examples: {len(examples)}"
        f"  (skipped no-gold-desc: {n_skipped_no_gold_desc},"
        f" no-neg-in-top50: {n_skipped_no_neg})"
    )

    # Device pick.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading base encoder: {BASE}")
    model = SentenceTransformer(BASE, device=device)
    model.max_seq_length = max_len

    train_dl = DataLoader(examples, shuffle=True, batch_size=args.batch)
    loss = losses.MultipleNegativesRankingLoss(model)

    print(f"\nFine-tuning ({args.epochs} epochs, batch={args.batch}, lr={args.lr}, max_len={max_len})...")
    t0 = time()
    model.fit(
        train_objectives=[(train_dl, loss)],
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(train_dl) * args.epochs),
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=str(OUT),
    )
    print(f"\nDone in {time() - t0:.1f}s")
    print(f"Encoder saved to {OUT}")
    print(
        "\nNext: re-train hybrid retriever with this encoder:\n"
        f"  python -m agent_tool_router.train_descriptions "
        f"--backend hybrid --alpha 0.5 "
        f"--encoder-model {OUT} "
        f"--out models/baseline-v1-desc-hybrid-next-v1"
    )


if __name__ == "__main__":
    main()

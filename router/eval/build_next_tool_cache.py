"""Cache embeddings + top-50 retrieval candidates for all next-tool triplets.

Shared input for `eval_next_tool_markov.py`, `train_next_tool_mlp.py`, and
`eval_next_tool_mlp.py`. Running the retrieval batch once and saving avoids
re-paying the SentenceTransformer encode + cosine cost for every experiment.

Outputs (under `data/cache/next_tool/`):
  - query_embs.npy        [N, 384] float32, L2-normalized task_text encodings
  - cand_idx.npy          [N, 50] int32, indices into router.vocab
  - cand_scores.npy       [N, 50] float32, retrieval scores (hybrid backend)
  - prev_idx.npy          [N] int32, index of history[-1] in router.vocab
                           (or -1 if unseen — should be rare since the
                           triplet builder filters gold against the catalog,
                           but history tools aren't guaranteed to be in it)
  - gold_idx.npy          [N] int32, index of next_tool in router.vocab
                           (-1 if absent — also rare)
  - meta.json             N, vocab name, position split

Triplet order matches `data/next_tool_triplets.jsonl` line order so any
downstream split can re-derive the same trace_id 80/20 with seed=17.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import time

import numpy as np

from agent_tool_router import Router

ROOT = Path(__file__).resolve().parents[2]
TRIPLETS = ROOT / "data" / "next_tool_triplets.jsonl"
CACHE_DIR = ROOT / "data" / "cache" / "next_tool"
MODEL = "baseline-v1-desc-hybrid"
TOP_N = 50
BATCH = 256


def normalize(name: str) -> str:
    return (name or "").strip().lower()


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading router '{MODEL}'...")
    router = Router.from_pretrained(MODEL)
    name_to_idx = {normalize(v): i for i, v in enumerate(router.vocab)}

    triplets = [json.loads(line) for line in TRIPLETS.open()]
    N = len(triplets)
    print(f"Triplets: {N}")

    print("Encoding task_texts...")
    # Force-load the encoder by calling route() once; then reuse it directly.
    _ = router.route("warmup", k=1)
    assert router.encoder_model is not None
    tasks = [t["task_text"] or "" for t in triplets]
    t0 = time()
    query_embs = router.encoder_model.encode(
        tasks,
        batch_size=BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  encoded in {time() - t0:.1f}s, shape={query_embs.shape}")

    print(f"\nRouting top-{TOP_N} for each triplet (hybrid backend)...")
    cand_idx = np.full((N, TOP_N), -1, dtype=np.int32)
    cand_scores = np.zeros((N, TOP_N), dtype=np.float32)
    t0 = time()
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        batch_tasks = tasks[start:end]
        results = router.route(batch_tasks, k=TOP_N, return_scores=True)
        for i, res in enumerate(results):
            for j, r in enumerate(res):
                idx = name_to_idx.get(normalize(r.tool))
                if idx is not None:
                    cand_idx[start + i, j] = idx
                cand_scores[start + i, j] = r.score
        if (start // BATCH) % 5 == 0:
            print(f"  {end}/{N} ({time() - t0:.1f}s)")
    print(f"Routed in {time() - t0:.1f}s")

    prev_idx = np.full(N, -1, dtype=np.int32)
    gold_idx = np.full(N, -1, dtype=np.int32)
    prev_unseen = 0
    gold_unseen = 0
    for i, t in enumerate(triplets):
        if t["history"]:
            pi = name_to_idx.get(normalize(t["history"][-1]))
            if pi is None:
                prev_unseen += 1
            else:
                prev_idx[i] = pi
        gi = name_to_idx.get(normalize(t["next_tool"]))
        if gi is None:
            gold_unseen += 1
        else:
            gold_idx[i] = gi

    print(f"prev tool not in catalog: {prev_unseen}/{N}")
    print(f"gold tool not in catalog: {gold_unseen}/{N}  (should be 0)")

    np.save(CACHE_DIR / "query_embs.npy", query_embs)
    np.save(CACHE_DIR / "cand_idx.npy", cand_idx)
    np.save(CACHE_DIR / "cand_scores.npy", cand_scores)
    np.save(CACHE_DIR / "prev_idx.npy", prev_idx)
    np.save(CACHE_DIR / "gold_idx.npy", gold_idx)
    (CACHE_DIR / "meta.json").write_text(
        json.dumps(
            {
                "n": N,
                "top_n": TOP_N,
                "model": MODEL,
                "encoder_dim": int(query_embs.shape[1]),
                "vocab_size": len(router.vocab),
                "prev_unseen": prev_unseen,
                "gold_unseen": gold_unseen,
            },
            indent=2,
        )
    )
    print(f"\nCache written to {CACHE_DIR}")


if __name__ == "__main__":
    main()

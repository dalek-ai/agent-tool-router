"""Train a centroid-retrieval Router on data/traces.jsonl and save it.

Usage:
  python -m agent_tool_router.train --out models/baseline-v0
  python -m agent_tool_router.train --out models/baseline-v0 \\
      --exclude-source swe-bench-verified --min-tool-freq 3

The defaults reproduce the session-03 cross-corpus baseline:
  swe-bench excluded (degenerate vocab=1)
  min tool frequency in train = 3
  TF-IDF 1-2grams, max_features=50000, sublinear_tf=True
  test_size=0.2, random_state=42
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .router import Router

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "data" / "traces.jsonl"


def _load_traces(traces_path: Path, excluded: set[str]):
    rows = []
    with traces_path.open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            src = t.get("source_dataset", "")
            if src in excluded:
                continue
            task = (t.get("task_text") or "").strip()
            tools = [tc.get("name") for tc in (t.get("tools_called") or [])]
            tools = [n for n in tools if isinstance(n, str) and n]
            if not task or not tools:
                continue
            rows.append((task, tools, src))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output model directory")
    p.add_argument("--traces", default=str(TRACES))
    p.add_argument(
        "--exclude-source",
        action="append",
        default=["swe-bench-verified"],
        help="dataset name to exclude (repeatable)",
    )
    p.add_argument("--min-tool-freq", type=int, default=3)
    p.add_argument("--max-features", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    args = p.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import normalize

    excluded = set(args.exclude_source)
    rows = _load_traces(Path(args.traces), excluded)
    print(f"[load] {len(rows)} usable traces (excluded: {sorted(excluded)})",
          file=sys.stderr)

    # Dedupe (task, tool-tuple) — required because tau-bench ships repeated
    # trial copies that would leak between train and test.
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped = []
    for task, tools, src in rows:
        key = (task, tuple(tools))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((task, tools, src))
    print(f"[dedupe] {len(deduped)} unique (task, tools) pairs", file=sys.stderr)

    tasks = [r[0] for r in deduped]
    gold = [r[1] for r in deduped]
    sources = [r[2] for r in deduped]

    X_train, X_test, gold_train, gold_test, _, src_test = train_test_split(
        tasks, gold, sources,
        test_size=args.test_size, random_state=args.seed, stratify=sources
    )
    print(f"[split] train={len(X_train)} test={len(X_test)}", file=sys.stderr)

    train_freq: Counter[str] = Counter()
    for seq in gold_train:
        for n in set(seq):
            train_freq[n] += 1
    vocab = sorted(t for t, c in train_freq.items() if c >= args.min_tool_freq)
    name_to_idx = {n: i for i, n in enumerate(vocab)}
    V = len(vocab)
    print(f"[vocab] {V} tools (freq>={args.min_tool_freq}); "
          f"raw train vocab was {len(train_freq)}", file=sys.stderr)
    if V < 50:
        print("[abort] filtered vocab too small", file=sys.stderr)
        return 1

    vec = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
    )
    Xt_train = vec.fit_transform(X_train)
    n_feat = Xt_train.shape[1]

    centroids = np.zeros((V, n_feat), dtype=np.float64)
    counts = np.zeros(V, dtype=np.int64)
    for row_i, seq in enumerate(gold_train):
        for tool in set(seq):
            tidx = name_to_idx.get(tool)
            if tidx is None:
                continue
            centroids[tidx] += Xt_train[row_i].toarray().ravel()
            counts[tidx] += 1
    nonzero = counts > 0
    centroids[nonzero] /= counts[nonzero, None]
    centroids = normalize(centroids, axis=1)

    router = Router(vec=vec, centroids=centroids, vocab=vocab)
    out = router.save(args.out)
    print(f"[save] model -> {out}", file=sys.stderr)

    # Quick test-set top-3 sanity (so the saved model is verified on disk).
    Xt_test = vec.transform(X_test)
    Xt_test_n = normalize(Xt_test, axis=1)
    scores = np.asarray(Xt_test_n @ centroids.T)
    ranked = np.argsort(-scores, axis=1)
    hits = total = 0
    for i, golds in enumerate(gold_test):
        topk = set(ranked[i, :3].tolist())
        for tool in golds:
            if tool not in name_to_idx:
                continue
            total += 1
            if name_to_idx[tool] in topk:
                hits += 1
    if total:
        acc = hits / total
        rnd = 3 / V
        print(
            f"[eval] top-3 per-call acc = {acc:.4f}  random = {rnd:.6f}  "
            f"ratio = {acc/rnd:.1f}x  (V={V}, n_calls={total})",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

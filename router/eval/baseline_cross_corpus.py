"""Cross-corpus baseline router on the full 14K trace dataset.

Why centroid retrieval and not OneVsRest LogReg like baseline_tfidf.py:
  - The merged vocab has ~12K tool names with extreme long-tail
    (toolace alone contributes 10K names with very low overlap to other sources).
  - Training one binary classifier per tool would take forever and most heads
    would see <5 positive examples — useless.
  - Centroid retrieval scales linearly in vocab size, has no per-tool training,
    and gracefully handles rare tools (their centroid is just based on the
    handful of tasks where they appeared).

Algorithm:
  1. Build TF-IDF over task_text on the train split (no test leakage).
  2. For each tool t, compute mean TF-IDF vector across train tasks calling t.
     This is a "tool centroid" in task-space.
  3. At test time, score(task, t) = cosine(tfidf(task), centroid(t)). Rank
     tools by score; report per-call top-K hit rate.

We also filter the vocab to tools that appear >= MIN_TOOL_FREQ times in train
to avoid evaluating against singletons that no model could learn.

Run: python3 router/eval/baseline_cross_corpus.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "data" / "traces.jsonl"
SEED = 42
MIN_TOOL_FREQ = 3  # tool must appear in >= this many training tasks
EXCLUDED_SOURCES = {"swe-bench-verified"}  # vocab=1 (single "edit"), degenerate
KEEP_SOURCES_NOTE = "(toolace, hermes, tau-bench; swe-bench dropped: vocab=1)"


def load_traces() -> list[tuple[str, list[str], str]]:
    out: list[tuple[str, list[str], str]] = []
    with TRACES.open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            src = t.get("source_dataset", "")
            if src in EXCLUDED_SOURCES:
                continue
            task = (t.get("task_text") or "").strip()
            tools = [tc.get("name") for tc in (t.get("tools_called") or [])]
            tools = [n for n in tools if isinstance(n, str) and n]
            if not task or not tools:
                continue
            out.append((task, tools, src))
    return out


def main() -> int:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import normalize

    rows = load_traces()
    print(f"[load] {len(rows)} usable traces {KEEP_SOURCES_NOTE}", file=sys.stderr)

    # Dedupe by (task, tool-tuple). Tau-bench in particular ships 12 trial
    # copies per task — without dedup the test set leaks.
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped: list[tuple[str, list[str], str]] = []
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

    by_src = Counter(sources)
    print(f"[per-source] {dict(by_src)}", file=sys.stderr)

    X_train, X_test, gold_train, gold_test, src_train, src_test = train_test_split(
        tasks, gold, sources, test_size=0.2, random_state=SEED, stratify=sources
    )
    print(f"[split] train={len(X_train)} test={len(X_test)}", file=sys.stderr)

    # Vocab filter from training only.
    train_tool_freq: Counter[str] = Counter()
    for seq in gold_train:
        for n in set(seq):
            train_tool_freq[n] += 1
    vocab = sorted(t for t, c in train_tool_freq.items() if c >= MIN_TOOL_FREQ)
    name_to_idx = {n: i for i, n in enumerate(vocab)}
    V = len(vocab)
    print(
        f"[vocab] {V} tools with freq>={MIN_TOOL_FREQ} (train); "
        f"raw train vocab was {len(train_tool_freq)}",
        file=sys.stderr,
    )
    if V < 50:
        print("[abort] filtered vocab too small", file=sys.stderr)
        return 1

    # TF-IDF on train task text only.
    vec = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
    )
    Xt_train = vec.fit_transform(X_train)
    Xt_test = vec.transform(X_test)

    # Build [V, n_features] tool-centroid matrix from train.
    n_feat = Xt_train.shape[1]
    centroids = np.zeros((V, n_feat), dtype=np.float64)
    counts = np.zeros(V, dtype=np.int64)
    for row_i, seq in enumerate(gold_train):
        # Use a set so a tool repeated within one task doesn't double-count.
        for tool in set(seq):
            tidx = name_to_idx.get(tool)
            if tidx is None:
                continue
            centroids[tidx] += Xt_train[row_i].toarray().ravel()
            counts[tidx] += 1
    nonzero = counts > 0
    centroids[nonzero] /= counts[nonzero, None]
    centroids = normalize(centroids, axis=1)

    # Test scoring.
    Xt_test_n = normalize(Xt_test, axis=1)
    # scores[i, t] = cosine(test_i, centroid_t)
    scores = Xt_test_n @ centroids.T   # sparse × dense
    scores = np.asarray(scores)
    ranked = np.argsort(-scores, axis=1)

    # Per-call top-K accuracy.
    print()
    print("=== overall (cross-corpus) ===")
    for K in (1, 3, 5, 10):
        hits = 0
        total = 0
        for i, golds in enumerate(gold_test):
            topk = set(ranked[i, :K].tolist())
            for tool in golds:
                if tool not in name_to_idx:
                    continue
                total += 1
                if name_to_idx[tool] in topk:
                    hits += 1
        acc = hits / total if total else 0.0
        rnd = min(1.0, K / V)
        ratio = acc / rnd if rnd else float("inf")
        print(
            f"top-{K:>2} per-call acc = {acc:.4f}  random = {rnd:.6f}  "
            f"ratio = {ratio:.1f}x  (V={V}, n_calls={total})"
        )

    # Per-source breakdown — does the model generalize, or just memorize a source?
    print()
    print("=== per-source top-3 ===")
    by_src_test = Counter(src_test)
    for src in sorted(by_src_test):
        hits = 0
        total = 0
        for i, golds in enumerate(gold_test):
            if src_test[i] != src:
                continue
            topk = set(ranked[i, :3].tolist())
            for tool in golds:
                if tool not in name_to_idx:
                    continue
                total += 1
                if name_to_idx[tool] in topk:
                    hits += 1
        acc = hits / total if total else 0.0
        rnd = 3 / V
        ratio = acc / rnd if rnd else float("inf")
        print(
            f"  {src:30s} n_test={by_src_test[src]:>4} "
            f"calls={total:>5} top-3={acc:.4f}  ratio={ratio:.1f}x"
        )

    # Sanity check: does the model do worse if we *exclude* the source from
    # train? That would tell us whether cross-source transfer is real or zero.
    # We don't run that here (would take a while), but we log the question for N11.
    print()
    print("[note] cross-source transfer (leave-one-source-out) deferred to N11.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

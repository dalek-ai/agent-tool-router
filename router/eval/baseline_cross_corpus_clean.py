"""Cross-corpus baseline router with leak filter applied.

Same algorithm as baseline_cross_corpus.py, but excludes rows where the gold
tool name leaks into the task text (verbatim or as in-order subtokens within
a 4-token window). The point is to publish a number that nobody can wave
away on HN with "but Hermes is leaky".

After filtering, dataset shrinks from 14000 to about 11400 rows. Hermes goes
from 2180 to ~670 rows (the rest are essentially string-matching exercises).
ToolACE drops ~12%, tau-bench drops ~0.6%, swe-bench/osworld unchanged.

Run: python3 router/eval/baseline_cross_corpus_clean.py
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
MIN_TOOL_FREQ = 3
EXCLUDED_SOURCES = {"swe-bench-verified"}  # vocab=1, degenerate
LEAK_MODE = "loose"

sys.path.insert(0, str(ROOT / "router" / "eval"))
from leak_filter import is_leaky  # noqa: E402


def load_traces() -> tuple[list[tuple[str, list[str], str]], Counter[str]]:
    out: list[tuple[str, list[str], str]] = []
    dropped: Counter[str] = Counter()
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
            if is_leaky(task, tools, mode=LEAK_MODE):
                dropped[src] += 1
                continue
            out.append((task, tools, src))
    return out, dropped


def main() -> int:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import normalize

    rows, dropped = load_traces()
    print(f"[load] {len(rows)} non-leaky traces", file=sys.stderr)
    print(f"[dropped] {dict(dropped)} (mode={LEAK_MODE})", file=sys.stderr)

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

    vec = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
    )
    Xt_train = vec.fit_transform(X_train)
    Xt_test = vec.transform(X_test)

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

    Xt_test_n = normalize(Xt_test, axis=1)
    scores = Xt_test_n @ centroids.T
    scores = np.asarray(scores)
    ranked = np.argsort(-scores, axis=1)

    print()
    print("=== overall (cross-corpus, leak-filtered) ===")
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

    print()
    print("=== per-source top-3 (leak-filtered) ===")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Baseline TF-IDF tool router on tau-bench gold trajectories.

Question we want to answer: given a task description in natural language,
can a dumb bag-of-words classifier rank the *gold* tools higher than chance?

Phase-0 success criterion: top-3 per-call accuracy >= 2x random baseline.

Per-call top-K accuracy:
  Flatten (task_text, gold_tool) pairs from the test split. For each pair,
  ask the model to rank all tools by predicted probability and check whether
  the gold tool falls in the top K.
  Random baseline = K / |vocab| (uniform sampling of K tools).

Run: python3 router/eval/baseline_tfidf.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "data" / "traces.jsonl"
SOURCE = "tau-bench"
SEED = 42


def load_taubench_traces() -> list[tuple[str, list[str], dict]]:
    rows: list[tuple[str, list[str], dict]] = []
    with TRACES.open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t.get("source_dataset") != SOURCE:
                continue
            task = (t.get("task_text") or "").strip()
            tools = [tc.get("name") for tc in (t.get("tools_called") or [])]
            tools = [n for n in tools if n]
            if not task or not tools:
                continue
            rows.append((task, tools, t.get("metadata") or {}))
    return rows


def main() -> int:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    rows = load_taubench_traces()
    print(f"[load] {len(rows)} tau-bench traces with non-empty gold sequences",
          file=sys.stderr)
    if len(rows) < 100:
        print("[abort] too few traces to train", file=sys.stderr)
        return 1

    # Dedupe identical (task, tool-set) pairs to avoid leakage of trial-i copies.
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped: list[tuple[str, list[str]]] = []
    for task, tools, _ in rows:
        key = (task, tuple(tools))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((task, tools))
    print(f"[dedupe] {len(deduped)} unique (task, tools) pairs", file=sys.stderr)

    tasks = [r[0] for r in deduped]
    gold = [r[1] for r in deduped]

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(gold)
    vocab = list(mlb.classes_)
    V = len(vocab)
    print(f"[vocab] {V} unique tool names", file=sys.stderr)

    counter = Counter(t for seq in gold for t in seq)
    avg_seq = np.mean([len(s) for s in gold])
    print(f"[stats] mean gold seq len={avg_seq:.2f}, top tools:", file=sys.stderr)
    for name, n in counter.most_common(10):
        print(f"        {n:>4}  {name}", file=sys.stderr)

    X_train, X_test, Y_train, Y_test, gold_train, gold_test = train_test_split(
        tasks, Y, gold, test_size=0.2, random_state=SEED
    )
    print(f"[split] train={len(X_train)} test={len(X_test)}", file=sys.stderr)

    vec = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
    )
    Xt_train = vec.fit_transform(X_train)
    Xt_test = vec.transform(X_test)

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, C=4.0, solver="liblinear"),
        n_jobs=-1,
    )
    clf.fit(Xt_train, Y_train)

    # Build a probability matrix [n_test, V] preserving the mlb.classes_ order.
    proba = np.zeros((Xt_test.shape[0], V), dtype=np.float64)
    for i, est in enumerate(clf.estimators_):
        try:
            p = est.predict_proba(Xt_test)[:, 1]
        except (AttributeError, IndexError):
            # All-zero label in train: degenerate column, leave as zeros.
            continue
        proba[:, i] = p

    # Rank per row: argsort descending.
    ranked = np.argsort(-proba, axis=1)

    # Per-call top-K accuracy.
    name_to_idx = {n: i for i, n in enumerate(vocab)}
    for K in (1, 3, 5):
        hits = 0
        total = 0
        for row_i, golds in enumerate(gold_test):
            topk = set(ranked[row_i, :K].tolist())
            for tool in golds:
                idx = name_to_idx.get(tool)
                if idx is None:
                    continue
                total += 1
                if idx in topk:
                    hits += 1
        acc = hits / total if total else 0.0
        random_baseline = min(1.0, K / V)
        ratio = acc / random_baseline if random_baseline else float("inf")
        print(
            f"top-{K} per-call accuracy = {acc:.4f}  "
            f"random = {random_baseline:.4f}  ratio = {ratio:.2f}x"
        )

    # Macro-F1 on a binary-decision view: predict positive iff proba > 0.5.
    Y_pred = (proba > 0.5).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score
    macro_f1 = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
    macro_p = precision_score(Y_test, Y_pred, average="macro", zero_division=0)
    macro_r = recall_score(Y_test, Y_pred, average="macro", zero_division=0)
    print(
        f"macro precision={macro_p:.4f}  recall={macro_r:.4f}  F1={macro_f1:.4f}"
    )

    # Sequence-level: how often is the *exact* gold set inside the model's
    # top-N predictions where N = len(gold)?
    seq_hits = 0
    for row_i, golds in enumerate(gold_test):
        n = len(golds)
        topn = set(ranked[row_i, :n].tolist())
        gold_idx = {name_to_idx[t] for t in golds if t in name_to_idx}
        if gold_idx and gold_idx.issubset(topn):
            seq_hits += 1
    print(
        f"exact-set@len(gold) accuracy = {seq_hits / len(gold_test):.4f}  "
        f"({seq_hits}/{len(gold_test)})"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

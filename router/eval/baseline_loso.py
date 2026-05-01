"""Leave-one-source-out evaluation.

For each source S in {toolace, hermes-function-calling-v1, tau-bench}, we:
  1. Train the centroid retriever on traces from all OTHER sources.
  2. Evaluate on traces from S.

Two scores are reported per held-out source:
  - strict top-K   : count every gold call, even ones whose tool name was
    never in the train vocab (counts as a miss). This is the floor: it
    answers "if I deploy on a new ecosystem of tools, what do I get?".
  - shared top-K   : count only gold calls whose tool name appears in the
    train vocab. This answers "for the tools that *are* shared between the
    held-out domain and the train domains, can the router pick them?".

Both numbers matter. The strict number is the honest headline (a centroid
retriever cannot predict tools it has never seen). The shared number tells
you whether the textual signal in the task transfers across ecosystems.

Run: python3 router/eval/baseline_loso.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "data" / "traces.jsonl"
MIN_TOOL_FREQ = 3
EXCLUDED_SOURCES = {"swe-bench-verified", "osworld"}


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


def _dedupe(rows):
    seen = set()
    deduped = []
    for task, tools, src in rows:
        key = (task, tuple(tools))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((task, tools, src))
    return deduped


def evaluate_held_out(rows, held_out: str) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    train_rows = [r for r in rows if r[2] != held_out]
    test_rows = [r for r in rows if r[2] == held_out]
    if not train_rows or not test_rows:
        return {"held_out": held_out, "skipped": True}

    X_train = [r[0] for r in train_rows]
    gold_train = [r[1] for r in train_rows]
    X_test = [r[0] for r in test_rows]
    gold_test = [r[1] for r in test_rows]

    train_tool_freq: Counter[str] = Counter()
    for seq in gold_train:
        for n in set(seq):
            train_tool_freq[n] += 1
    vocab = sorted(t for t, c in train_tool_freq.items() if c >= MIN_TOOL_FREQ)
    name_to_idx = {n: i for i, n in enumerate(vocab)}
    V = len(vocab)
    if V < 10:
        return {"held_out": held_out, "skipped": True, "reason": f"vocab={V}"}

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
    scores = np.asarray(Xt_test_n @ centroids.T)
    ranked = np.argsort(-scores, axis=1)

    out = {
        "held_out": held_out,
        "vocab_size": V,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "test_calls_total": 0,
        "test_calls_in_vocab": 0,
    }

    for K in (1, 3, 5, 10):
        strict_hits = 0
        strict_total = 0
        shared_hits = 0
        shared_total = 0
        for i, golds in enumerate(gold_test):
            topk = set(ranked[i, :K].tolist())
            for tool in golds:
                strict_total += 1
                if tool in name_to_idx:
                    shared_total += 1
                    if name_to_idx[tool] in topk:
                        shared_hits += 1
                        strict_hits += 1
        rnd = min(1.0, K / V)
        out[f"strict_top{K}"] = {
            "acc": strict_hits / strict_total if strict_total else 0.0,
            "n": strict_total,
            "ratio_vs_random": (strict_hits / strict_total) / rnd if rnd and strict_total else 0.0,
        }
        out[f"shared_top{K}"] = {
            "acc": shared_hits / shared_total if shared_total else 0.0,
            "n": shared_total,
            "ratio_vs_random": (shared_hits / shared_total) / rnd if rnd and shared_total else 0.0,
        }
        if K == 1:
            out["test_calls_total"] = strict_total
            out["test_calls_in_vocab"] = shared_total
    return out


def main() -> int:
    rows = _dedupe(load_traces())
    print(f"[load+dedupe] {len(rows)} unique (task, tools) pairs", file=sys.stderr)
    by_src = Counter(r[2] for r in rows)
    print(f"[per-source] {dict(by_src)}", file=sys.stderr)
    print()

    sources = sorted(by_src)
    results = []
    for src in sources:
        print(f"=== held out: {src} ===")
        res = evaluate_held_out(rows, src)
        if res.get("skipped"):
            print(f"  skipped: {res.get('reason', 'too small')}")
            print()
            continue
        results.append(res)
        in_vocab_pct = 100 * res["test_calls_in_vocab"] / max(1, res["test_calls_total"])
        print(
            f"  V (train) = {res['vocab_size']}, "
            f"n_train = {res['n_train']}, n_test = {res['n_test']}"
        )
        print(
            f"  test calls: {res['test_calls_total']} total, "
            f"{res['test_calls_in_vocab']} in train-vocab ({in_vocab_pct:.1f}%)"
        )
        for K in (1, 3, 5):
            s = res[f"strict_top{K}"]
            sh = res[f"shared_top{K}"]
            print(
                f"  top-{K:>2}  strict acc = {s['acc']:.4f}  "
                f"({s['ratio_vs_random']:.1f}x random, n={s['n']})  "
                f"|  shared-vocab acc = {sh['acc']:.4f}  "
                f"({sh['ratio_vs_random']:.1f}x random, n={sh['n']})"
            )
        print()

    # Summary line for README quoting.
    print("=== summary (top-3) ===")
    for res in results:
        s3 = res["strict_top3"]
        sh3 = res["shared_top3"]
        print(
            f"  {res['held_out']:30s}  strict={s3['acc']:.3f} ({s3['ratio_vs_random']:.1f}x)  "
            f"shared={sh3['acc']:.3f} ({sh3['ratio_vs_random']:.1f}x)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Leave-one-source-out evaluation, but routing on tool *descriptions*
instead of tool *names*.

Motivation: baseline_loso.py reports ~0% strict top-3 across all three
held-out sources, because the train and test corpora share essentially no
tool names (vocab overlap 0.0–0.1%). That tells us names don't transfer.
But descriptions might. "Search for flights between two airports" is a
sentence that overlaps lexically across ecosystems, even if the tool is
called search_direct_flight in one and FlightSearchAPI in another.

Algorithm:
  For held-out source S:
    1. Train task/desc pool = (task texts) + (descriptions of tools used in
       training-source rows), drawn from sources != S.
    2. Fit a single TfidfVectorizer on that pool. This anchors the embedding
       space in natural language vocabulary, not synthetic tool names.
    3. Test pool = S's tool catalog. For each test row (task, gold_tools),
       embed the task with the trained vectorizer, embed every candidate
       tool's description, score by cosine.
    4. Top-K = how often the gold tool's description ranks in the top K
       among S's catalog.

Random baseline: K / |S's catalog|. This is a different denominator than
baseline_loso.py — there, V was the *training* vocab; here, |S's catalog|
is the *test* vocab. So ratios across the two scripts are not directly
comparable, but ratios within this script are.

Run: python3 router/eval/baseline_loso_descriptions.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "data" / "traces.jsonl"
DESCS = ROOT / "data" / "tool_descriptions.jsonl"
EXCLUDED_SOURCES = {"swe-bench-verified", "osworld"}


def load_traces():
    out = []
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


def load_descs() -> dict[str, dict[str, str]]:
    """Returns {source: {tool_name: description}}."""
    out: dict[str, dict[str, str]] = {}
    with DESCS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            src = r["source"]
            name = r["name"]
            desc = r.get("description") or ""
            out.setdefault(src, {})[name] = desc
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


def _tool_text(name: str, desc: str, *, include_name: bool = True) -> str:
    """Tool 'document' for retrieval.

    By default = description + name spelled out (camelCase + _ split).
    The name carries useful natural-language signal ('cancel_pending_order'
    encodes {cancel, pending, order}). We strip it to subtokens so we don't
    reintroduce string-match leakage.

    With include_name=False, we score on description text only — useful as
    a sanity check that the cross-source signal isn't all coming from name
    subtoken overlap.
    """
    if not include_name:
        return desc.strip()
    import re
    parts = re.split(r"[_\.\s\-]+", name)
    subs = []
    for p in parts:
        subs.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", p))
    name_text = " ".join(t.lower() for t in subs if len(t) >= 2)
    return f"{desc} {name_text}".strip()


def evaluate_held_out(rows, descs_by_src, held_out: str,
                      *, include_name: bool = True) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    train_rows = [r for r in rows if r[2] != held_out]
    test_rows = [r for r in rows if r[2] == held_out]
    if not train_rows or not test_rows:
        return {"held_out": held_out, "skipped": True}

    held_out_catalog = descs_by_src.get(held_out, {})
    if not held_out_catalog:
        return {"held_out": held_out, "skipped": True,
                "reason": f"no descriptions for {held_out}"}

    # Restrict held-out catalog to tools that actually appear as gold in
    # the test set. The full ToolACE catalog has 16K entries — if 90% of
    # them never appear in test, padding the candidate pool with them just
    # makes the random baseline harder without measuring anything new.
    test_gold_tools = set()
    for _, gold, _ in test_rows:
        test_gold_tools.update(gold)
    catalog = {n: d for n, d in held_out_catalog.items() if n in test_gold_tools}
    # Also drop tools whose description is empty (we'd be scoring noise).
    catalog = {n: d for n, d in catalog.items() if d}
    if len(catalog) < 5:
        return {"held_out": held_out, "skipped": True,
                "reason": f"catalog too small ({len(catalog)})"}

    catalog_names = sorted(catalog)
    name_to_idx = {n: i for i, n in enumerate(catalog_names)}
    V = len(catalog_names)

    # Training corpus: train task texts + descriptions of training-source
    # tools that exist (we use the descs we extracted, not all tools).
    train_corpus = [r[0] for r in train_rows]
    for src, dmap in descs_by_src.items():
        if src == held_out:
            continue
        for n, d in dmap.items():
            if d:
                train_corpus.append(_tool_text(n, d, include_name=include_name))

    vec = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
    )
    vec.fit(train_corpus)

    tool_docs = [_tool_text(n, catalog[n], include_name=include_name)
                 for n in catalog_names]
    tool_vecs = normalize(vec.transform(tool_docs), axis=1)

    test_tasks = [r[0] for r in test_rows]
    Xt_test = normalize(vec.transform(test_tasks), axis=1)
    scores_sparse = Xt_test @ tool_vecs.T
    # sparse @ sparse returns sparse; densify for argsort
    if hasattr(scores_sparse, "toarray"):
        scores = scores_sparse.toarray()
    else:
        scores = np.asarray(scores_sparse)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    ranked = np.argsort(-scores, axis=1)

    out = {
        "held_out": held_out,
        "catalog_size": V,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "test_calls_total": 0,
        "test_calls_in_catalog": 0,
    }
    for K in (1, 3, 5, 10):
        hits = 0
        total = 0
        in_cat = 0
        for i, (_, golds, _) in enumerate(test_rows):
            topk = set(ranked[i, :K].tolist())
            for tool in golds:
                total += 1
                if tool in name_to_idx:
                    in_cat += 1
                    if name_to_idx[tool] in topk:
                        hits += 1
        rnd = min(1.0, K / V)
        out[f"top{K}"] = {
            "acc": hits / in_cat if in_cat else 0.0,
            "n": in_cat,
            "ratio_vs_random": (hits / in_cat) / rnd if rnd and in_cat else 0.0,
        }
        if K == 1:
            out["test_calls_total"] = total
            out["test_calls_in_catalog"] = in_cat
    return out


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-name", action="store_true",
                   help="Use tool descriptions only, drop name subtokens.")
    args = p.parse_args()
    include_name = not args.no_name

    rows = _dedupe(load_traces())
    print(f"[load+dedupe] {len(rows)} unique (task, tools) pairs", file=sys.stderr)
    by_src = Counter(r[2] for r in rows)
    print(f"[per-source] {dict(by_src)}", file=sys.stderr)

    descs_by_src = load_descs()
    print(
        f"[descriptions] "
        f"{ {s: len(d) for s, d in descs_by_src.items()} }",
        file=sys.stderr,
    )
    print(f"[mode] include_name={include_name}", file=sys.stderr)
    print()

    results = []
    for src in sorted(by_src):
        print(f"=== held out: {src} ===")
        res = evaluate_held_out(rows, descs_by_src, src, include_name=include_name)
        if res.get("skipped"):
            print(f"  skipped: {res.get('reason', 'too small')}")
            print()
            continue
        results.append(res)
        print(
            f"  catalog (held-out source) = {res['catalog_size']} tools  "
            f"n_train = {res['n_train']}, n_test = {res['n_test']}"
        )
        in_cat_pct = (
            100 * res["test_calls_in_catalog"] / max(1, res["test_calls_total"])
        )
        print(
            f"  test calls: {res['test_calls_total']} total, "
            f"{res['test_calls_in_catalog']} in held-out catalog ({in_cat_pct:.1f}%)"
        )
        for K in (1, 3, 5, 10):
            r = res[f"top{K}"]
            print(
                f"  top-{K:>2} acc = {r['acc']:.4f}  "
                f"({r['ratio_vs_random']:.1f}x random, n={r['n']})"
            )
        print()

    print("=== summary (top-3, description retrieval) ===")
    for res in results:
        r3 = res["top3"]
        print(
            f"  {res['held_out']:30s}  "
            f"top-3={r3['acc']:.3f} ({r3['ratio_vs_random']:.1f}x random, "
            f"V={res['catalog_size']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

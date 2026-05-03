"""LOSO eval, descriptions retrieval, with a hybrid TF-IDF + bi-encoder
score: ``score = alpha * cos_tfidf(task, desc) + (1 - alpha) * cos_enc(task, desc)``.

Why:
  baseline_loso_descriptions.py (TF-IDF) and baseline_loso_descriptions_biencoder.py
  (sentence-transformers MiniLM) report opposite strengths.
    Hermes  held-out: TF-IDF 73.5%  /  bi-enc 67.9%  -> TF-IDF wins
    tau-bench       : TF-IDF 19.8%  /  bi-enc 31.1%  -> bi-enc wins
    ToolACE         : TF-IDF 34.6%  /  bi-enc 58.5%  -> bi-enc wins
  TF-IDF dominates when task and description share lexical surface; the bi-encoder
  dominates when descriptions paraphrase the task semantically without sharing words.
  Question for v0.4: does a linear combination Pareto-dominate both?

Protocol:
  For each held-out source S, compute the TF-IDF-and-bi-encoder cosine matrices
  exactly as the two existing scripts do, then sweep alpha in {0.0, 0.1, ..., 1.0}
  and report top-K at each alpha. We DO NOT learn alpha on the held-out test set
  (that would be cheating); we report all alphas so the per-source best alpha is an
  oracle upper bound, and a single global alpha (alpha=0.5 by default) shows what a
  no-tuning hybrid gives.

  Both score matrices are cosine similarities on L2-normalized vectors, so they
  share the same [-1, 1] range and a flat linear combo is sensible without
  per-source calibration. (RRF would be the next-cleanest alternative.)

Run: python3 router/eval/baseline_loso_descriptions_hybrid.py
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
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALPHAS = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.1)]


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
    if not include_name:
        return desc.strip()
    import re
    parts = re.split(r"[_\.\s\-]+", name)
    subs = []
    for p in parts:
        subs.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", p))
    name_text = " ".join(t.lower() for t in subs if len(t) >= 2)
    return f"{desc} {name_text}".strip()


def _topk_recall(ranked, test_rows, name_to_idx, K):
    hits = 0
    in_cat = 0
    for i, (_, golds, _) in enumerate(test_rows):
        topk = set(ranked[i, :K].tolist())
        for tool in golds:
            if tool in name_to_idx:
                in_cat += 1
                if name_to_idx[tool] in topk:
                    hits += 1
    return hits, in_cat


def evaluate_held_out(model, rows, descs_by_src, held_out: str,
                      *, include_name: bool = True) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    test_rows = [r for r in rows if r[2] == held_out]
    train_rows = [r for r in rows if r[2] != held_out]
    if not test_rows or not train_rows:
        return {"held_out": held_out, "skipped": True}

    held_out_catalog = descs_by_src.get(held_out, {})
    if not held_out_catalog:
        return {"held_out": held_out, "skipped": True,
                "reason": f"no descriptions for {held_out}"}

    test_gold_tools = set()
    for _, gold, _ in test_rows:
        test_gold_tools.update(gold)
    catalog = {n: d for n, d in held_out_catalog.items()
               if n in test_gold_tools and d}
    if len(catalog) < 5:
        return {"held_out": held_out, "skipped": True,
                "reason": f"catalog too small ({len(catalog)})"}

    catalog_names = sorted(catalog)
    name_to_idx = {n: i for i, n in enumerate(catalog_names)}
    V = len(catalog_names)
    tool_docs = [_tool_text(n, catalog[n], include_name=include_name)
                 for n in catalog_names]
    test_tasks = [r[0] for r in test_rows]

    train_corpus = [r[0] for r in train_rows]
    for src, dmap in descs_by_src.items():
        if src == held_out:
            continue
        for n, d in dmap.items():
            if d:
                train_corpus.append(_tool_text(n, d, include_name=include_name))
    vec = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2), min_df=2,
        sublinear_tf=True, lowercase=True,
    )
    vec.fit(train_corpus)
    tool_tfidf = normalize(vec.transform(tool_docs), axis=1)
    task_tfidf = normalize(vec.transform(test_tasks), axis=1)
    s_tfidf = (task_tfidf @ tool_tfidf.T)
    if hasattr(s_tfidf, "toarray"):
        s_tfidf = s_tfidf.toarray()
    s_tfidf = np.asarray(s_tfidf, dtype=np.float32)

    tool_enc = model.encode(
        tool_docs, batch_size=64, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    task_enc = model.encode(
        test_tasks, batch_size=64, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    s_enc = task_enc @ tool_enc.T

    out = {
        "held_out": held_out,
        "catalog_size": V,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "alphas": {},
    }
    total_test_calls = sum(len(g) for _, g, _ in test_rows)
    in_cat_calls = sum(1 for _, g, _ in test_rows for t in g if t in name_to_idx)
    out["test_calls_total"] = total_test_calls
    out["test_calls_in_catalog"] = in_cat_calls

    for alpha in ALPHAS:
        score = alpha * s_tfidf + (1.0 - alpha) * s_enc
        ranked = np.argsort(-score, axis=1)
        per_alpha = {}
        for K in (1, 3, 5, 10):
            hits, in_cat = _topk_recall(ranked, test_rows, name_to_idx, K)
            rnd = min(1.0, K / V)
            per_alpha[f"top{K}"] = {
                "acc": hits / in_cat if in_cat else 0.0,
                "n": in_cat,
                "ratio_vs_random": (hits / in_cat) / rnd if rnd and in_cat else 0.0,
            }
        out["alphas"][alpha] = per_alpha
    return out


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-name", action="store_true",
                   help="Use tool descriptions only, drop name subtokens.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()
    include_name = not args.no_name

    print(f"[model] loading {args.model}", file=sys.stderr)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    rows = _dedupe(load_traces())
    print(f"[load+dedupe] {len(rows)} unique (task, tools) pairs", file=sys.stderr)
    by_src = Counter(r[2] for r in rows)
    print(f"[per-source] {dict(by_src)}", file=sys.stderr)
    descs_by_src = load_descs()
    print(f"[mode] include_name={include_name}", file=sys.stderr)
    print()

    all_results = []
    for src in sorted(by_src):
        print(f"=== held out: {src} ===")
        res = evaluate_held_out(model, rows, descs_by_src, src,
                                include_name=include_name)
        if res.get("skipped"):
            print(f"  skipped: {res.get('reason', 'too small')}")
            print()
            continue
        all_results.append(res)
        print(f"  V={res['catalog_size']} n_train={res['n_train']} n_test={res['n_test']}")
        print(f"  alpha   top-1     top-3     top-5     top-10")
        for alpha in ALPHAS:
            r = res["alphas"][alpha]
            print(
                f"  {alpha:>4.1f}   "
                f"{r['top1']['acc']:.4f}    "
                f"{r['top3']['acc']:.4f}    "
                f"{r['top5']['acc']:.4f}    "
                f"{r['top10']['acc']:.4f}"
            )
        print()

    print("=== summary: top-3 acc per (held-out, alpha) ===")
    header = "  held_out".ljust(30) + "  ".join(f"a={a:.1f}" for a in ALPHAS)
    print(header)
    for res in all_results:
        cells = [f"{res['alphas'][a]['top3']['acc']:.3f}" for a in ALPHAS]
        print(f"  {res['held_out']:28s}  " + "  ".join(cells))

    print()
    print("=== summary: best alpha per held-out (top-3) ===")
    for res in all_results:
        best_a, best_acc = max(
            ((a, res['alphas'][a]['top3']['acc']) for a in ALPHAS),
            key=lambda x: x[1],
        )
        tfidf_acc = res['alphas'][1.0]['top3']['acc']
        enc_acc = res['alphas'][0.0]['top3']['acc']
        print(
            f"  {res['held_out']:28s}  "
            f"tfidf={tfidf_acc:.3f}  enc={enc_acc:.3f}  "
            f"best(a={best_a:.1f})={best_acc:.3f}  "
            f"delta_vs_better={(best_acc - max(tfidf_acc, enc_acc)):+.3f}"
        )

    print()
    print("=== summary: global alpha=0.5 (no tuning) vs each backend (top-3) ===")
    for res in all_results:
        a05 = res['alphas'][0.5]['top3']['acc']
        tfidf_acc = res['alphas'][1.0]['top3']['acc']
        enc_acc = res['alphas'][0.0]['top3']['acc']
        print(
            f"  {res['held_out']:28s}  "
            f"tfidf={tfidf_acc:.3f}  enc={enc_acc:.3f}  "
            f"hybrid05={a05:.3f}  "
            f"delta_vs_better={(a05 - max(tfidf_acc, enc_acc)):+.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

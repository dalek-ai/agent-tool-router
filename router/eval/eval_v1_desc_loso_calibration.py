"""LOSO refit eval with per-task score normalization before linear combo.

The shipped hybrid scores tools as alpha * cos_tfidf + (1-alpha) * cos_enc with
no per-task rescaling. The two cosines have different distributions (tfidf
sparse, encoder dense around 0.3-0.5), so a fixed alpha is unevenly weighted
in practice.

This script reuses the LOSO refit setup of eval_v1_desc_loso_hybrid but applies
a per-task normalization to s_tfidf and s_enc before combining. We test:

  none    : baseline, what the shipped hybrid does
  minmax  : (s - row_min) / (row_max - row_min)
  zscore  : (s - row_mean) / row_std
  rank    : 1 - rank/V (uniform [0, 1] regardless of source distribution)

Hypothesis: rank or zscore should specifically help tau-bench, where tfidf
cosines are very sparse on the 18K-tool catalog and the linear combo is
de facto encoder-dominated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

DESCS_PATH = ROOT / "data" / "tool_descriptions.jsonl"
TRACES_PATH = ROOT / "data" / "traces.jsonl"
EXCLUDED_SOURCES = {"swe-bench-verified", "osworld"}
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

NORM_MODES = ["none", "minmax", "zscore", "rank"]


def _name_subtokens(name: str) -> str:
    parts = re.split(r"[_\.\s\-]+", name)
    subs: list[str] = []
    for p in parts:
        subs.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", p))
    return " ".join(t.lower() for t in subs if len(t) >= 2)


def _tool_text(name: str, desc: str, *, include_name: bool = True) -> str:
    if not include_name:
        return desc.strip()
    return f"{desc} {_name_subtokens(name)}".strip()


def load_descriptions() -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    with DESCS_PATH.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            name = d.get("name")
            desc = (d.get("description") or "").strip()
            src = d.get("source", "")
            if not isinstance(name, str) or not name or name in seen or not desc:
                continue
            seen.add(name)
            rows.append({"name": name, "description": desc, "source": src})
    return rows


def load_traces() -> list[tuple[str, list[str], str]]:
    out: list[tuple[str, list[str], str]] = []
    with TRACES_PATH.open(encoding="utf-8") as f:
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


def normalize_scores(s: np.ndarray, mode: str) -> np.ndarray:
    """Per-row normalization. s shape (n_tasks, V)."""
    if mode == "none":
        return s
    if mode == "minmax":
        mins = s.min(axis=1, keepdims=True)
        maxs = s.max(axis=1, keepdims=True)
        denom = np.maximum(maxs - mins, 1e-9)
        return (s - mins) / denom
    if mode == "zscore":
        mu = s.mean(axis=1, keepdims=True)
        sigma = np.maximum(s.std(axis=1, keepdims=True), 1e-9)
        return (s - mu) / sigma
    if mode == "rank":
        ranks = np.argsort(np.argsort(-s, axis=1), axis=1).astype(np.float32)
        V = s.shape[1]
        return 1.0 - ranks / V
    raise ValueError(f"unknown norm mode: {mode}")


def topk_accuracy(
    scores: np.ndarray,
    test_rows: list[tuple[str, list[str], str]],
    catalog_names: list[str],
    name_to_idx: dict[str, int],
    ks: list[int],
) -> tuple[int, dict[int, int]]:
    k_max = max(ks)
    topk_idx = np.argpartition(-scores, k_max - 1, axis=1)[:, :k_max]
    row_idx = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row_idx, topk_idx], axis=1)
    ranked = topk_idx[row_idx, order]

    hits = {k: 0 for k in ks}
    n_calls = 0
    for i, (_, gold, _) in enumerate(test_rows):
        top_lists = {k: {catalog_names[j] for j in ranked[i, :k]} for k in ks}
        for tool in gold:
            if tool not in name_to_idx:
                continue
            n_calls += 1
            for k in ks:
                if tool in top_lists[k]:
                    hits[k] += 1
    return n_calls, hits


def evaluate_held_out(
    held_out: str,
    *,
    descs: list[dict],
    traces: list[tuple[str, list[str], str]],
    model,
    catalog_encodings: np.ndarray,
    catalog_names: list[str],
    name_to_idx: dict[str, int],
    alpha: float,
    norms: list[str],
    ks: list[int],
    include_name: bool,
) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    test_rows = [r for r in traces if r[2] == held_out]
    train_rows = [r for r in traces if r[2] != held_out]
    if not test_rows or not train_rows:
        return {"held_out": held_out, "skipped": True}

    print(f"[loso] held_out={held_out}  n_test={len(test_rows)}  "
          f"n_train={len(train_rows)}", file=sys.stderr)

    train_corpus: list[str] = [r[0] for r in train_rows]
    for d in descs:
        if d["source"] == held_out:
            continue
        train_corpus.append(_tool_text(d["name"], d["description"],
                                       include_name=include_name))

    t0 = time.time()
    vec = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2), min_df=2,
        sublinear_tf=True, lowercase=True,
    )
    vec.fit(train_corpus)
    print(f"[loso] tfidf fit |vocab|={len(vec.vocabulary_)} in "
          f"{time.time()-t0:.1f}s", file=sys.stderr)

    catalog_docs = [_tool_text(d["name"], d["description"],
                               include_name=include_name) for d in descs]
    tool_tfidf = normalize(vec.transform(catalog_docs), axis=1)

    test_tasks = [r[0] for r in test_rows]
    task_tfidf = normalize(vec.transform(test_tasks), axis=1)

    s_tfidf = task_tfidf @ tool_tfidf.T
    if hasattr(s_tfidf, "toarray"):
        s_tfidf = s_tfidf.toarray()
    s_tfidf = np.asarray(s_tfidf, dtype=np.float32)

    t0 = time.time()
    task_enc = model.encode(
        test_tasks, batch_size=128, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    print(f"[loso] task encode {len(test_tasks)} in {time.time()-t0:.1f}s",
          file=sys.stderr)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        s_enc = task_enc @ catalog_encodings.T

    V = len(catalog_names)
    out: dict = {
        "held_out": held_out,
        "catalog_size": V,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "norms": {},
    }

    for norm in norms:
        s_t = normalize_scores(s_tfidf, norm)
        s_e = normalize_scores(s_enc, norm)
        score = alpha * s_t + (1.0 - alpha) * s_e

        n_calls, hits = topk_accuracy(
            score, test_rows, catalog_names, name_to_idx, ks
        )
        per_norm: dict = {"n_calls_in_catalog": n_calls}
        for k in ks:
            acc = hits[k] / n_calls if n_calls else 0.0
            rnd = k / V
            per_norm[f"top{k}"] = {
                "acc": acc,
                "ratio_vs_random": acc / rnd if rnd else 0.0,
            }
        out["norms"][norm] = per_norm

    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.5,
                   help="mixing weight on tfidf (default 0.5)")
    p.add_argument("--norms", default=",".join(NORM_MODES),
                   help=f"comma-separated norm modes from {NORM_MODES}")
    p.add_argument("--ks", default="1,3,5,10")
    p.add_argument("--encoder-model", default=DEFAULT_MODEL)
    p.add_argument("--no-name", action="store_true")
    args = p.parse_args()

    norms = [n.strip() for n in args.norms.split(",") if n.strip()]
    for n in norms:
        if n not in NORM_MODES:
            raise SystemExit(f"unknown norm: {n}; choose from {NORM_MODES}")
    ks = [int(k) for k in args.ks.split(",")]
    include_name = not args.no_name

    print(f"[load] descriptions from {DESCS_PATH}", file=sys.stderr)
    descs = load_descriptions()
    by_src = Counter(d["source"] for d in descs)
    print(f"[load] {len(descs)} unique tools, sources={dict(by_src)}",
          file=sys.stderr)

    catalog_names = [d["name"] for d in descs]
    name_to_idx = {n: i for i, n in enumerate(catalog_names)}

    print(f"[load] traces from {TRACES_PATH}", file=sys.stderr)
    traces = load_traces()
    by_trace_src = Counter(r[2] for r in traces)
    print(f"[load] {len(traces)} traces, sources={dict(by_trace_src)}",
          file=sys.stderr)

    print(f"[encoder] loading {args.encoder_model}", file=sys.stderr)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.encoder_model)

    catalog_docs = [_tool_text(d["name"], d["description"],
                               include_name=include_name) for d in descs]
    print(f"[encoder] encoding {len(catalog_docs)} catalog docs", file=sys.stderr)
    t0 = time.time()
    catalog_encodings = model.encode(
        catalog_docs, batch_size=256, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    print(f"[encoder] catalog encoded in {time.time()-t0:.1f}s",
          file=sys.stderr)

    sources = sorted(s for s in by_trace_src if s in by_src)
    print(f"[loso] sources to hold out: {sources}  alpha={args.alpha}  "
          f"norms={norms}", file=sys.stderr)

    all_results: list[dict] = []
    for src in sources:
        print(f"\n=== held out: {src} ===", flush=True)
        res = evaluate_held_out(
            src, descs=descs, traces=traces, model=model,
            catalog_encodings=catalog_encodings,
            catalog_names=catalog_names,
            name_to_idx=name_to_idx,
            alpha=args.alpha, norms=norms, ks=ks,
            include_name=include_name,
        )
        if res.get("skipped"):
            print(f"  skipped"); continue
        all_results.append(res)
        V = res["catalog_size"]
        print(f"  V={V}  n_test={res['n_test']}")
        for norm in norms:
            r = res["norms"][norm]
            line = f"  norm={norm:7s}  n_calls={r['n_calls_in_catalog']}"
            for k in ks:
                acc = r[f"top{k}"]["acc"]
                line += f"   top{k}={acc:.4f}"
            print(line)

    print()
    print(f"=== summary: top-3 acc per (held_out, norm) at alpha={args.alpha} ===")
    header = "  held_out".ljust(28) + "  ".join(f"{n:>7s}" for n in norms)
    print(header)
    weighted = {n: 0.0 for n in norms}
    weighted_n = {n: 0 for n in norms}
    for res in all_results:
        cells = []
        for n in norms:
            acc = res["norms"][n]["top3"]["acc"]
            n_calls = res["norms"][n]["n_calls_in_catalog"]
            cells.append(f"{acc:.3f}")
            weighted[n] += acc * n_calls
            weighted_n[n] += n_calls
        print(f"  {res['held_out']:26s}  " + "  ".join(f"{c:>7s}" for c in cells))
    print()
    print("  weighted overall (by n_calls):")
    for n in norms:
        wm = weighted[n] / weighted_n[n] if weighted_n[n] else 0.0
        print(f"    {n:7s}  {wm:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

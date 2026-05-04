"""LOSO refit eval of the hybrid v1-desc pipeline on the full 18K-tool catalog.

This is the apples-to-apples cross-source number for the shipped
baseline-v1-desc-hybrid model. eval_v1_desc_encoder.py reports the full-catalog
numbers when the model has trained on descriptions from all three sources;
this script answers: what would those numbers be if the TF-IDF vocabulary
had never seen the held-out source?

The bi-encoder is pretrained and source-agnostic, so only the TF-IDF half
gets the LOSO treatment here. The catalog at eval time is still the full
18K tools (so that we measure top-k accuracy at the same scale as the shipped
model), but the TF-IDF vocabulary is fit only on (train task texts + train
tool descriptions from the two non-held-out sources). The S tools are
vectorized using that S-leaked-out vocab, which is the realistic test of
how the shipped pipeline would behave on a brand-new source.

Usage:
  python -m router.eval.eval_v1_desc_loso_hybrid
  python -m router.eval.eval_v1_desc_loso_hybrid --alpha 0.5

Reports per-source top-k accuracy at alpha in {0.0, 0.5, 1.0} so the read
covers encoder-only, hybrid, and tfidf-only with the same LOSO split.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

DESCS_PATH = ROOT / "data" / "tool_descriptions.jsonl"
TRACES_PATH = ROOT / "data" / "traces.jsonl"
EXCLUDED_SOURCES = {"swe-bench-verified", "osworld"}
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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


def evaluate_held_out(
    held_out: str,
    *,
    descs: list[dict],
    traces: list[tuple[str, list[str], str]],
    model,
    catalog_encodings: np.ndarray,
    catalog_names: list[str],
    name_to_idx: dict[str, int],
    alphas: list[float],
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
        "alphas": {},
    }

    for alpha in alphas:
        if alpha == 1.0:
            score = s_tfidf
        elif alpha == 0.0:
            score = s_enc
        else:
            score = alpha * s_tfidf + (1.0 - alpha) * s_enc

        k_max = max(ks)
        topk_idx = np.argpartition(-score, k_max - 1, axis=1)[:, :k_max]
        row_idx = np.arange(score.shape[0])[:, None]
        order = np.argsort(-score[row_idx, topk_idx], axis=1)
        ranked = topk_idx[row_idx, order]

        per_alpha: dict = {}
        per_source_calls = 0
        per_source_hits = {k: 0 for k in ks}
        for i, (_, gold, _) in enumerate(test_rows):
            topk_names = {catalog_names[j] for j in ranked[i, :k_max]}
            top_lists = {k: {catalog_names[j] for j in ranked[i, :k]} for k in ks}
            for tool in gold:
                if tool not in name_to_idx:
                    continue
                per_source_calls += 1
                for k in ks:
                    if tool in top_lists[k]:
                        per_source_hits[k] += 1
        per_alpha["n_calls_in_catalog"] = per_source_calls
        for k in ks:
            acc = per_source_hits[k] / per_source_calls if per_source_calls else 0.0
            rnd = k / V
            per_alpha[f"top{k}"] = {
                "acc": acc,
                "ratio_vs_random": acc / rnd if rnd else 0.0,
            }
        out["alphas"][alpha] = per_alpha

    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", default="0.0,0.5,1.0",
                   help="comma-separated mixing weights on tfidf to evaluate")
    p.add_argument("--ks", default="1,3,5,10")
    p.add_argument("--encoder-model", default=DEFAULT_MODEL)
    p.add_argument("--no-name", action="store_true",
                   help="Use tool descriptions only, drop name subtokens.")
    args = p.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
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
    print(f"[loso] sources to hold out: {sources}", file=sys.stderr)

    all_results: list[dict] = []
    for src in sources:
        print(f"\n=== held out: {src} ===", flush=True)
        res = evaluate_held_out(
            src, descs=descs, traces=traces, model=model,
            catalog_encodings=catalog_encodings,
            catalog_names=catalog_names,
            name_to_idx=name_to_idx,
            alphas=alphas, ks=ks, include_name=include_name,
        )
        if res.get("skipped"):
            print(f"  skipped"); continue
        all_results.append(res)
        V = res["catalog_size"]
        print(f"  V={V}  n_test={res['n_test']}")
        for alpha in alphas:
            r = res["alphas"][alpha]
            line = f"  alpha={alpha:.1f}  n_calls={r['n_calls_in_catalog']}"
            for k in ks:
                acc = r[f"top{k}"]["acc"]
                ratio = r[f"top{k}"]["ratio_vs_random"]
                line += f"   top{k}={acc:.4f} ({ratio:.0f}x)"
            print(line)

    print()
    print("=== summary: top-3 acc per (held_out, alpha) ===")
    header = "  held_out".ljust(28) + "  ".join(f"a={a:.1f}" for a in alphas)
    print(header)
    for res in all_results:
        cells = [f"{res['alphas'][a]['top3']['acc']:.3f}" for a in alphas]
        print(f"  {res['held_out']:26s}  " + "  ".join(cells))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

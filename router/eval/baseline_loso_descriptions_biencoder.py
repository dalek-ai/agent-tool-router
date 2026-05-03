"""LOSO eval, descriptions retrieval, with a sentence-transformer bi-encoder
instead of TF-IDF.

Same setup as baseline_loso_descriptions.py:
  for each held-out source S, the test set is S's traces, and the candidate
  pool is S's tool catalog (filtered to tools that actually appear in test
  with non-empty descriptions). For each (task, gold_tools), we score every
  candidate by cosine(encode(task), encode(description)) and report top-K
  recall over the gold tools.

The hypothesis under test: a pre-trained sentence encoder transfers semantic
similarity across domains better than a TF-IDF vectorizer fitted on a single
training pool. If true, tau-bench held-out (the weak case at 1.5x random for
TF-IDF, because its 23 customer-service tools have low lexical overlap with
the rest of the corpus) should improve substantially. The Hermes/ToolACE
held-out cases already saturate around 73%/35% with TF-IDF, so the more
interesting metric here is the tau-bench delta.

Model: sentence-transformers/all-MiniLM-L6-v2 (default, ~22M params, 80MB).
This is the standard small bi-encoder baseline. If results justify it, a
v0.4 sweep over larger models (mpnet, bge-small) is straightforward.

Run: python3 router/eval/baseline_loso_descriptions_biencoder.py
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


def _encode(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Wrapper around model.encode that always returns L2-normalized
    np.float32 (N, D)."""
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32, copy=False)


def evaluate_held_out(model, rows, descs_by_src, held_out: str,
                      *, include_name: bool = True) -> dict:
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
    catalog = {n: d for n, d in held_out_catalog.items() if n in test_gold_tools}
    catalog = {n: d for n, d in catalog.items() if d}
    if len(catalog) < 5:
        return {"held_out": held_out, "skipped": True,
                "reason": f"catalog too small ({len(catalog)})"}

    catalog_names = sorted(catalog)
    name_to_idx = {n: i for i, n in enumerate(catalog_names)}
    V = len(catalog_names)

    tool_docs = [_tool_text(n, catalog[n], include_name=include_name)
                 for n in catalog_names]
    tool_vecs = _encode(model, tool_docs)

    test_tasks = [r[0] for r in test_rows]
    task_vecs = _encode(model, test_tasks)
    scores = task_vecs @ tool_vecs.T
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
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"sentence-transformers model (default: {DEFAULT_MODEL})")
    args = p.parse_args()
    include_name = not args.no_name

    print(f"[model] loading {args.model}", file=sys.stderr)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    print(f"[model] loaded, dim={model.get_sentence_embedding_dimension()}",
          file=sys.stderr)

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
        res = evaluate_held_out(model, rows, descs_by_src, src,
                                include_name=include_name)
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

    print("=== summary (top-3, bi-encoder description retrieval) ===")
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

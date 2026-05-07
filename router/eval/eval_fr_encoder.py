"""Qualitative FR/EN evaluation: how does an encoder handle French queries
against an English-only tool catalog?

Loads fr_eval_queries.json (parallel EN/FR pairs with expected tool names),
encodes both queries with one or more SentenceTransformer models, computes
top-k against the same 18K-tool catalog (descriptions, source-agnostic),
and reports per-query and aggregate top-k accuracy.

This is a small qualitative probe (n=15), not a benchmark. The point is to
see whether switching to a multilingual encoder makes FR queries usable
without crashing the EN baseline.

Usage:
  python -m router.eval.eval_fr_encoder \\
      --models sentence-transformers/all-MiniLM-L6-v2,sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DESCS_PATH = ROOT / "data" / "tool_descriptions.jsonl"
QUERIES_PATH = ROOT / "router" / "eval" / "fr_eval_queries.json"


def _name_subtokens(name: str) -> str:
    parts = re.split(r"[_\.\s\-]+", name)
    subs: list[str] = []
    for p in parts:
        subs.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", p))
    return " ".join(t.lower() for t in subs if len(t) >= 2)


def _tool_text(name: str, desc: str) -> str:
    return f"{desc} {_name_subtokens(name)}".strip()


def load_descriptions() -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    with DESCS_PATH.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            name = d.get("name")
            desc = (d.get("description") or "").strip()
            if not isinstance(name, str) or not name or name in seen or not desc:
                continue
            seen.add(name)
            rows.append({"name": name, "description": desc})
    return rows


def encode_catalog(model, descs: list[dict]) -> np.ndarray:
    docs = [_tool_text(d["name"], d["description"]) for d in descs]
    return model.encode(
        docs, batch_size=256, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32, copy=False)


def topk(scores: np.ndarray, names: list[str], k: int) -> list[str]:
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [names[i] for i in idx]


_NORM_RE = re.compile(r"[^a-z0-9]+")


def _tokens(s: str) -> set[str]:
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", s)
    if parts:
        return {p.lower() for p in parts if len(p) >= 2}
    return {t for t in _NORM_RE.split(s.lower()) if t}


def _hit_one(returned_name: str, concept: str) -> bool:
    rtokens = _tokens(returned_name)
    ctokens = {t for t in concept.lower().split() if t}
    return ctokens.issubset(rtokens)


def hits_any(returned: list[str], expected_concepts: list[str]) -> bool:
    return any(_hit_one(r, c) for r in returned for c in expected_concepts)


def evaluate(model_name: str, descs: list[dict], queries: list[dict], ks: list[int]) -> dict:
    print(f"\n=== {model_name} ===", flush=True)
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(model_name)
    print(f"  loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    catalog_enc = encode_catalog(model, descs)
    print(f"  catalog ({len(descs)}) encoded in {time.time()-t0:.1f}s",
          file=sys.stderr)
    names = [d["name"] for d in descs]

    en_texts = [q["en"] for q in queries]
    fr_texts = [q["fr"] for q in queries]
    en_enc = model.encode(en_texts, normalize_embeddings=True,
                          convert_to_numpy=True).astype(np.float32, copy=False)
    fr_enc = model.encode(fr_texts, normalize_embeddings=True,
                          convert_to_numpy=True).astype(np.float32, copy=False)
    s_en = en_enc @ catalog_enc.T
    s_fr = fr_enc @ catalog_enc.T

    out = {"model": model_name, "per_query": [], "aggregate": {}}
    for k in ks:
        en_hits = 0
        fr_hits = 0
        for i, q in enumerate(queries):
            en_top = topk(s_en[i], names, k)
            fr_top = topk(s_fr[i], names, k)
            concepts = q.get("expected_concepts", q.get("expected_any", []))
            en_ok = hits_any(en_top, concepts)
            fr_ok = hits_any(fr_top, concepts)
            en_hits += int(en_ok)
            fr_hits += int(fr_ok)
            if k == 3:
                out["per_query"].append({
                    "id": q["id"],
                    "en_top3": en_top,
                    "fr_top3": fr_top,
                    "en_hit": en_ok,
                    "fr_hit": fr_ok,
                })
        n = len(queries)
        out["aggregate"][f"top{k}"] = {
            "en_acc": en_hits / n,
            "fr_acc": fr_hits / n,
            "en_hits": en_hits,
            "fr_hits": fr_hits,
            "n": n,
        }
        print(f"  top{k}:  EN {en_hits}/{n} ({en_hits/n:.0%})   "
              f"FR {fr_hits}/{n} ({fr_hits/n:.0%})")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models",
                   default="sentence-transformers/all-MiniLM-L6-v2,"
                           "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--ks", default="1,3,5")
    p.add_argument("--show-misses", action="store_true",
                   help="print FR misses with returned top-3 for inspection")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ks = [int(k) for k in args.ks.split(",")]

    print(f"[load] descriptions from {DESCS_PATH}", file=sys.stderr)
    descs = load_descriptions()
    print(f"[load] {len(descs)} unique tools", file=sys.stderr)

    with QUERIES_PATH.open(encoding="utf-8") as f:
        queries = json.load(f)["queries"]
    print(f"[load] {len(queries)} parallel EN/FR queries", file=sys.stderr)

    results = []
    for m in models:
        r = evaluate(m, descs, queries, ks)
        results.append(r)

    print("\n=== summary ===")
    for r in results:
        print(f"\n{r['model']}")
        for k, agg in r["aggregate"].items():
            print(f"  {k}:  EN={agg['en_acc']:.2f}  FR={agg['fr_acc']:.2f}  "
                  f"(delta = {agg['fr_acc'] - agg['en_acc']:+.2f})")

    if args.show_misses:
        print("\n=== FR misses per model ===")
        for r in results:
            print(f"\n--- {r['model']} ---")
            for q in r["per_query"]:
                if not q["fr_hit"]:
                    print(f"  [{q['id']}] FR top3: {q['fr_top3']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

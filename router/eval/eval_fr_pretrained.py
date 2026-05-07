"""FR/EN qualitative eval on the *shipped* pretrained models.

Loads each pretrained model via Router.from_pretrained(...) and routes the
fr_eval_queries.json pairs (15 parallel EN/FR queries) against the model's
own catalog. This is the apples-to-apples measurement for the shipped
artifacts (TF-IDF + multilingual encoder hybrid, etc.) — not just the
encoder seul probe in eval_fr_encoder.py.

Usage:
  python -m router.eval.eval_fr_pretrained
  python -m router.eval.eval_fr_pretrained --models baseline-v1-desc-hybrid-multilingual,baseline-v1-desc-hybrid
  python -m router.eval.eval_fr_pretrained --show-misses
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUERIES_PATH = ROOT / "router" / "eval" / "fr_eval_queries.json"

DEFAULT_MODELS = [
    "baseline-v1-desc",
    "baseline-v1-desc-hybrid",
    "baseline-v1-desc-hybrid-multilingual",
]


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


def evaluate(model_name: str, queries: list[dict], ks: list[int]) -> dict:
    from agent_tool_router import Router

    print(f"\n=== {model_name} ===", flush=True)
    t0 = time.time()
    r = Router.from_pretrained(model_name)
    print(f"  loaded in {time.time()-t0:.1f}s  "
          f"(backend={r.backend} alpha={r.alpha} V={len(r.vocab)})",
          file=sys.stderr)

    max_k = max(ks)
    en_top_all: list[list[str]] = []
    fr_top_all: list[list[str]] = []

    t0 = time.time()
    en_top_all = r.route([q["en"] for q in queries], k=max_k)
    fr_top_all = r.route([q["fr"] for q in queries], k=max_k)
    print(f"  routed {2*len(queries)} queries in {time.time()-t0:.2f}s",
          file=sys.stderr)

    out = {"model": model_name, "per_query": [], "aggregate": {}}
    for k in ks:
        en_hits = 0
        fr_hits = 0
        for i, q in enumerate(queries):
            en_top = en_top_all[i][:k]
            fr_top = fr_top_all[i][:k]
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
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--ks", default="1,3,5")
    p.add_argument("--show-misses", action="store_true")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ks = [int(k) for k in args.ks.split(",")]

    with QUERIES_PATH.open(encoding="utf-8") as f:
        queries = json.load(f)["queries"]
    print(f"[load] {len(queries)} parallel EN/FR queries", file=sys.stderr)

    results = []
    for m in models:
        try:
            r = evaluate(m, queries, ks)
        except Exception as e:
            print(f"  ERROR on {m}: {e}", file=sys.stderr)
            continue
        results.append(r)

    print("\n=== summary ===")
    print(f"{'model':<46}  {'top1 EN':>7} {'top1 FR':>7}  "
          f"{'top3 EN':>7} {'top3 FR':>7}  {'top5 EN':>7} {'top5 FR':>7}")
    for r in results:
        a1 = r["aggregate"].get("top1", {})
        a3 = r["aggregate"].get("top3", {})
        a5 = r["aggregate"].get("top5", {})
        print(f"{r['model']:<46}  "
              f"{a1.get('en_acc',0):>7.0%} {a1.get('fr_acc',0):>7.0%}  "
              f"{a3.get('en_acc',0):>7.0%} {a3.get('fr_acc',0):>7.0%}  "
              f"{a5.get('en_acc',0):>7.0%} {a5.get('fr_acc',0):>7.0%}")

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

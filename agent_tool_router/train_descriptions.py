"""Train a description-based Router on data/tool_descriptions.jsonl and save it.

Unlike train.py, which builds centroids by aggregating TF-IDF vectors of tasks
that called each tool (and filters tools by call frequency), this trains a
Router whose centroids are TF-IDF vectors of the tools' own descriptions.
That removes the frequency filter: every tool with a non-empty description is
routable, including singletons that train.py would drop.

Usage:
  python -m agent_tool_router.train_descriptions --out models/baseline-v1-desc

Defaults match Router.from_descriptions(backend="tfidf", include_name=True)
so the saved model behaves identically to that constructor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .router import Router

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS = ROOT / "data" / "tool_descriptions.jsonl"


def _load_descriptions(path: Path):
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            name = (row.get("name") or "").strip()
            desc = (row.get("description") or "").strip()
            if not name or not desc:
                continue
            rows.append((name, desc))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output model directory")
    p.add_argument("--descriptions", default=str(DESCRIPTIONS))
    p.add_argument("--no-name", action="store_true",
                   help="exclude tool name subtokens from the description doc")
    p.add_argument("--max-features", type=int, default=50000)
    p.add_argument("--backend", choices=("tfidf", "encoder", "hybrid"),
                   default="tfidf",
                   help="scoring backend; encoder/hybrid require the "
                        "[encoder] extras (sentence-transformers + torch).")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="hybrid mixing weight on tfidf (default 0.5)")
    p.add_argument("--encoder-model",
                   default="sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args()

    rows = _load_descriptions(Path(args.descriptions))
    print(f"[load] {len(rows)} (name, description) pairs from "
          f"{args.descriptions}", file=sys.stderr)

    # Router.from_descriptions dedupes on name internally and drops empty docs.
    router = Router.from_descriptions(
        rows,
        backend=args.backend,
        alpha=args.alpha,
        encoder_model_name=args.encoder_model,
        include_name=not args.no_name,
        max_features=args.max_features,
    )
    print(f"[vocab] {len(router.vocab)} tools after dedupe (backend="
          f"{router.backend})", file=sys.stderr)

    out = router.save(args.out)
    print(f"[save] model -> {out}", file=sys.stderr)

    samples = [
        "cancel my pending order and refund the credit",
        "translate this text to french",
        "compute the area of a circle with radius 3",
        "what is the latest stock price for apple",
        "look up the user with email alice@example.com",
    ]
    print("[smoke] sample top-3:", file=sys.stderr)
    for q in samples:
        top = router.route(q, k=3)
        print(f"  {q!r}", file=sys.stderr)
        for t in top:
            print(f"    -> {t}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

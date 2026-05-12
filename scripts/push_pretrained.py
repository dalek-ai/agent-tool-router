"""Push the locally-trained pretrained models to HuggingFace Hub under
the ``dalek-ai/`` namespace, so that ``Router.from_pretrained("name")``
works zero-friction for users who clone the repo.

Usage:
    export HF_TOKEN=hf_...
    python scripts/push_pretrained.py                       # push all 4 models (full upload)
    python scripts/push_pretrained.py baseline-v1-desc      # push one only
    python scripts/push_pretrained.py --card-only           # update READMEs only
    python scripts/push_pretrained.py --card-only -- baseline-v1-desc-hybrid

Each model is pushed as its own repo: ``dalek-ai/<name>``. Repos are
created if missing. Existing files are overwritten.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

DEFAULT_MODELS = [
    "baseline-v0",
    "baseline-v1-desc",
    "baseline-v1-desc-hybrid",
    "baseline-v1-desc-hybrid-multilingual",
]

ORG = "dalek-ai"

# YAML frontmatter that pushes us into the right HF discovery surfaces:
#   pipeline_tag: sentence-similarity   (cosine retrieval)
#   tags: agents / tool-routing / function-calling / retrieval / mcp
#   language: en (or [en, fr] for multilingual)
# The body uses fenced code blocks so HF renders the snippets cleanly.
_FRONTMATTER_EN = """\
---
library_name: agent-tool-router
license: mit
pipeline_tag: sentence-similarity
language:
  - en
tags:
  - agents
  - tool-routing
  - function-calling
  - retrieval
  - mcp
---
"""

_FRONTMATTER_MULTI = """\
---
library_name: agent-tool-router
license: mit
pipeline_tag: sentence-similarity
language:
  - en
  - fr
  - multilingual
tags:
  - agents
  - tool-routing
  - function-calling
  - retrieval
  - mcp
  - multilingual
  - french
---
"""

_INSTALL_NOTE = (
    "Install the SDK directly from GitHub (PyPI publish pending):\n"
    "```bash\n"
    "pip install git+https://github.com/dalek-ai/agent-tool-router.git\n"
    "```\n"
)

_DEMO_LINK = (
    "Live demo: [dalek-ai/agent-tool-router-demo](https://huggingface.co/spaces/dalek-ai/agent-tool-router-demo) "
    "(gradio Space, FR/EN)."
)


def card_baseline_v0() -> str:
    return _FRONTMATTER_EN + f"""
# baseline-v0

Centroid retrieval router over **265 tool names** from public agent
benchmarks (tau-bench, Hermes, ToolACE). TF-IDF only, no torch, no GPU,
no API key. Smallest of the four pretrained baselines.

## Quick start

{_INSTALL_NOTE}
```python
from agent_tool_router import Router
r = Router.from_pretrained("baseline-v0")
r.route("cancel my pending order and refund the credit", k=3)
```

`Router.from_pretrained("<name>")` resolves the bare name to
`dalek-ai/<name>` on HuggingFace Hub and caches locally on first call.

## Numbers

Cross-corpus, held-out 2 041 tasks, vocab = 265 tools (≥ 3 occurrences):

| metric | model | random | ratio |
|---|---:|---:|---:|
| top-1 | 33.0% | 0.38% | 87.5× |
| top-3 | 63.8% | 1.13% | 56.4× |
| top-5 | 83.0% | 1.89% | 44.0× |

For the long-tail catalog (18 671 tools) prefer
[`baseline-v1-desc`](https://huggingface.co/dalek-ai/baseline-v1-desc) or
[`baseline-v1-desc-hybrid`](https://huggingface.co/dalek-ai/baseline-v1-desc-hybrid).

## Repo & demo

[github.com/dalek-ai/agent-tool-router](https://github.com/dalek-ai/agent-tool-router) · MIT.
{_DEMO_LINK}
"""


def card_baseline_v1_desc() -> str:
    return _FRONTMATTER_EN + f"""
# baseline-v1-desc

Centroid retrieval router over **18 671 tools**, scored by
`cosine(task_tfidf, description_tfidf)`. TF-IDF only, ~6 MB, no torch,
no GPU. The default model behind `pip install agent-tool-router`.

## Quick start

{_INSTALL_NOTE}
```python
from agent_tool_router import Router
r = Router.from_pretrained("baseline-v1-desc")
r.route("cancel my pending order and refund the credit", k=3)
# ['refundOrder', 'modify_pending_order_items', 'cancel_pending_order']
```

## Numbers

Per-call top-3 against the full 18 671-tool catalog (n=30 425 calls,
held-out across the corpus). Random baseline = 3/V = 0.016%:

| source | n calls | top-3 |
|---|---:|---:|
| Hermes function-calling-v1 | 4 376 | 74.3% |
| ToolACE | 17 169 | 52.4% |
| tau-bench | 8 880 | 3.2% |
| **overall** | **30 425** | **41.2%** |

For Pareto-better top-3 across all three sources at the cost of ~250 MB
of `torch + sentence-transformers`, switch to
[`baseline-v1-desc-hybrid`](https://huggingface.co/dalek-ai/baseline-v1-desc-hybrid).

## Repo & demo

[github.com/dalek-ai/agent-tool-router](https://github.com/dalek-ai/agent-tool-router) · MIT.
{_DEMO_LINK}
"""


def card_baseline_v1_desc_hybrid() -> str:
    return _FRONTMATTER_EN + f"""
# baseline-v1-desc-hybrid

Centroid retrieval router over **18 671 tools**, hybrid scoring:

```
score = 0.5 · cosine(task_tfidf, desc_tfidf) + 0.5 · cosine(task_enc, desc_enc)
```

Encoder = `sentence-transformers/all-MiniLM-L6-v2` (22M params,
English). Encoder is lazy-loaded on the first `route()` call. ~35 MB
of centroids on disk.

## Quick start

{_INSTALL_NOTE}
```bash
pip install "agent-tool-router[encoder] @ git+https://github.com/dalek-ai/agent-tool-router.git"
```

```python
from agent_tool_router import Router
r = Router.from_pretrained("baseline-v1-desc-hybrid")
r.route("cancel my order and refund the credit", k=3)
# ['cancel_pending_order', 'cancel_order', 'refundOrder']
```

## History-aware rerank (≥ 0.3.0)

The model ships a Markov-1 transition table (~21 KB) trained on the
same 7 184 multi-turn traces. Pass the tool names already called in
the trace as `history=` and the top-200 retrieval candidates are
reranked with a learned prior:

```python
r.route(
    "I want to add a checked bag to my reservation",
    k=3,
    history=["update_reservation_flights"],
)
# ['update_reservation_baggages', 'update_reservation_passengers', 'cancel_reservation']
```

Measured lift on n=2094 held-out triplets (split by trace_id, train-only
Markov to avoid prior-on-test leakage; alpha sweep-best per bucket):

| Setup                                  | top-1 | top-3 | top-5 |
|---                                     |---:|---:|---:|
| Retrieval-only                         | 13.8% | 32.7% | 38.8% |
| Markov-1 rerank top-50 (α=0.4)         | 34.6% | 48.0% | 50.5% |
| **Markov-1 rerank top-200 (α=0.1)** ⬅ default ≥ 0.3.0 | **39.0%** | **54.9%** | **57.7%** |

Widening the retrieval bucket from 50 → 200 lifts top-3 by +6.7pp without
any new training: same prior, more candidates available to rerank.
Retrieval recall@K is the mechanical ceiling: recall@50 = 58.9%,
recall@200 = 69.6% on this test split.

No-op when `history` is omitted or empty. Override with
`markov_alpha=0.0` (prior-only) or `1.0` (retrieval-only), and the bucket
width via `markov_rerank_n` (default 200).

## Numbers

Per-call top-3 against the full 18 671-tool catalog (n=30 425 calls):

| source | n calls | tfidf | encoder | **hybrid α=0.5** |
|---|---:|---:|---:|---:|
| Hermes function-calling-v1 | 4 376 | 74.3% | 60.7% | **74.9%** |
| ToolACE | 17 169 | 52.4% | 54.8% | **62.8%** |
| tau-bench | 8 880 | 3.2% | 6.1% | **9.9%** |
| **overall** | **30 425** | 41.2% | 41.4% | **49.1%** |

The hybrid Pareto-dominates both solo backends on every source and every
top-k (+7.9pp overall vs TF-IDF). LOSO refit (TF-IDF retrained per
held-out source, encoder pretrained) loses at most -4.1pp vs the
in-distribution number above.

For routing on French / non-English tasks, see
[`baseline-v1-desc-hybrid-multilingual`](https://huggingface.co/dalek-ai/baseline-v1-desc-hybrid-multilingual).

## Repo & demo

[github.com/dalek-ai/agent-tool-router](https://github.com/dalek-ai/agent-tool-router) · MIT.
{_DEMO_LINK}
"""


def card_baseline_v1_desc_hybrid_multilingual() -> str:
    return _FRONTMATTER_MULTI + f"""
# baseline-v1-desc-hybrid-multilingual

Same hybrid pipeline as
[`baseline-v1-desc-hybrid`](https://huggingface.co/dalek-ai/baseline-v1-desc-hybrid),
but the encoder is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(117M params, 50+ languages). ~80 MB of centroids on disk. Built for
agents that receive tasks in French or any of the 50+ languages
supported by the encoder.

**En français :** ce modèle route une tâche écrite en français vers les
bons outils parmi un catalogue de 18 671 outils anglophones (les
descriptions des outils restent en anglais, l'encoder multilingue
aligne les deux espaces).

## Quick start

{_INSTALL_NOTE}
```bash
pip install "agent-tool-router[encoder] @ git+https://github.com/dalek-ai/agent-tool-router.git"
```

```python
from agent_tool_router import Router
r = Router.from_pretrained("baseline-v1-desc-hybrid-multilingual")
r.route("envoie un message slack à l'équipe data", k=3)
# ['post_message_to_slack', 'sendSlackMessage', ...]
r.route("traduis ce document en anglais", k=3)
```

## Numbers

**Parallel EN/FR probe (n=50 hand-written queries against the full
18 671-tool catalog, top-3 per call):**

| model | EN top-3 | FR top-3 | EN top-5 | FR top-5 |
|---|---:|---:|---:|---:|
| baseline-v1-desc-hybrid (default English encoder) | 82% | 26% | 90% | 30% |
| **baseline-v1-desc-hybrid-multilingual** | **82%** | **54%** | 90% | 62% |

**+28pp top-3 on French queries with no measurable cost on English
top-3** at this panel size. Reproduce:
`python -m router.eval.eval_fr_pretrained`.

**English-only benchmark (LOSO refit, full catalog, top-3):** the
multilingual encoder trails the default English-only encoder by ~3.9pp
weighted overall (Hermes -8.8pp, ToolACE -3.9pp, tau-bench -1.5pp).
Prefer the default model if all your queries are English.

## Repo & demo

[github.com/dalek-ai/agent-tool-router](https://github.com/dalek-ai/agent-tool-router) · MIT.
{_DEMO_LINK}
"""


CARDS = {
    "baseline-v0": card_baseline_v0,
    "baseline-v1-desc": card_baseline_v1_desc,
    "baseline-v1-desc-hybrid": card_baseline_v1_desc_hybrid,
    "baseline-v1-desc-hybrid-multilingual": card_baseline_v1_desc_hybrid_multilingual,
}


def push_full(name: str, token: str) -> None:
    from huggingface_hub import HfApi, create_repo

    local = MODELS_DIR / name
    if not local.exists():
        sys.exit(f"error: {local} does not exist. Train it first.")

    repo_id = f"{ORG}/{name}"
    print(f"creating repo {repo_id} (idempotent)...")
    create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model")

    card_fn = CARDS.get(name)
    if card_fn is None:
        sys.exit(f"error: no card defined for {name}")
    (local / "README.md").write_text(card_fn(), encoding="utf-8")

    api = HfApi(token=token)
    print(f"uploading {local} -> {repo_id} ...")
    api.upload_folder(
        folder_path=str(local),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"upload {name}",
    )
    print(f"  -> https://huggingface.co/{repo_id}")


def push_card_only(name: str, token: str) -> None:
    from huggingface_hub import HfApi, create_repo

    repo_id = f"{ORG}/{name}"
    print(f"refreshing card for {repo_id} ...")
    create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model")

    card_fn = CARDS.get(name)
    if card_fn is None:
        sys.exit(f"error: no card defined for {name}")

    api = HfApi(token=token)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(card_fn())
        tmp_path = f.name

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="docs: refresh model card",
    )
    os.unlink(tmp_path)
    print(f"  -> https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "names",
        nargs="*",
        help="Model names to push (default: all four baselines).",
    )
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="Only update README.md (no model artifact upload).",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit(
            "error: set HF_TOKEN to a HuggingFace write token before running."
        )

    names = args.names if args.names else DEFAULT_MODELS
    for n in names:
        if args.card_only:
            push_card_only(n, token)
        else:
            push_full(n, token)


if __name__ == "__main__":
    main()

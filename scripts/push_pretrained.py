"""Push the locally-trained pretrained models to HuggingFace Hub under
the ``dalek-ai/`` namespace, so that ``Router.from_pretrained("name")``
works zero-friction for users who clone the repo.

Usage:
    export HF_TOKEN=hf_...
    python scripts/push_pretrained.py            # push all 3 default models
    python scripts/push_pretrained.py baseline-v1-desc-hybrid  # one only

Each model is pushed as its own repo: ``dalek-ai/<name>``. Repos are
created if missing. Existing files are overwritten.
"""

from __future__ import annotations

import argparse
import os
import sys
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

MODEL_CARDS = {
    "baseline-v0": (
        "Centroid retrieval router over 265 tool names from public agent "
        "benchmarks (tau-bench, Hermes, ToolACE). TF-IDF only, ~1 MB. "
        "Smallest of the three pretrained baselines. See "
        "https://github.com/dalek-ai/agent-tool-router for details."
    ),
    "baseline-v1-desc": (
        "Centroid retrieval router over 18 671 tools, scored by cosine on "
        "natural-language descriptions. TF-IDF only, ~6 MB, no extra "
        "dependencies. Top-3 overall = 41.2% on a held-out 30 425-call "
        "split. See https://github.com/dalek-ai/agent-tool-router."
    ),
    "baseline-v1-desc-hybrid": (
        "Centroid retrieval router over 18 671 tools, hybrid scoring "
        "``0.5 * cos_tfidf + 0.5 * cos_encoder`` (sentence-transformers "
        "MiniLM-L6-v2). ~35 MB. Top-3 overall = 49.1% on a held-out 30 "
        "425-call split, Pareto-dominates both solo backends. Requires "
        "``pip install agent-tool-router[encoder]`` for inference. See "
        "https://github.com/dalek-ai/agent-tool-router."
    ),
    "baseline-v1-desc-hybrid-multilingual": (
        "Same hybrid pipeline as ``baseline-v1-desc-hybrid``, but the "
        "encoder is ``paraphrase-multilingual-MiniLM-L12-v2`` (50+ "
        "languages). On a 15-query parallel EN/FR probe, FR top-3 jumps "
        "from 27% (default English-only encoder) to 67%, while EN top-3 "
        "stays at 80%. On the full LOSO refit benchmark (English) it is "
        "~3.9pp behind the default model overall, so prefer the default "
        "if your queries are all in English. Requires ``pip install "
        "agent-tool-router[encoder]``. See "
        "https://github.com/dalek-ai/agent-tool-router."
    ),
}


def push_one(name: str, token: str) -> None:
    from huggingface_hub import HfApi, create_repo

    local = MODELS_DIR / name
    if not local.exists():
        sys.exit(f"error: {local} does not exist. Train it first.")

    repo_id = f"{ORG}/{name}"
    print(f"creating repo {repo_id} (idempotent)...")
    create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model")

    card = MODEL_CARDS.get(name, f"agent-tool-router pretrained: {name}")
    readme = local / "README.md"
    readme.write_text(
        f"---\nlibrary_name: agent-tool-router\nlicense: mit\n---\n\n# {name}\n\n{card}\n",
        encoding="utf-8",
    )

    api = HfApi(token=token)
    print(f"uploading {local} -> {repo_id} ...")
    api.upload_folder(
        folder_path=str(local),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"upload {name}",
    )
    print(f"  -> https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "names",
        nargs="*",
        help="Model names to push (default: all three baselines).",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit(
            "error: set HF_TOKEN to a HuggingFace write token before running."
        )

    names = args.names if args.names else DEFAULT_MODELS
    for n in names:
        push_one(n, token)


if __name__ == "__main__":
    main()

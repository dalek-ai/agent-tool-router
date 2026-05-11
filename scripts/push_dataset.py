"""Push the parallel EN/FR eval queries as a dataset on HuggingFace Hub
under dalek-ai/agent-tool-router-eval-fr.

Usage:
    HF_TOKEN=$(cat .hf_token_tmp) python scripts/push_dataset.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "datasets" / "eval_fr"
DATASET_REPO = "dalek-ai/agent-tool-router-eval-fr"


def main() -> None:
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("error: set HF_TOKEN to a HuggingFace write token before running.")

    if not DATASET_DIR.exists():
        sys.exit(f"error: {DATASET_DIR} does not exist.")

    print(f"creating dataset {DATASET_REPO} (idempotent)...")
    create_repo(
        repo_id=DATASET_REPO,
        token=token,
        exist_ok=True,
        repo_type="dataset",
    )

    api = HfApi(token=token)
    print(f"uploading {DATASET_DIR} -> {DATASET_REPO} ...")
    api.upload_folder(
        folder_path=str(DATASET_DIR),
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="publish parallel EN/FR eval queries (n=50)",
    )
    print(f"  -> https://huggingface.co/datasets/{DATASET_REPO}")


if __name__ == "__main__":
    main()

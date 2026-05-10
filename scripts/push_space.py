"""Push the gradio demo Space to HuggingFace under dalek-ai/agent-tool-router-demo.

Usage:
    export HF_TOKEN=hf_...
    python scripts/push_space.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACE_DIR = REPO_ROOT / "spaces" / "demo"
SPACE_REPO = "dalek-ai/agent-tool-router-demo"


def main() -> None:
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("error: set HF_TOKEN to a HuggingFace write token before running.")

    if not SPACE_DIR.exists():
        sys.exit(f"error: {SPACE_DIR} does not exist.")

    print(f"creating space {SPACE_REPO} (idempotent, sdk=gradio)...")
    create_repo(
        repo_id=SPACE_REPO,
        token=token,
        exist_ok=True,
        repo_type="space",
        space_sdk="gradio",
    )

    api = HfApi(token=token)
    print(f"uploading {SPACE_DIR} -> {SPACE_REPO} ...")
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=SPACE_REPO,
        repo_type="space",
        commit_message="deploy gradio demo",
    )
    print(f"  -> https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
